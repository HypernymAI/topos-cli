# topos/channel/debatesim.py
import hashlib

from typing import Dict, List

import os
import threading
from queue import Queue

from datetime import datetime, timedelta
import time

from dotenv import load_dotenv

from uuid import uuid4

import json
import jwt
from jwt.exceptions import InvalidTokenError

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import entropy

from fastapi import WebSocket, WebSocketDisconnect
from ..FC.argument_detection import ArgumentDetection
from ..config import get_openai_api_key
from ..models.llm_classes import vision_models
from ..generations.ollama_chat import stream_chat
from ..services.database.app_state import AppState
from ..utilities.utils import create_conversation_string
from ..services.classification_service.base_analysis import base_text_classifier, base_token_classifier
from topos.FC.conversation_cache_manager import ConversationCacheManager
from topos.FC.semantic_compression import SemanticCompression
from topos.FC.ontological_feature_detection import OntologicalFeatureDetection

# chess is more complicated than checkers but less complicated than go

# current:
# graph LR
#     timestamp["Timestamp: 2024-06-08T23:47:36.059626"]
#     user["user (USER)"]
#     sessionTEMP["sessionTEMP (SESSION)"]
#     userPRIME["userPRIME (USER)"]
#     than --> checkers
#     sessionTEMP --> of
#     checkers --> complicated
#     message --> is
#     userPRIME --> for
#     is --> chess
#     is --> message
#     checkers --> than
#     of --> sessionTEMP
#     chess --> is
#     for --> userPRIME
#     complicated --> is
#     timestamp --> user

# target:
# graph LR
#     userPRIME["userPRIME (USER)"]
#     sessionTEMP["sessionTEMP (SESSION)"]
#     timestamp["Timestamp: 2024-06-08T23:18:05.206590"]
#     message["message"]
#     chess["chess"]
#     more_complicated["more complicated"]
#     checkers["checkers"]
#     less_complicated["less complicated"]
#     go["go"]
#
#     userPRIME --> user
#     sessionTEMP --> session
#     timestamp --> user
#     message --> userPRIME
#     message --> sessionTEMP
#     message --> timestamp
#     chess --> message
#     more_complicated --> chess
#     more_complicated --> checkers
#     less_complicated --> chess
#     less_complicated --> go


class Cluster:
    def __init__(self, cluster_id, sentences, user_id, generation, session_id):
        self.cluster_id = cluster_id
        self.sentences = sentences
        self.cluster_hash = self.generate_hash()
        self.user_id = user_id
        self.generation = generation
        self.session_id = session_id

    def generate_hash(self):
        sorted_sentences = sorted(self.sentences)
        return hashlib.sha256(json.dumps(sorted_sentences).encode()).hexdigest()

    def to_dict(self):
        return {
            "cluster_id": self.cluster_id,
            "sentences": self.sentences,
            "cluster_hash": self.cluster_hash,
            "user_id": self.user_id,
            "generation": self.generation,
            "session_id": self.session_id
        }


class DebateSimulator:
    _instance = None
    _lock = threading.Lock()

    @staticmethod
    def get_instance():
        if DebateSimulator._instance is None:
            with DebateSimulator._lock:
                if DebateSimulator._instance is None:
                    DebateSimulator._instance = DebateSimulator()
        return DebateSimulator._instance

    def __init__(self, use_neo4j=False):
        if DebateSimulator._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            # Load the pre-trained model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model = AutoModel.from_pretrained('bert-base-uncased')

            self.operational_llm_model = "ollama:dolphin-llama3"

            # Initialize the SentenceTransformer model for embedding text
            self.fast_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.argument_detection = ArgumentDetection(model=self.operational_llm_model, api_key="ollama")

            self.semantic_compression = SemanticCompression(model=self.operational_llm_model, api_key="ollama")

            self.app_state = AppState.get_instance()

            load_dotenv()  # Load environment variables

            neo4j_uri = os.getenv("NEO4J_URI")
            neo4j_user = os.getenv("NEO4J_USER")
            neo4j_password = os.getenv("NEO4J_PASSWORD")
            self.showroom_db_name = os.getenv("NEO4J_SHOWROOM_DATABASE")
            self.use_neo4j = use_neo4j

            # self.cache_manager = ConversationCacheManager()
            self.ontological_feature_detection = OntologicalFeatureDetection(neo4j_uri, neo4j_user, neo4j_password,
                                                                             self.showroom_db_name, self.use_neo4j)

            # JWT secret key (should be securely stored, e.g., in environment variables)
            self.jwt_secret = os.getenv("JWT_SECRET")

            self.current_generation = None

            self.task_queue = Queue()
            self.processing_thread = threading.Thread(target=self.process_tasks)
            self.processing_thread.daemon = True
            self.processing_thread.start()

    def generate_jwt_token(self, user_id, session_id):
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "exp": datetime.utcnow() + timedelta(hours=1)  # Token valid for 1 hour
        }
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        return token

    def add_task(self, task):
        self.task_queue.put(task)

    def process_tasks(self):
        while True:
            task = self.task_queue.get()
            if task['type'] == 'reset':
                self.reset_processing_queue()
            else:
                self.execute_task(task)
            self.task_queue.task_done()

    def reset_processing_queue(self):
        # Logic to reset the processing queue
        while not self.task_queue.empty():
            self.task_queue.get()
            self.task_queue.task_done()

        self.current_generation = None
        print("Processing queue has been reset.")

    def execute_task(self, task):
        # Process the task based on its type and data
        if task['type'] == 'check_and_reflect':
            self.current_generation = task['generation_nonce']
            self.check_and_reflect(task['session_id'], task['user_id'], task['generation_nonce'], task['message_id'], task['message'])
        elif task['type'] == 'broadcast':
            self.start_broadcast_subprocess(task['websocket'], task['message'])
        # Add other task types as needed
        print(f"Executed task: {task['type']}")

    @staticmethod
    def websocket_broadcast(websocket, message):
        while True:
            if message:  # Condition to broadcast
                websocket.send(message)
            time.sleep(1)  # Adjust as necessary

    # Function to start the subprocess
    def start_broadcast_subprocess(self, websocket, message):
        broadcast_thread = threading.Thread(target=self.websocket_broadcast, args=(websocket, message))
        broadcast_thread.start()

    def stop_all_reflect_tasks(self):
        self.add_task({'type': 'reset'})

    def get_ontology(self, user_id, session_id, message_id, message):
        composable_string = f"for user {user_id}, of {session_id}, the message is: {message}"
        print(f"\t\t[ composable_string :: {composable_string} ]")

        entities, pos_tags, dependencies, relations, srl_results, timestamp, context_entities = self.ontological_feature_detection.build_ontology_from_paragraph(
            user_id, session_id, message_id, composable_string)

        if self.use_neo4j:
            self.ontological_feature_detection.store_ontology(user_id, session_id, message_id, message, timestamp, context_entities, relations)

        input_components = message, entities, dependencies, relations, srl_results, timestamp, context_entities

        mermaid_syntax = self.ontological_feature_detection.extract_mermaid_syntax(input_components, input_type="components")
        return mermaid_syntax

    def has_message_id(self, message_id):
        if self.use_neo4j:
            return self.ontological_feature_detection.check_message_exists(message_id)
        else:
            return False


    # @note: integrate is post, due to constant
    async def integrate(self, token, data, app_state):
        payload = json.loads(data)
        message = payload["message"]

        # create a new message id, with 36 characters max
        message_id = str(uuid4())

        # check for collisions
        while self.has_message_id(message_id):
            # re-roll a new message id, with 36 characters max
            message_id = str(uuid4())

        # Decode JWT token to extract user_id and session_id
        try:
            decoded_token = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            user_id = decoded_token.get("user_id", "")
            session_id = decoded_token.get("session_id", "")
        except InvalidTokenError:
            # await websocket.send_json({"status": "error", "response": "Invalid JWT token"})
            return

        # if no user_id, bail
        if user_id == "" or session_id == "":
            return

        current_topic = payload.get("topic", "Unknown")

        # from app state
        message_history = app_state.get_value(f"message_history_{session_id}", [])

        prior_ontology = app_state.get_value(f"prior_ontology_{session_id}", [])

        current_ontology = self.get_ontology(user_id, session_id, message_id, message)

        print(f"[ prior_ontology: {prior_ontology} ]")
        print(f"[ current_ontology: {current_ontology} ]")

        prior_ontology.append(current_ontology)

        app_state.set_state(f"prior_ontology{session_id}_", prior_ontology)

        mermaid_to_ascii = self.ontological_feature_detection.mermaid_to_ascii(current_ontology)
        print(f"[ mermaid_to_ascii: {mermaid_to_ascii} ]")

        message_history.append(message)

        app_state.set_value(f"message_history_{session_id}", message_history)

        # Create new Generation
        generation_nonce = self.generate_nonce()

        self.stop_all_reflect_tasks()

        self.add_task({
            'type': 'check_and_reflect',
            'session_id': session_id,
            'user_id': user_id,
            'generation_nonce': generation_nonce,
            'message_id': message_id,
            'message': message}
        )

        return current_ontology

    @staticmethod
    def generate_nonce():
        return str(uuid4())

    @staticmethod
    def aggregate_user_messages(message_history: List[Dict]) -> Dict[str, List[str]]:
        user_messages = {}
        for message in message_history:
            user_id = message['data']['user_id']
            content = message['data']['content']
            if user_id not in user_messages:
                user_messages[user_id] = []
            user_messages[user_id].append(content)
        return user_messages

    def incremental_clustering(self, clusters, previous_clusters):
        updated_clusters = {}
        for user_id, user_clusters in clusters.items():
            if user_id not in previous_clusters:
                updated_clusters[user_id] = user_clusters
            else:
                updated_clusters[user_id] = {}
                for cluster_id, cluster in user_clusters.items():
                    previous_cluster_hash = previous_clusters[user_id].get(cluster_id, None)
                    if not previous_cluster_hash or cluster.cluster_hash != previous_cluster_hash.cluster_hash:
                        updated_clusters[user_id][cluster_id] = cluster

        return updated_clusters

    @staticmethod
    async def broadcast_to_websocket_group(websocket_group, json_message):
        for websocket in websocket_group:
            await websocket.send_json(json_message)

    def check_generation_halting(self, generation_nonce):
        if self.current_generation is not None and self.current_generation != generation_nonce:
            return True

        return False

    async def check_and_reflect(self, session_id, user_id, generation_nonce, message_id, message):
        # "Reflect"
        # cluster message callback
        # each cluster is defined by a cluster id (a hash of its messages, messages sorted alphabetically)

        # 1. early out if the cluster is identical
        # 2. total message completion is based on all messages (a generation)
        # 3. previous generations DO NOT complete - they are halted upon a new message
        # 4. clustering will not be affected by other Users if their message has not changed, but generations
        #    always will because every new message from another player is dealt with re: claims/counterclaims
        # 5. technically, each generation has a final score (though because of processing reqs, we're not expecting
        #    to have more than each generation w/ a final score, technically this can be done as well, but
        #    it's probably not germane to the convo needs, so let's just not)

        # prioritize wepcc (warrant evidence persuasiveness/justification claim counterclaim) for the user's cluster

        app_state = AppState().get_instance()

        message_history = app_state.get_value(f"message_history_{session_id}", [])

        # Step 1: Gather message history for specific users
        user_messages = self.aggregate_user_messages(message_history)
        print(f"\t[ reflect :: user_messages :: {user_messages} ]")

        # Step 2: Cluster analysis for each user's messages
        clusters = self.cluster_messages(user_messages, generation_nonce, session_id)
        print(f"\t[ reflect :: clustered_messages :: {clusters} ]")

        websocket_group = app_state.get_value(f"websocket_group_{session_id}", [])

        # Send initial cluster data back to frontend
        await self.broadcast_to_websocket_group(websocket_group, {
            "status": "initial_clusters",
            "clusters": {user_id: [cluster.to_dict() for cluster in user_clusters.values()] for user_id, user_clusters
                         in clusters.items()},
            "generation": generation_nonce
        })
        if self.check_generation_halting(generation_nonce) is True:
            return

        # Perform incremental clustering if needed
        previous_clusters = app_state.get_value(f"previous_clusters_{session_id}", {})
        updated_clusters = self.incremental_clustering(clusters, previous_clusters)
        app_state.set_value(f"previous_clusters_{session_id}", clusters)

        # Send updated cluster data back to frontend
        await self.broadcast_to_websocket_group(websocket_group, {
            "status": "updated_clusters",
            "clusters": {user_id: [cluster.to_dict() for cluster in user_clusters.values()] for user_id, user_clusters
                         in updated_clusters.items()},
            "generation": generation_nonce
        })
        if self.check_generation_halting(generation_nonce) is True:
            return

        async def report_wepcc_result(generation_nonce, user_id, cluster_id, cluster_hash, wepcc_result):
            await self.broadcast_to_websocket_group(websocket_group, {
                "status": "wepcc_result",
                "generation": generation_nonce,
                "user_id": user_id,
                "cluster_id": cluster_id,
                "cluster_hash": cluster_hash,
                "wepcc_result": wepcc_result,
            })
            if self.check_generation_halting(generation_nonce) is True:
                return

        # Step 3: Run WEPCC on each cluster
        # these each take a bit to process, so we're passing in the websocket group to stream the results back out
        # due to timing these may be inconsequential re: generation, but they're going to send back the results anyhow.
        wepcc_results = self.wepcc_cluster(updated_clusters, report_wepcc_result)
        print(f"\t[ reflect :: wepcc_results :: {wepcc_results} ]")

        if len(clusters) < 2 or any(len(user_clusters) < 1 for user_clusters in clusters.values()):
            print("\t[ reflect :: Not enough clusters or users to perform argument matching ]")
            return

        # Define similarity cutoff threshold
        cutoff = 0.5

        # Define unaddressed score multiplier
        unaddressed_score_multiplier = 2.5

        # Initialize phase similarity and cluster weight modulator
        # Step 4: Match each user's Counterclaims with all other users' Claims
        # This function takes a moment, as it does an embedding check. Not super heavy, but with enough participants
        # certainly an async operation
        cluster_weight_modulator = self.get_cluster_weight_modulator(wepcc_results, cutoff)

        # Step 5: Calculate the counter-factual shadow coverage for each cluster
        # Create a new dictionary to hold the final combined scores
        # This function is very fast, relatively speaking
        cluster_shadow_coverage = self.get_cluster_shadow_coverage(cluster_weight_modulator, cutoff)

        # Step 6: Final aggregation and ranking
        (aggregated_scores,
         addressed_clusters,
         unaddressed_clusters,
         results) = self.gather_final_results(cluster_shadow_coverage, wepcc_results, unaddressed_score_multiplier)

        print(f"\t[ reflect :: aggregated_scores :: {aggregated_scores} ]")
        print(f"\t[ reflect :: addressed_clusters :: {addressed_clusters} ]")
        print(f"\t[ reflect :: unaddressed_clusters :: {unaddressed_clusters} ]")

        app_state.set_state("wepcc_results", wepcc_results)
        app_state.set_state("aggregated_scores", aggregated_scores)
        app_state.set_state("addressed_clusters", addressed_clusters)
        app_state.set_state("unaddressed_clusters", unaddressed_clusters)

        print(f"\t[ reflect :: Completed ]")

        return results

    def cluster_messages(self, user_messages, generation, session_id):
        clustered_messages = {}
        for user_id, messages in user_messages.items():
            if len(messages) > 1:
                clusters = self.argument_detection.cluster_sentences(messages, distance_threshold=1.45)
                clustered_messages[user_id] = {
                    cluster_id: Cluster(user_id, generation, session_id, cluster_id, cluster_sentences)
                    for cluster_id, cluster_sentences in clusters.items()
                }
        return clustered_messages

    def wepcc_cluster(self, clusters: Dict[str, Cluster], report_wepcc_result):
        wepcc_results = {}
        for user_id, user_clusters in clusters.items():
            wepcc_results[user_id] = {}
            for cluster_id, cluster in user_clusters.items():
                print(f"\t[ reflect :: Running WEPCC for user {user_id}, cluster {cluster_id} ]")
                warrant, evidence, persuasiveness_justification, claim, counterclaim = self.argument_detection.fetch_argument_definition(
                    cluster.sentences)
                wepcc_results[user_id][cluster_id] = {
                    'warrant': warrant,
                    'evidence': evidence,
                    'persuasiveness_justification': persuasiveness_justification,
                    'claim': claim,
                    'counterclaim': counterclaim
                }
                print(
                    f"\t[ reflect :: WEPCC for user {user_id}, cluster {cluster_id} :: {wepcc_results[user_id][cluster_id]} ]")

                # Output to websocket
                report_wepcc_result(cluster.cluster_hash, user_id, cluster_id, cluster.cluster_hash,
                                    wepcc_results[user_id][cluster_id])
        return wepcc_results

    def get_cluster_weight_modulator(self, wepcc_results, cutoff):
        cluster_weight_modulator = {}
        for user_idA, clustersA in wepcc_results.items():
            cluster_weight_modulator[user_idA] = cluster_weight_modulator.get(user_idA, {})

            for cluster_idA, wepccA in clustersA.items():
                phase_sim_A = []
                for user_idB, clustersB in wepcc_results.items():
                    if user_idA != user_idB:
                        for cluster_idB, wepccB in clustersB.items():
                            # Calculate cosine similarity between counterclaims and claims
                            counterclaim_embedding = self.fast_embedding_model.encode(wepccA['counterclaim'])
                            claim_embedding = self.fast_embedding_model.encode(wepccB['claim'])
                            sim_score = cosine_similarity([counterclaim_embedding], [claim_embedding])[0][0]
                            print(
                                f"\t[ reflect :: Sim score between {user_idA}'s counterclaim (cluster {cluster_idA}) and {user_idB}'s claim (cluster {cluster_idB}) :: {sim_score} ]")
                            if sim_score > cutoff:
                                phase_sim_A.append((sim_score, cluster_idB, user_idB))
                if cluster_idA not in cluster_weight_modulator[user_idA]:
                    cluster_weight_modulator[user_idA][cluster_idA] = []
                for sim_score, cluster_idB, user_idB in phase_sim_A:
                    normalized_value = (sim_score - cutoff) / (1 - cutoff)
                    cluster_weight_modulator[user_idA][cluster_idA].append(normalized_value)
                    print(
                        f"\t[ reflect :: Normalized value for {user_idA} (cluster {cluster_idA}) :: {normalized_value} ]")
        return cluster_weight_modulator

    def gather_final_results(self, cluster_shadow_coverage, wepcc_results, unaddressed_score_multiplier):
        aggregated_scores = {}
        addressed_clusters = {}
        unaddressed_clusters = {}

        results = []

        for user_id, weight_mods in cluster_shadow_coverage.items():
            total_score = 0
            addressed_clusters[user_id] = []
            unaddressed_clusters[user_id] = []

            user_result = {"user": user_id, "clusters": []}

            for cluster_id, modulator in weight_mods.items():
                try:
                    persuasiveness_object = json.loads(
                        wepcc_results[user_id][cluster_id]['persuasiveness_justification'])
                    persuasiveness_score = float(persuasiveness_object['content']['persuasiveness_score'])
                    addressed_score = (1 - modulator) * persuasiveness_score
                    total_score += addressed_score
                    addressed_clusters[user_id].append((cluster_id, addressed_score))
                    user_result["clusters"].append({
                        "cluster": cluster_id,
                        "type": "addressed",
                        "score": addressed_score
                    })
                    print(
                        f"\t[ reflect :: Addressed score for User {user_id}, Cluster {cluster_id} :: {addressed_score} ]")
                except json.JSONDecodeError as e:
                    print(f"\t[ reflect :: JSONDecodeError for User {user_id}, Cluster {cluster_id} :: {e} ]")
                    print(
                        f"\t[ reflect :: Invalid JSON :: {wepcc_results[user_id][cluster_id]['persuasiveness_justification']} ]")

            # Add unaddressed arguments' scores
            for cluster_id, wepcc in wepcc_results[user_id].items():
                if cluster_id not in weight_mods:
                    try:
                        persuasiveness_object = json.loads(wepcc['persuasiveness_justification'])
                        persuasiveness_score = float(persuasiveness_object['content']['persuasiveness_score'])
                        unaddressed_score = persuasiveness_score * unaddressed_score_multiplier
                        total_score += unaddressed_score
                        unaddressed_clusters[user_id].append((cluster_id, unaddressed_score))
                        user_result["clusters"].append({
                            "cluster": cluster_id,
                            "type": "unaddressed",
                            "score": unaddressed_score
                        })
                        print(
                            f"\t[ reflect :: Unaddressed score for User {user_id}, Cluster {cluster_id} :: {unaddressed_score} ]")
                    except json.JSONDecodeError as e:
                        print(f"\t[ reflect :: JSONDecodeError for User {user_id}, Cluster {cluster_id} :: {e} ]")
                        print(f"\t[ reflect :: Invalid JSON :: {wepcc['persuasiveness_justification']} ]")

            aggregated_scores[user_id] = total_score
            user_result["total_score"] = total_score
            results.append(user_result)
            print(f"\t[ reflect :: Aggregated score for User {user_id} :: {total_score} ]")

        return aggregated_scores, addressed_clusters, unaddressed_clusters, results

    def get_cluster_shadow_coverage(self, cluster_weight_modulator, cutoff):
        final_scores = {}

        # Post-process the collected normalized values for each cluster
        for user_id, cluster_data in cluster_weight_modulator.items():
            final_scores[user_id] = final_scores.get(user_id, {})
            for cluster_idA, normalized_values in cluster_data.items():
                if normalized_values:
                    highest = max(normalized_values)
                    shadow_coverage = highest
                    for value in normalized_values:
                        if value != highest:
                            shadow_coverage += (value * (1.0 - cutoff)) * (1 - shadow_coverage)
                            # Since we're adding coverage, shadow_coverage should naturally stay within [0,1]
                            # No need to clamp or use min

                    # Initialize the nested dictionary if it doesn't exist
                    if cluster_idA not in final_scores[user_id]:
                        final_scores[user_id][cluster_idA] = 0

                    # Store the final score
                    final_scores[user_id][cluster_idA] = shadow_coverage
                    print(
                        f"\t[ reflect :: Combined score for {user_id} (cluster {cluster_idA}) :: {shadow_coverage} ]")

        return final_scores
