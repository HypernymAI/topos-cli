from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime
import time

from ..generations.ollama_chat import stream_chat
from topos.FC.semantic_compression import SemanticCompression
from ..config import get_openai_api_key
from ..models.llm_classes import vision_models
import json

from ..services.classification_service.base_analysis import base_text_classifier, base_token_classifier
router = APIRouter()

# cache database
from topos.FC.conversation_cache_manager import ConversationCacheManager


@router.websocket("/websocket_chat")
async def chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            conversation_id = payload["conversation_id"]
            message_id = payload["message_id"]
            chatbot_msg_id = payload["chatbot_msg_id"]
            message = payload["message"]
            message_history = payload["message_history"]
            model = payload.get("model", "solar")
            temperature = float(payload.get("temperature", 0.04))
            current_topic = payload.get("topic", "Unknown")
            processing_config = payload.get("processing_config", {})

            # Set default values if any key is missing or if processing_config is None
            default_config = {
                "showInMessageNER": True,
                "calculateInMessageNER": True,
                "showModerationTags": True,
                "calculateModerationTags": True,
                "showSidebarBaseAnalytics": True
            }

            # Update default_config with provided processing_config, if any
            config = {**default_config, **processing_config}

            # Set system prompt
            has_topic = False
            
            if current_topic != "Unknown":
                has_topic = True
                prompt = f"You are a smooth talking, eloquent, poignant, insightful AI moderator. The current topic is {current_topic}.\n"

            system_prompt = f"You are a smooth talking, eloquent, poignant, insightful AI moderator. The current topic is unknown, so try not to make any judgements thus far - only re-express the input words in your own style:"
            user_prompt = ""
            if message_history:
                # Add the message history prior to the message
                user_prompt += '\n'.join(msg['role'] + ": " + msg['content'] for msg in message_history)

            # print(f"\t[ system prompt :: {system_prompt} ]")
            # print(f"\t[ user prompt :: {user_prompt} ]")
            simp_msg_history = [{'role': 'system', 'content': system_prompt}]

            # Simplify message history to required format
            # If user uses a vision model, load images, else don't
            isVisionModel = model in vision_models
            print(f"\t[ using model :: {model} :: 🕶️  isVision ]") if isVisionModel else print(f"\t[ using model :: {model} ]")  
            
            for message in message_history:
                simplified_message = {'role': message['role'], 'content': message['content']}
                if 'images' in message and isVisionModel:
                    simplified_message['images'] = message['images']
                simp_msg_history.append(simplified_message)
            

            last_message = simp_msg_history[-1]['content']
            # Fetch base, per-message token classifiers
            if config['calculateInMessageNER']:
                start_time = time.time()
                base_analysis = base_token_classifier(last_message)  # this is only an ner dict atm
                duration = time.time() - start_time
                print(f"\t[ base_token_classifier duration: {duration:.4f} seconds ]")
            
            # Fetch base, per-message text classifiers
            # Start timer for base_text_classifier
            if config['calculateModerationTags']:
                start_time = time.time()
                text_classifiers = {}
                try:
                    text_classifiers = base_text_classifier(last_message)
                except Exception as e:
                    logging.error(f"Failed to compute base_text_classifier: {cache_path}: {e}")
                duration = time.time() - start_time
                print(f"\t[ base_text_classifier duration: {duration:.4f} seconds ]")
            
            conv_cache_manager = ConversationCacheManager()
            if config['calculateModerationTags'] or config['calculateInMessageNER']:
                print(f"\t[ save to conv cache :: conversation {conversation_id}-{message_id} ]")
                dummy_data = {
                    message_id : 
                        {
                        'message': last_message, 
                        'timestamp': datetime.now(), 
                    }}
                if config['calculateInMessageNER']:
                    dummy_data[message_id]['in_line'] = {'base_analysis': base_analysis}
                if config['calculateModerationTags']:
                    dummy_data[message_id]['commenter'] = {'base_analysis': text_classifiers}

                conv_cache_manager.save_to_cache(conversation_id, dummy_data)
                # Removing the keys from the nested dictionary
                if message_id in dummy_data:
                    dummy_data[message_id].pop('message', None)
                    dummy_data[message_id].pop('timestamp', None)
                # Sending first batch of user message analysis back to the UI
                await websocket.send_json({"status": "fetched_user_analysis", 'user_message': dummy_data})
            else:
                # Saving an empty dictionary for the messag id
                conv_cache_manager.save_to_cache(conversation_id, {
                    message_id : 
                        {
                        'message': last_message, 
                        'timestamp': datetime.now(), 
                    }})

            # Processing the chat
            output_combined = ""
            for chunk in stream_chat(simp_msg_history, model=model, temperature=temperature):
                output_combined += chunk
                await websocket.send_json({"status": "generating", "response": output_combined, 'completed': False})

            # Fetch semantic category from the output
            semantic_compression = SemanticCompression(model=f"ollama:{model}", api_key=get_openai_api_key())
            semantic_category = semantic_compression.fetch_semantic_category(output_combined)

            # Start timer for base_token_classifier
            if config['calculateInMessageNER']:
                start_time = time.time()
                base_analysis = base_token_classifier(output_combined)
                duration = time.time() - start_time
                print(f"\t[ base_token_classifier duration: {duration:.4f} seconds ]")

            # Start timer for base_text_classifier
            if config['calculateModerationTags']:
                start_time = time.time()
                text_classifiers = base_text_classifier(output_combined)
                duration = time.time() - start_time
                print(f"\t[ base_text_classifier duration: {duration:.4f} seconds ]")

            if config['calculateModerationTags'] or config['calculateInMessageNER']:
                print(f"\t[ save to conv cache :: conversation {conversation_id}-{chatbot_msg_id} ]")
                dummy_bot_data = {
                    chatbot_msg_id : 
                        {
                        'message': output_combined, 
                        'timestamp': datetime.now(), 
                    }}
                if config['calculateInMessageNER']:
                    dummy_bot_data[chatbot_msg_id]['in_line'] = {'base_analysis': base_analysis}
                if config['calculateModerationTags']:
                    dummy_bot_data[chatbot_msg_id]['commenter'] = {'base_analysis': text_classifiers}
                conv_cache_manager.save_to_cache(conversation_id, dummy_bot_data)
                # Removing the keys from the nested dictionary
                if chatbot_msg_id in dummy_bot_data:
                    dummy_bot_data[chatbot_msg_id].pop('message', None)
                    dummy_bot_data[chatbot_msg_id].pop('timestamp', None)
            else:
                # Saving an empty dictionary for the messag id
                print(f"\t[ save to conv cache :: conversation {conversation_id}-{chatbot_msg_id} ]")
                conv_cache_manager.save_to_cache(conversation_id, {
                    chatbot_msg_id : 
                        {
                        'message': output_combined, 
                        'timestamp': datetime.now(), 
                    }})

            # Send the final completed message
            send_pkg = {"status": "completed", "response": output_combined, "semantic_category": semantic_category, "completed": True}
            if config['calculateModerationTags'] or config['calculateInMessageNER']:
                send_pkg['user_message'] = dummy_data
                send_pkg['bot_data'] = dummy_bot_data
                
            await websocket.send_json(send_pkg)

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.send_json({"status": "error", "message": str(e)})
        await websocket.close()


@router.websocket("/websocket_debate")
async def debate(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload["message"]
            message_history = payload["message_history"]
            model = payload.get("model", "solar")
            temperature = float(payload.get("temperature", 0.04))
            current_topic = payload.get("topic", "Unknown")

            # Set system prompt
            has_topic = False
            
            if current_topic != "Unknown":
                has_topic = True
                prompt = f"You are a smooth talking, eloquent, poignant, insightful AI moderator. The current topic is {current_topic}.\n"

            system_prompt = f"You are a smooth talking, eloquent, poignant, insightful AI moderator. The current topic is unknown, so try not to make any judgements thus far - only re-express the input words in your own style:"
            user_prompt = ""
            if message_history:
                # Add the message history prior to the message
                user_prompt += '\n'.join(msg['role'] + ": " + msg['content'] for msg in message_history)

            print(f"\t[ system prompt :: {system_prompt} ]")
            print(f"\t[ user prompt :: {user_prompt} ]")
            simp_msg_history = [{'role': 'system', 'content': system_prompt}]

            # Simplify message history to required format
            for message in message_history:
                simplified_message = {'role': message['role'], 'content': message['content']}
                if 'images' in message:
                    simplified_message['images'] = message['images']
                simp_msg_history.append(simplified_message)

            # Processing the chat
            output_combined = ""
            for chunk in stream_chat(simp_msg_history, model=model, temperature=temperature):
                output_combined += chunk
                await websocket.send_json({"status": "generating", "response": output_combined, 'completed': False})

            # Fetch semantic category from the output
            semantic_compression = SemanticCompression(model=f"ollama:{model}", api_key=get_openai_api_key())
            semantic_category = semantic_compression.fetch_semantic_category(output_combined)

            # Send the final completed message
            await websocket.send_json(
                {"status": "completed", "response": output_combined, "semantic_category": semantic_category, "completed": True})

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.send_json({"status": "error", "message": str(e)})
        await websocket.close()
