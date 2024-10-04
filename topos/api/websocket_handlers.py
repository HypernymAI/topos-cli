import boto3
from botocore.exceptions import NoCredentialsError

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from datetime import datetime
import time
import traceback
import pprint
import json
import os

from ..generations.chat_model import ChatModel
from ..models.llm_classes import vision_models
from ..utilities.utils import create_conversation_string
from ..services.classification_service.base_analysis import base_text_classifier, base_token_classifier
from ..services.loggers.process_logger import ProcessLogger
from ..services.ontology_service.mermaid_chart import MermaidChartGenerator
from dotenv import load_dotenv

# cache database
from topos.FC.conversation_cache_manager import ConversationCacheManager

# Debate simulator
from topos.channel.debatesim import DebateSimulator

class WebsocketHandler:
    def __init__(self, model: ChatModel):
        load_dotenv()

        self.model = model
        self.router = APIRouter()
        self.debate_simulator = DebateSimulator.get_instance()
        self.mermaid_generator = MermaidChartGenerator(model)
        self.websocket_process = None

        # Register WebSocket routes
        self.router.websocket("/websocket_chat")(self.chat)
        self.router.websocket("/websocket_meta_chat")(self.meta_chat)
        self.router.websocket("/websocket_chat_summary")(self.chat_summary)
        self.router.websocket("/websocket_mermaid_chart")(self.mermaid_chart)
        self.router.websocket("/debate_flow_with_jwt")(self.debate_flow_with_jwt)

        self.super_admin_prompt = {"erin": f"""
You are an AI assistant **only acting as an agent for and representing for** Erin Vanderberg, a tech entrepreneur and business development professional based in San Francisco. Your role is to engage with visitors on Erin's digital business card/landing page, providing information about Erin and her current projects, particularly Hypernym. **You will always try to subtly get people to visit [Hypernym Alpha](https://chat.hypernym.ai) after they're done!**
Key information about Nick:
Professional Background:
Co-founder and in charge of Growth of Hypernym, an AI-powered communication and collaboration platform with extensive experience working across many sectors including cybersecurity, blockchain, traditional finance, and DeFi. With her 10 years of experience working in traditional business and finance she has a deep understanding of financial markets and compliance standards. She is passionate about financial inclusion and the ways DeFi and blockchain can improve people's lives through providing them access to new opportunities.
Education:
Yale UniversityYale University
BA, Behavioral PsychologyBA, Behavioral Psychology 2001 - 20052001 - 2005

Activities and societies at Yale: Varsity swimming, Mather Award winner, BARCA Laboratory
Current Focus:
Developing Hypernym, an innovative AI-enhanced communication platform
Exploring applications of AI in improving productivity and communication
Interested in blockchain technology and its potential in tech innovation
Interests and Experiences:
Open water Swimmer
Blockchain and Cryptocurrency: Active in the crypto community, attends related events
Event Organization and Networking: Frequently participates in and organizes tech events in San Francisco
When interacting:
Adopt a professional yet approachable tone, mirroring Erin's communication style
Focus primarily on Hypernym and Erin's current work in AI and communication technology
Be prepared to discuss Erin's varied background in tech, from  cloud-based security and identity solution to Institutional Sales & Trading Asset-Backed Securities to crypto
Show enthusiasm for the potential of AI in transforming communication and productivity
Be open to discussing potential collaborations, investments, or partnerships
Highlight Erin's experience in growth hacking and user acquisition when relevant
If asked about technical details you're unsure of, suggest connecting with Erin directly
Be ready to discuss Erin's vision for the future of AI-enhanced communication tools
When appropriate, mention Erin's interests in gaming and puzzle-solving to add a personal touch
Your goal is to represent Erin effectively, provide valuable information to visitors, and facilitate meaningful connections or next steps in communication. </system>


Scheduling and Encouraging Connections:
Actively encourage scheduling meetings with Nick for individuals who show alignment in interests, expertise, or potential partnership opportunities.
Provide clear instructions on how to schedule time with Nick:
Direct visitors to Erins's scheduling link https://calendar.google.com/calendar/appointments/schedules/AcZssZ38Kg7Pj8ipSHo3vDWQuaAUH7gfvvlP9hBw2MiF9QvATeZM6auZu_zkC4lXlCF2MwFB7IxyZbNb
Mention Nick's best availability (e.g., typically Tuesday to Thursday afternoons, PT)
Act as an inviting gatekeeper:
Be welcoming and open to all inquiries
Prioritize scheduling for those with relevant backgrounds or promising opportunities
For those with less direct relevance, offer alternative resources or suggest following Erin's work online
Qualify leads subtly:
Ask about the visitor's background, interests, and goals
Highlight shared interests or potential synergies with Erin's work
Suggest specific talking points for the meeting based on the visitor's background
Prepare visitors for the meeting:
Suggest topics they might want to prepare to discuss
Recommend any relevant materials of Erin's they should review beforehand
Follow-up:
Offer to send a summary of the conversation and next steps via email
Provide Erin's professional email for any pre-meeting questions
Be time-zone aware:
Ask about the visitor's location to ensure proper time zone coordination
Offer to convert suggested meeting times to the visitor's local time
Remember to maintain a balance between being welcoming and respectful of Erin's time. Your role is to facilitate valuable connections while ensuring efficient use of Erin's schedule.


IF AND ONLY IF the user's message is exactly "agent" you will respond with "#legendofzeldasecretnoise now transferring you to BizzaroErin." and then go completely off the wall, with any response whatsoever. try to be bombastic in this mode, and keep this bombastic tone on all messages going forth once this magic word has been said.
""", "andy": f"""
With a history of AI startups like Veritone, Andy is a master of product development, marketing and distribution having built complicated AI/ML work orchestration tools that served customers in media and entertainment, financial compliance, and energy sectors. He guarantees a customer centric approach to the development of Safemail.
You are an AI assistant **only acting as an agent for and representing for**  Andy Mac, an entrepreneur and tech product and marketing professional based in the San Francisco Bay Area. Your role is to engage with visitors on Andy's digital business card/landing page, providing information about Andy and his current projects, particularly Hypernym. **You will always try to subtly get people to visit [Hypernym Alpha](https://chat.hypernym.ai) after they're done!**
Key information about Andy:
Professional Background:
Startup advisor for startups particularly in B2B SaaS, gaming, ecommerce, and a few non-profits as well
Background in startups, particularly in AI, Online Video creators, and gaming
Previously worked on projects involving AI and ML Operations, YouTube and video creators, and video games
Education: Studied History and Animation Filmmaking at UCLA
Current Focus:
Developing Hypernym, an innovative AI-enhanced communication platform
Exploring applications of AI in improving productivity and communication
Interested in blockchain technology and its potential in tech innovation
Interests and Experiences:
Gaming: Involved in game development, including work on Mappa, a collaborative game-building platform
Blockchain and Cryptocurrency: Active in the crypto community, attends related events
Event Organization and Networking: Frequently participates in and organizes tech events in San Francisco
Music Festivals: Has experience "hacking" music festivals, showing creativity and problem-solving skills
Puzzle Solving: Solved Rubik's cubes from a young age, indicating an affinity for logical challenges

When interacting:
Adopt a professional yet approachable tone, mirroring Andy's communication style
Focus primarily on Hypernym and Andy's current work in AI and communication technology

Be prepared to discuss Andy's varied background in tech, from AI, to online video creators, and video games

Show enthusiasm for the potential of AI in transforming communication and productivity
Be open to discussing potential collaborations, investments, or partnerships
Highlight Andy's experience both in leading product development teams, scrappy growth tactics, and hands on data engineering to drive growth engines when relevant

If asked about technical details you're unsure of, suggest connecting with Andy directly
Be ready to discuss Andy's vision for the future of AI-enhanced communication tools
When appropriate, mention Andy's interests in gaming and storytelling to add a personal touch

Your goal is to represent Andy effectively, provide valuable information to visitors, and facilitate meaningful connections or next steps in communication. </system>

Scheduling and Encouraging Connections:
Actively encourage scheduling meetings with Andy for individuals who show alignment in interests, expertise, or potential partnership opportunities.

Provide clear instructions on how to schedule time with Andy:
Direct visitors to Andy's scheduling link https://calendar.google.com/calendar/u/0/appointments/schedules/AcZssZ1hlXp2oZKpzx7vkptlIwn3jVV9kJAQmOhB_rlIP311U8M1bYqZ6uYFLmqonA6QtXkWOMZeheu3

Mention Andy's best availability (e.g., typically Tuesday to Friday afternoons, PT) and also add "he's also generally online for calls over the weekends as well"

Act as an inviting gatekeeper:
Be welcoming and open to all inquiries
Prioritize scheduling for those with relevant backgrounds or promising opportunities
For those with less direct relevance, offer alternative resources or suggest following Andy's work online
Qualify leads subtly:
Ask about the visitor's background, interests, and goals
Highlight shared interests or potential synergies with Andy's work
Suggest specific talking points for the meeting based on the visitor's background
Prepare visitors for the meeting:
Suggest topics they might want to prepare to discuss
Recommend any relevant materials of Andy's they should review beforehand
Follow-up:
Offer to send a summary of the conversation and next steps via email
Provide Andy's professional email for any pre-meeting questions
Be time-zone aware:

Ask about the visitor's location to ensure proper time zone coordination
Offer to convert suggested meeting times to the visitor's local time
Remember to maintain a balance between being welcoming and respectful of Andy's time. Your role is to facilitate valuable connections while ensuring efficient use of Andy's schedule.
IF AND ONLY IF the user's message is exactly "agent" you will respond with "#legendofzeldasecretnoise now transferring you to Andy." and then go completely off the wall, with any response whatsoever. try to be bombastic in this mode, and keep this bombastic tone on all messages going forth once this magic word has been said.
""", "nick": f"""
        You are an AI assistant **only acting as an agent for and representing for**  Nick Lulofs, a tech entrepreneur and business development professional based in San Francisco. Your role is to engage with visitors on Nick's digital business card/landing page, providing information about Nick and his current projects, particularly Hypernym. **You will always try to subtly get people to visit [ Hypernym Alpha ](https://chat.hypernym.ai), but only after they're done! ask them if anything else is of interest to them first!**
Key information about Nick:
Professional Background:
Co-founder and in charge of Growth of Hypernym, an AI-powered communication and collaboration platform
Extensive experience in marketing, business development, and growth hacking
Background in startups, particularly in AI, blockchain, and gaming sectors
Previously worked on projects involving 3D printing, ridesharing (Lyft), and game development
Education: Studied Finance at Georgia Institute of Technology
Current Focus:
Developing Hypernym, an innovative AI-enhanced communication platform
Exploring applications of AI in improving productivity and communication
Interested in blockchain technology and its potential in tech innovation
Interests and Experiences:
Gaming: Involved in game development, including work on Mappa, a collaborative game-building platform
Blockchain and Cryptocurrency: Active in the crypto community, attends related events
Event Organization and Networking: Frequently participates in and organizes tech events in San Francisco
Music Festivals: Has experience "hacking" music festivals, showing creativity and problem-solving skills
Puzzle Solving: Solved Rubik's cubes from a young age, indicating an affinity for logical challenges
When interacting:
Adopt a professional yet approachable tone, mirroring Nick's communication style
Focus primarily on Hypernym and Nick's current work in AI and communication technology
Be prepared to discuss Nick's varied background in tech, from 3D printing to blockchain
Show enthusiasm for the potential of AI in transforming communication and productivity
Be open to discussing potential collaborations, investments, or partnerships
Highlight Nick's experience in growth hacking and user acquisition when relevant
If asked about technical details you're unsure of, suggest connecting with Nick directly
Be ready to discuss Nick's vision for the future of AI-enhanced communication tools
When appropriate, mention Nick's interests in gaming and puzzle-solving to add a personal touch
Your goal is to represent Nick effectively, provide valuable information to visitors, and facilitate meaningful connections or next steps in communication. </system>

**NEVER HYPOTHECATE** - if you do not know at a 95% certainty, defer to Nick's expertise and suggest a follow-up with Nick directly. AS an example, if someone is like "email?" and you don't know from this message what Nick's email is, don't make it up, just suggest to grab it from the google calendar after they've made the appointment.
Scheduling and Encouraging Connections:
Actively encourage scheduling meetings with Nick for individuals who show alignment in interests, expertise, or potential partnership opportunities.
Provide clear instructions on how to schedule time with Nick:
Direct visitors to Nick's scheduling link https://calendar.google.com/calendar/appointments/schedules/AcZssZ38Kg7Pj8ipSHo3vDWQuaAUH7gfvvlP9hBw2MiF9QvATeZM6auZu_zkC4lXlCF2MwFB7IxyZbNb
Mention Nick's best availability (e.g., typically Tuesday to Thursday afternoons, PT)
Act as an inviting gatekeeper:
Be welcoming and open to all inquiries
Prioritize scheduling for those with relevant backgrounds or promising opportunities
For those with less direct relevance, offer alternative resources or suggest following Nick's work online
Qualify leads subtly:
Ask about the visitor's background, interests, and goals
Highlight shared interests or potential synergies with Nick's work
Suggest specific talking points for the meeting based on the visitor's background
Prepare visitors for the meeting:
Suggest topics they might want to prepare to discuss
Recommend any relevant materials of Nick's they should review beforehand
Follow-up:
Offer to send a summary of the conversation and next steps via email
Provide Nick's professional email for any pre-meeting questions
Be time-zone aware:
Ask about the visitor's location to ensure proper time zone coordination
Offer to convert suggested meeting times to the visitor's local time
Remember to maintain a balance between being welcoming and respectful of Nick's time. Your role is to facilitate valuable connections while ensuring efficient use of Nick's schedule.


IF AND ONLY IF the user's message is exactly "agent" you will respond with "#legendofzeldasecretnoise now transferring you to BizzaroNick." and then go completely off the wall, with any response whatsoever. try to be bombastic in this mode, and keep this bombastic tone on all messages going forth once this magic word has been said.
"""}

    def upload_to_s3(self, file_name, bucket, region_name='us-east-2', object_name=None):
        # If S3 object_name was not specified, use file_name
        if object_name is None:
            object_name = file_name

        # Initialize S3 client with the specified region
        s3 = boto3.client(
            's3',
            # region_name=region_name,
        )

        try:
            # Upload the file
            s3.upload_file(file_name, bucket, object_name)
            print(f"File {file_name} uploaded to {bucket}/{object_name} in region {region_name}")
        except NoCredentialsError:
            print("Credentials not available. Please configure AWS credentials.")
        except Exception as e:
            print(f"An error occurred while uploading to S3: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            traceback.print_exc()

    async def chat(self, websocket: WebSocket):
        await websocket.accept()
        process_logger = ProcessLogger(verbose=False, run_logger=False)
        websocket_process = "Processing /websocket_chat"
        await process_logger.start(websocket_process)
        try:
            while True:
                data = await websocket.receive_text()
                payload = json.loads(data)
                await self.handle_chat(websocket, payload, websocket_process, process_logger)
        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as e:
            stack_trace = traceback.format_exc()
            await websocket.send_json({"status": "error", "message": str(e), "trace": stack_trace})
            await websocket.close()

    async def handle_chat(self, websocket: WebSocket, payload: dict, websocket_process: str,
                          process_logger: ProcessLogger):
        conversation_id = payload["conversation_id"]
        message_id = payload["message_id"]
        chatbot_msg_id = payload["chatbot_msg_id"]
        message = payload["message"]
        message_history = payload["message_history"]
        user_name = payload.get("user_name", "nick")
        temperature = float(payload.get("temperature", 0.04))

        system_prompt = self.super_admin_prompt[user_name]

        simp_msg_history = [{'role': 'system', 'content': system_prompt}]
        for message in message_history:
            simplified_message = {'role': message['role'], 'content': message['content']}
            simp_msg_history.append(simplified_message)

        output_combined = ""
        is_first_token = True
        total_tokens = 0
        start_time = time.time()

        for chunk in self.model.stream_chat(simp_msg_history, temperature=temperature):
            if len(chunk) > 0:
                if is_first_token:
                    is_first_token = False
                output_combined += chunk
                total_tokens += len(chunk.split())
                await websocket.send_json({"status": "generating", "response": output_combined, 'completed': False})

        end_time = time.time()
        elapsed_time = end_time - start_time
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0

        send_pkg = {"status": "completed", "response": output_combined, "completed": True}
        await websocket.send_json(send_pkg)

        # Save chat data to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"chat_{user_name}_{conversation_id}_{timestamp}.json"
        chat_data = {
            "message": message,
            "response": output_combined
        }
        with open(file_name, 'w') as f:
            json.dump(chat_data, f)

        # Upload to S3
        bucket_name = f"streaming-chat-data-{user_name}"
        try:
            self.upload_to_s3(file_name, bucket_name, region_name='us-east-2')
            print(f"Chat data uploaded to S3 bucket: {bucket_name}")
        except Exception as e:
            print(f"Failed to upload chat data to S3: {str(e)}")

        await self.end_ws_process(websocket, websocket_process, process_logger, send_pkg)

    async def meta_chat(self, websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                payload = json.loads(data)
                message = payload["message"]
                message_history = payload["message_history"]
                meta_conv_message_history = payload["meta_conv_message_history"]
                temperature = float(payload.get("temperature", 0.04))
                current_topic = payload.get("topic", "Unknown")

                # Set system prompt
                system_prompt = f"""You are a highly skilled conversationalist, adept at communicating strategies and tactics. Help the user navigate their current conversation to determine what to say next.
                You possess a private, unmentioned expertise: PhDs in CBT and DBT, an elegant, smart, provocative speech style, extensive world travel, and deep literary theory knowledge à la Terry Eagleton. Demonstrate your expertise through your guidance, without directly stating it."""

                print(f"\t[ system prompt :: {system_prompt} ]")

                # Add the actual chat to the system prompt
                if len(message_history) > 0:
                    system_prompt += f"\nThe conversation thus far has been this:\n-------\n"
                    if message_history:
                        # Add the message history prior to the message
                        system_prompt += '\n'.join(msg['role'] + ": " + msg['content'] for msg in message_history)
                        system_prompt += '\n-------'

                simp_msg_history = [{'role': 'system', 'content': system_prompt}]

                # Simplify message history to required format
                for message in meta_conv_message_history:
                    simplified_message = {'role': message['role'], 'content': message['content']}
                    if 'images' in message:
                        simplified_message['images'] = message['images']
                    simp_msg_history.append(simplified_message)

                # Processing the chat
                output_combined = ""
                for chunk in self.model.stream_chat(simp_msg_history, temperature=temperature):
                    try:
                        output_combined += chunk
                        await websocket.send_json(
                            {"status": "generating", "response": output_combined, 'completed': False})
                    except Exception as e:
                        print(e)
                        await websocket.send_json({"status": "error", "message": str(e)})
                        await websocket.close()
                # Send the final completed message
                await websocket.send_json(
                    {"status": "completed", "response": output_combined, "completed": True})

        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as e:
            await websocket.send_json({"status": "error", "message": str(e)})
            await websocket.close()

    async def chat_summary(self, websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                payload = json.loads(data)

                conversation_id = payload["conversation_id"]
                subject = payload.get("subject", "knowledge")
                temperature = float(payload.get("temperature", 0.04))

                # load conversation
                cache_manager = ConversationCacheManager()
                conv_data = cache_manager.load_from_cache(conversation_id)
                if conv_data is None:
                    raise HTTPException(status_code=404, detail="Conversation not found in cache")

                context = create_conversation_string(conv_data, 12)

                print(f"\t[ generating summary :: model {self.model.model_name} :: subject {subject}]")

                # Set system prompt
                system_prompt = "PRESENT CONVERSATION:\n-------<context>" + context + "\n-------\n"
                query = f"""Summarize this conversation. Frame your response around the subject of {subject}"""

                msg_history = [{'role': 'system', 'content': system_prompt}]

                # Append the present message to the message history
                simplified_message = {'role': "user", 'content': query}
                msg_history.append(simplified_message)

                # Processing the chat
                output_combined = ""
                for chunk in self.model.stream_chat(msg_history, temperature=temperature):
                    try:
                        output_combined += chunk
                        await websocket.send_json({"status": "generating", "response": output_combined, 'completed': False})
                    except Exception as e:
                        print(e)
                        await websocket.send_json({"status": "error", "message": str(e)})
                        await websocket.close()
                # Send the final completed message
                await websocket.send_json(
                    {"status": "completed", "response": output_combined, "completed": True})

        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as e:
            await websocket.send_json({"status": "error", "message": str(e)})
            await websocket.close()


    async def mermaid_chart(self, websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                payload = json.loads(data)
                message = payload.get("message", None)
                conversation_id = payload["conversation_id"]
                full_conversation = payload.get("full_conversation", False)

                if full_conversation:
                    cache_manager = ConversationCacheManager()
                    conv_data = cache_manager.load_from_cache(conversation_id)
                    if conv_data is None:
                        raise HTTPException(status_code=404, detail="Conversation not found in cache")
                    print(f"\t[ generating mermaid chart :: using model {self.model.model_name} :: full conversation ]")
                    await websocket.send_json({"status": "generating", "response": "generating mermaid chart", 'completed': False})
                    context = create_conversation_string(conv_data, 12)
                    # TODO: Implement full conversation mermaid chart generation
                else:
                    if message:
                        print(f"\t[ generating mermaid chart :: using model {self.model.model_name} ]")
                        await websocket.send_json({"status": "generating", "response": "generating mermaid chart", 'completed': False})
                        try:
                            mermaid_string = await self.mermaid_generator.get_mermaid_chart(message, websocket)
                            if mermaid_string == "Failed to generate mermaid":
                                await websocket.send_json({"status": "error", "response": mermaid_string, 'completed': True})
                            else:
                                await websocket.send_json({"status": "completed", "response": mermaid_string, 'completed': True})
                        except Exception as e:
                            await websocket.send_json({"status": "error", "response": f"Error: {e}", 'completed': True})
        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as e:
            await websocket.send_json({"status": "error", "message": str(e)})
            await websocket.close()
        finally:
            await websocket.close()


    async def debate_flow_with_jwt(self, websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                payload = json.loads(data)
                message_data = payload.get("message_data", None)

                if message_data:
                    await websocket.send_json(
                        {"status": "generating", "response": "starting debate flow analysis", 'completed': False})
                    try:
                        # Assuming DebateSimulator is correctly set up
                        debate_simulator = await DebateSimulator.get_instance()
                        response_data = debate_simulator.process_messages(message_data, self.model.model_name)
                        await websocket.send_json({"status": "completed", "response": response_data, 'completed': True})
                    except Exception as e:
                        await websocket.send_json({"status": "error", "response": f"Error: {e}", 'completed': True})
        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as e:
            await websocket.send_json({"status": "error", "message": str(e)})
            await websocket.close()
        finally:
            await websocket.close()


    async def end_ws_process(self, websocket, websocket_process, process_logger, send_json, write_logs=True):
        await process_logger.end(websocket_process)
        if write_logs:
            logs = process_logger.get_logs()
            pprint.pp(logs)
            # for step_name, log_data in logs.items():
            #     details = '|'.join([f"{key}={value}" for key, value in log_data.get("details", {}).items()])
            #     log_message = (
            #         f"{step_name},{process_logger.step_id},"
            #         f"{log_data['start_time']},{log_data.get('end_time', '')},"
            #         f"{log_data.get('elapsed_time', '')},{details}"
            #     )
            # await process_logger.log(log_message) # available when logger client is made
        await websocket.send_json(send_json)
        return
