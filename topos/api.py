from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
import uvicorn
import json
import os


from topos.summaries.summaries import stream_chat

import ssl
import requests

app = FastAPI()

# get the current working directory
project_dir = "/Users/dialogues/developer/topos/cli"
key_path = project_dir + "/key.pem"
cert_path = project_dir + "/cert.pem"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


import tkinter as tk
from tkinter import filedialog

def read_file_as_bytes(file_path):
    try:
        with open(file_path, 'rb') as file:
            file_bytes = list(file.read())
        return file_bytes
    except FileNotFoundError:
        print("File not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

@app.post("/get_files")
async def get_files():
    root = tk.Tk()
    root.withdraw()
    filetypes = [("PNG files", "*.png"), ("JPG files", "*.jpg"), ("JPEG files", "*.jpeg")]
    file_path = filedialog.askopenfilename(title="Select an image file",
                                       filetypes=(filetypes))
    print(file_path)
    
    # Use the os.path module
    system_path = os.path.abspath("/")
    print(system_path)
    bytes_list = read_file_as_bytes(file_path)
    media_type="application/json",
    
    # data = json.dumps(, ensure_ascii=False)
    # print(data[:110])
    return {"file_name" : [i for i in file_path], "bytes": bytes_list}

@app.post("/list_models")
async def list_models():
    url = "http://localhost:11434/api/tags"
    try:
        result = requests.get(url)
        if result.status_code == 200:
            return {"result": result.json()}
        else:
            print(f"Ping failed. Status code: {response.status_code}")
            return None
    except requests.ConnectionError:
        print("Ping failed. Connection refused. Check if the server is running.")
        return None



"""

Standard Chat

"""


@app.websocket("/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()
    data = await websocket.receive_text()
    payload = json.loads(data)  # Assuming the client sends JSON with 'option' and 'raw_text_prompt' keys
    await websocket.send_json({"status": "started"})
    message = payload["message"]
    message_history = payload["message_history"]

    """
    the message history is a list of messages and each message can have a lot of data associated with it

    for now we are just going to extract
    """
    
    model = payload["model"] if "model" in payload else "solar"

    temperature = 0.04 # default temperature
    
    if "temperature" in payload:
        if payload["temperature"] != None:
            tmp = float(payload["temperature"])
            temperature = tmp if tmp <= 1 else 1
    
    print(f"MSG: {message}")
    print(f"MSGHIST: {len(message_history)}")
    print(f"MODEL: {model}")
    print(f"TEMP: {temperature}")
    # First step, set the prompt for the short summarizer
    # insert entries information as system prompt
    simp_msg_history = [{'role': 'system', 'content': "You are an AI assistant being helpful, speaking to someone who is trying to get along"}]

    # convert message history into basic message history
    for i in message_history:
        if "images" in i:
            simp_msg_history.append({'role': i['role'], 'content': i['content'], 'images': i['images']})
        else:
            simp_msg_history.append({'role': i['role'], 'content': i['content']})

    try:
        text = []
        for chunk in stream_chat(simp_msg_history, model = model, temperature=temperature):
            text.append(chunk)
            story_summary = {'response':''.join(text), 'completed': False}
            await websocket.send_json({"status": "generating", **story_summary})
        story_summary = {'response':''.join(text), 'completed': True}  # llm function
        await websocket.send_json({"status": "completed", **story_summary})
    except Exception as e:
        await websocket.send_json({"status": "error", "message": "Generation failed"})
        await websocket.close()
        return

    await websocket.close()







"""

START API OPTIONS

There is the web option for networking with the online, webhosted version
There is the local option to connect the local apps to the Topos API (Grow debugging, and the Chat app)


"""




def start_local_api():
    print("\033[92mINFO:\033[0m     API docs available at: http://0.0.0.0:13341/docs")
    uvicorn.run(app, host="0.0.0.0", port=13341) # port for the local versions

def start_web_api():
    print("\033[92mINFO:\033[0m     API docs available at: https://0.0.0.0:13341/docs")
    uvicorn.run(app, host="0.0.0.0", port=13341, ssl_keyfile=key_path,
               ssl_certfile=cert_path) # the web version needs the ssl certificate loaded
    
if __name__ == "__main__":
    start_local_api()
    print("\033[92mINFO:\033[0m     API docs available at: http://0.0.0.0:13341/docs")
    uvicorn.run(app, host="0.0.0.0", port=13341)