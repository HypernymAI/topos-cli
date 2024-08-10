# https://platform.openai.com/docs/libraries/python-library
# setting the env var in a user data script in aws
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv('OPENAI_API_KEY'))
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
    )

def stream_chat(message_history, model="gpt-4o", temperature=0):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=message_history,
        temperature=temperature,
        stream=True
    )
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

def generate_response(context, prompt, model="gpt-4o", temperature=0):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    return response.choices[0].message.content

def generate_response_messages(message_history, model="gpt-4o", temperature=0):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=message_history,
        stream=False
    )
    return response.choices[0].message.content