from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

conversation = []

def chat(user_input):
    conversation.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=conversation
    )

    reply = response.choices[0].message.content

    conversation.append({"role": "assistant", "content": reply})

    return reply


print(chat("My name is Shubham"))
print(chat("What's my name?"))