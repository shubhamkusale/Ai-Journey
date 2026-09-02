from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def calculate(expression):
    return eval(expression)

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluates a math expression and returns the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression like '847 * 392'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

messages = [{"role": "user", "content": "What is 847 times 392?"}]

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=messages,
    tools=tools
)

reply = response.choices[0].message

if reply.tool_calls:
    tool_call = reply.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    result = calculate(args["expression"])
    print("Tool actually calculated:", result)
else:
    print("Model answered directly Yey:", reply.content)