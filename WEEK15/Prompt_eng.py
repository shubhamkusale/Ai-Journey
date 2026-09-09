from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

few_shot_prompt = """Convert casual sentences to formal English.

Casual: "wanna hang out later?"
Formal: "Would you like to meet up later?"

Casual: "can't make it, sry"
Formal: "I am unable to attend, my apologies."

Casual: "gonna grab lunch, brb"
Formal:"""

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": few_shot_prompt}
    ]
)

print(response.choices[0].message.content)
print("DEBUG - prompt is:", few_shot_prompt)