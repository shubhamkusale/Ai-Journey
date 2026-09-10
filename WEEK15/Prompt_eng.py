from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

combined_prompt = """You are a senior Python developer with 10 years of experience.

Here are examples of how to explain technical concepts clearly:

Concept: What is a list in Python?
Explanation: A list is an ordered collection that can hold multiple items and be changed after creation.

Concept: What is a dictionary in Python?
Explanation: A dictionary stores data as key-value pairs, letting you look up a value instantly using its key.

Now explain the following concept in the same style.
Think through the core idea step by step before writing your final explanation.

Concept: What is recursion in Python?

Rules:
- Final explanation must be exactly 2 sentences.
- Do not use any analogies.
- Show your step-by-step thinking first, then give the final 2-sentence explanation labeled "Final:"."""


response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": combined_prompt}
    ]
)

print(response.choices[0].message.content)
print("DEBUG - prompt is:", combined_prompt)