from groq import Groq

client = Groq(api_key="YOUR_API_KEY_HERE")

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "What is a neural network in one sentence?"}
    ]
)

print(response.choices[0].message.content)