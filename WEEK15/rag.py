from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os 
import numpy as np 

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
embedder = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Jarvis uses Whisper for speech-to-text conversion.",
    "Jarvis's memory layer is built using ChromaDB.",
    "The holographic interface uses Three.js for 3D rendering."
]

document_embeddings = embedder.encode(documents)

question = "How does Jarvis understand voice input?"
question_embedding = embedder.encode([question])

similarities = cosine_similarity(question_embedding, document_embeddings)

best_match_index =  np.argmax(similarities)
best_document = documents[best_match_index]
print(similarities)
print(document_embeddings.shape)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": f"Using this context: '{best_document}', answer this question: {question}"}
    ]
)

print("Final answer:", response.choices[0].message.content)