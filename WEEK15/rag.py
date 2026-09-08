from groq import groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os 
import numpy as np 

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
print(similarities)
print(document_embeddings.shape)