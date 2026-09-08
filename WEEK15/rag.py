from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embedder = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Jarvis uses Whisper for speech-to-text conversion.",
    "Jarvis's memory layer is built using ChromaDB.",
    "The holographic interface uses Three.js for 3D rendering."
]

documents_embeddings = embedder.encode(documents)


print(documents_embeddings.shape)