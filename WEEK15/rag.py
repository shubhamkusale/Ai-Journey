from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Jarvis uses Whisper for speech-to-text conversion.",
    "Jarvis's memory layer is built using ChromaDB.",
    "The holographic interface uses Three.js for 3D rendering."
]

