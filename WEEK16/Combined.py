from groq import Groq
from dotenv import load_dotenv
import os 
import pyaudio

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

audio_path = os.path.join(BASE_DIR,"Real_speech.wav")
with open(audio_path, "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3"
    )

spoken_text = transcription.text
print("You said:", spoken_text)
