from groq import Groq
from dotenv import load_dotenv
import os
import pyaudio

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

text_to_speak = "I may be far from where I want to be, but I’m closer than I was yesterday—and I will not stop until I become the man I know I can be"

response = client.audio.speech.create(
    model="canopylabs/orpheus-v1-english",
    voice="troy",
    input=text_to_speak,
    response_format="wav"
)

audio_data = response.read()

p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(2), channels=1, rate=24000, output=True)
stream.write(audio_data[44:])
stream.stop_stream()
stream.close()
p.terminate()