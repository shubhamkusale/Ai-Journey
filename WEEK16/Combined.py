from groq import Groq
from dotenv import load_dotenv
import os 
import pyaudio
import sounddevice as sd
from scipy.io.wavfile import write
load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
duration = 5  # seconds to record
sample_rate = 44100

print("Recording... speak now")
recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sd.wait() 
write("live_input.wav", sample_rate, recording)
print("Recording saved.")
with open("live_input.wav", "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3"
    )

spoken_text = transcription.text
print("You said:", spoken_text)

chat_response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": spoken_text + " (Answer in 2-3 sentences, conversational tone, since this will be spoken aloud.)"}]
)
answer_text = chat_response.choices[0].message.content
print("jarvis says:", answer_text)

speech_response = client.audio.speech.create(
    model="canopylabs/orpheus-v1-english",
    voice="troy",
    input=answer_text,
    response_format="wav"
)

audio_data = speech_response.read()

p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(2),channels=1, rate=24000, output= True)
stream.write(audio_data[44:])
stream.stop_stream()
stream.close()
p.terminate()