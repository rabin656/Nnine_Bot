import os
import speech_recognition as sr
from gtts import gTTS
import tempfile

# --- Configuration ---
OUTPUT_DIR = "generated_audio"

def generate_audio_from_text(text, filename="response.mp3"):
    """Converts text to speech using gTTS."""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        tts = gTTS(text=text, lang='en')
        tts.save(save_path)
        return save_path
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

def transcribe_audio(audio_file):
    """Transcribes audio from a file-like object using Google Speech Recognition."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None
