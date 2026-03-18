import os
import time
import pyaudio
import numpy as np
import google.generativeai as genai
from openwakeword.model import Model
from google.cloud import speech_v2, texttospeech
import pygame

# --- CONFIGURATION ---
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "your_google_key.json"
PROJECT_ID = "your-project-id"
LOCATION = "asia-southeast1" # Singapore region for lowest Malaysian latency
genai.configure(api_key="YOUR_GEMINI_API_KEY")

# Audio Parameters
RATE = 16000
CHUNK = 1280 

# Initialize Engines
pygame.mixer.init()
ping_sound = pygame.mixer.Sound("ping.wav") # Make sure this file exists!
oww_model = Model(wakeword_models=['baby'], inference_framework="tflite")
asr_client = speech_v2.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()
gemini_model = genai.GenerativeModel('gemini-3-flash-preview')

def speak_as_young_boy(text):
    """TTS with SSML for a Young Boy voice (+7 semitones)"""
    ssml_text = f"<speak><prosody pitch='+7st' rate='1.1'>{text}</prosody></speak>"
    input_text = texttospeech.SynthesisInput(ssml=ssml_text)
    voice = texttospeech.VoiceSelectionParams(language_code="ms-MY", name="ms-MY-Neural2-B")
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

    response = tts_client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream.write(response.audio_content)
    stream.stop_stream()
    stream.close()

def get_gemini_reply(user_input):
    """LLM with Boy Persona"""
    prompt = (
        f"Anda adalah 'Baby', seorang budak lelaki Malaysia berumur 8 tahun. "
        f"Gunakan Bahasa Melayu santai. Berikan jawapan yang ringkas. "
        f"Soalan: {user_input}"
    )
    response = gemini_model.generate_content(prompt)
    return response.text

def capture_command():
    """ASR using Google Chirp for Malay"""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=4096)
    print("🎤 Listening...")
    frames = [stream.read(4096) for _ in range(0, int(RATE / 4096 * 4))] # 4s recording
    stream.stop_stream()
    stream.close()

    config = speech_v2.types.RecognitionConfig(
        auto_decoding_config=speech_v2.types.AutoDetectDecodingConfig(),
        language_codes=["ms-MY"], model="chirp"
    )
    request = speech_v2.types.RecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/{LOCATION}/recognizers/_",
        config=config, content=b"".join(frames)
    )
    
    response = asr_client.recognize(request=request)
    return response.results[0].alternatives[0].transcript if response.results else None

# --- MAIN LOOP ---
def main():
    p = pyaudio.PyAudio()
    mic_stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("\n✅ 'Baby' is online. Say 'Hi Baby'...")

    try:
        while True:
            data = mic_stream.read(CHUNK, exception_on_overflow=False)
            audio_frame = np.frombuffer(data, dtype=np.int16)
            prediction = oww_model.predict(audio_frame)

            if prediction['baby'] > 0.6:
                print("\n✨ Waked Up!")
                ping_sound.play() # PLAY PING IMMEDIATELY
                mic_stream.stop_stream()
                
                user_text = capture_command()
                if user_text:
                    print(f"👤 User: {user_text}")
                    ai_reply = get_gemini_reply(user_text)
                    print(f"🤖 Baby: {ai_reply}")
                    speak_as_young_boy(ai_reply)
                
                mic_stream.start_stream()
                print("\n👂 Waiting for 'Hi Baby'...")

    except KeyboardInterrupt:
        print("\n👋 Bye!")
    finally:
        mic_stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
