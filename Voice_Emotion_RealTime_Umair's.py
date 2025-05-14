import os
import time
import sounddevice as sd
from scipy.io.wavfile import write
from speechbrain.inference.interfaces import foreign_class
import threading
import keyboard  # pip install keyboard

# 🧠 Load the custom classifier
classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier"
)

# 🎧 Settings
FILENAME = "recording.wav"
DURATION = 3  # seconds
FS = 16000  # sample rate

# 🎙️ Record audio
def record_audio(filename=FILENAME, duration=DURATION, fs=FS):
    print("\033[94m🎙️  Listening... ({} sec)\033[0m".format(duration))
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)

# 🔍 Predict emotion
def predict_emotion(filename=FILENAME):
    out_prob, score, index, text_lab = classifier.classify_file(filename)
    print(f"\033[92m🧠 Emotion: {text_lab} | 📊 Score: {score}\033[0m\n")

# 🔁 Main loop
def main_loop():
    print("\033[96m🔄 Starting real-time emotion detection (Press 'X' to stop)\033[0m\n")
    while not keyboard.is_pressed('x'):
        record_audio()
        predict_emotion()
    print("\n\033[91m🛑 Detection stopped by user.\033[0m")

# 🚀 Entry point
if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n\033[91m⛔ Interrupted manually.\033[0m")
    finally:
        if os.path.exists(FILENAME):
            os.remove(FILENAME)
