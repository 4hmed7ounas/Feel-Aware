import os
import numpy as np
import pyaudio
import wave
import time
import librosa
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import warnings
warnings.filterwarnings("ignore")

class VoiceEmotionDetector:
    def __init__(self):
        # Audio recording parameters
        self.RATE = 16000
        self.CHUNK = 1024
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        self.RECORD_SECONDS = 1  # Analyze 1-second chunks
        # Use absolute path for temporary file
        self.TEMP_WAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion_audio_temp.wav")
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Emotions to detect
        self.emotions = ["happy", "neutral", "sad", "angry"]
        
        print("🎭 Voice Emotion Detector initialized")
    
    def record_audio(self):
        """Records audio for RECORD_SECONDS and saves to a temporary WAV file."""
        stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS, 
                                rate=self.RATE, input=True, 
                                frames_per_buffer=self.CHUNK)
        
        frames = []
        for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        with wave.open(self.TEMP_WAV, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
        
        return self.TEMP_WAV
    
    def detect_emotion(self):
        """
        Detects emotion from recorded audio.
        Returns a tuple of (emotion, confidence_score)
        """
        try:
            # Record audio
            audio_file = self.record_audio()
            
            # Load audio using librosa
            audio, sr = librosa.load(audio_file, sr=self.RATE)
            
            # Extract features
            f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'),
                                                        fmax=librosa.note_to_hz('C7'))
            
            # Calculate basic features
            energy = np.mean(librosa.feature.rms(y=audio))
            zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y=audio))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            
            # Simple rules-based emotion detection
            if energy > 0.01:  # High energy
                if zero_crossing > 0.2:  # High zero crossing rate
                    emotion = "angry"
                    confidence = 0.7
                else:
                    emotion = "happy"
                    confidence = 0.6
            else:  # Low energy
                if spectral_centroid < 1000:
                    emotion = "sad"
                    confidence = 0.6
                else:
                    emotion = "neutral"
                    confidence = 0.8
            
            # Clean up
            if os.path.exists(audio_file):
                os.remove(audio_file)
                
            return emotion, confidence
            
        except Exception as e:
            print(f"Error in voice emotion detection: {str(e)}")
            return "neutral", 0.5
    
    def cleanup(self):
        """Clean up resources."""
        self.audio.terminate()
        if os.path.exists(self.TEMP_WAV):
            os.remove(self.TEMP_WAV)

# Example usage
if __name__ == "__main__":
    detector = VoiceEmotionDetector()
    try:
        print("🎙️ Listening for emotions... (Press Ctrl+C to stop)")
        while True:
            emotion, confidence = detector.detect_emotion()
            print(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping emotion detection")
    finally:
        detector.cleanup()
