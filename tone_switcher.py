import time
import threading
import queue
import os
from voice_emotion_detector import VoiceEmotionDetector
from text_sentiment_checker import TextSentimentChecker

class ToneSwitcher:
    def __init__(self):
        # Initialize components
        self.voice_detector = VoiceEmotionDetector()
        self.text_checker = TextSentimentChecker()
        
        # Queues for communication between threads
        self.transcript_queue = queue.Queue()
        self.voice_emotion_queue = queue.Queue()
        self.tone_queue = queue.Queue()
        
        # Current state
        self.current_tone = "neutral"
        self.current_voice_emotion = "neutral"
        self.current_text_sentiment = "neutral"
        self.current_transcript = ""
        
        # Weights for decision making
        self.voice_emotion_weight = 0.7  # Voice emotion has higher priority
        self.text_sentiment_weight = 0.3
        
        # Mapping of emotions/sentiments to TTS styles
        self.tone_mapping = {
            # Voice emotion based mappings
            "angry": {"style": "calm", "rate": "slow", "pitch": "low"},
            "sad": {"style": "gentle", "rate": "medium", "pitch": "medium"},
            "happy": {"style": "cheerful", "rate": "medium", "pitch": "high"},
            "neutral": {"style": "neutral", "rate": "medium", "pitch": "medium"},
            
            # Text sentiment based mappings
            "very_negative": {"style": "calm", "rate": "slow", "pitch": "low"},
            "negative": {"style": "gentle", "rate": "medium-slow", "pitch": "medium-low"},
            "positive": {"style": "friendly", "rate": "medium", "pitch": "medium-high"},
            "very_positive": {"style": "cheerful", "rate": "medium-fast", "pitch": "high"},
        }
        
        print("🎭 Tone Switcher initialized")
    
    def start(self):
        """Start all the processing threads"""
        # Start voice emotion detection thread
        self.voice_thread = threading.Thread(target=self._voice_emotion_loop)
        self.voice_thread.daemon = True
        self.voice_thread.start()
        
        # Start tone decision thread
        self.tone_thread = threading.Thread(target=self._tone_decision_loop)
        self.tone_thread.daemon = True
        self.tone_thread.start()
        
        print("🟢 Tone Switcher running - press Ctrl+C to stop")
    
    def _voice_emotion_loop(self):
        """Thread that continuously detects voice emotion"""
        while True:
            try:
                emotion, confidence = self.voice_detector.detect_emotion()
                self.voice_emotion_queue.put((emotion, confidence))
                time.sleep(0.1)  # Small delay to prevent CPU overuse
            except Exception as e:
                print(f"Error in voice emotion detection: {e}")
                time.sleep(1)  # Longer delay on error
    
    def _tone_decision_loop(self):
        """Thread that makes tone decisions based on inputs"""
        while True:
            try:
                # Process voice emotion
                try:
                    emotion, confidence = self.voice_emotion_queue.get(block=False)
                    self.current_voice_emotion = emotion
                    print(f"Voice emotion: {emotion} (confidence: {confidence:.2f})")
                except queue.Empty:
                    pass
                
                # Process text sentiment if we have a transcript
                if self.current_transcript:
                    score, label = self.text_checker.analyze_transcript(self.current_transcript)
                    if score is not None and label is not None:
                        self.current_text_sentiment = label
                        print(f"Text sentiment: {label} (score: {score:.2f})")
                
                # Make tone decision
                new_tone = self._decide_tone()
                if new_tone != self.current_tone:
                    self.current_tone = new_tone
                    self.tone_queue.put(new_tone)
                    print(f"🔄 Tone switched to: {new_tone['style']} (rate: {new_tone['rate']}, pitch: {new_tone['pitch']})")
                
                time.sleep(0.5)  # Check for tone changes every 0.5 seconds
            except Exception as e:
                print(f"Error in tone decision: {e}")
                time.sleep(1)
    
    def _decide_tone(self):
        """
        Decide which tone to use based on voice emotion and text sentiment
        Returns a tone settings dictionary
        """
        # Get tone settings for current voice emotion and text sentiment
        voice_tone = self.tone_mapping.get(self.current_voice_emotion, self.tone_mapping["neutral"])
        text_tone = self.tone_mapping.get(self.current_text_sentiment, self.tone_mapping["neutral"])
        
        # Priority rules:
        # 1. If voice emotion is angry or text sentiment is very_negative, use calm tone
        if self.current_voice_emotion == "angry" or self.current_text_sentiment == "very_negative":
            return self.tone_mapping["angry"]
        
        # 2. If voice emotion is happy or text sentiment is very_positive, use cheerful tone
        if self.current_voice_emotion == "happy" or self.current_text_sentiment == "very_positive":
            return self.tone_mapping["happy"]
        
        # 3. Otherwise, blend the two tones based on weights
        # For simplicity in this example, we'll just use the voice_tone if it's not neutral
        # In a real implementation, you might blend parameters more granularly
        if self.current_voice_emotion != "neutral":
            return voice_tone
        elif self.current_text_sentiment != "neutral":
            return text_tone
        else:
            return self.tone_mapping["neutral"]
    
    def update_transcript(self, transcript):
        """Update the current transcript"""
        self.current_transcript = transcript
    
    def get_current_tone(self):
        """Get the current tone settings"""
        return self.current_tone
    
    def generate_ssml(self, text, tone=None):
        """
        Generate SSML markup for the given text and tone
        If tone is None, uses the current tone
        """
        if tone is None:
            tone = self.current_tone
            
        # Create SSML with appropriate voice style, rate, and pitch
        ssml = f"""<speak>
  <voice name="en-US-Neural2-F">
    <prosody rate="{tone['rate']}" pitch="{tone['pitch']}">
      <mstts:express-as style="{tone['style']}">
        {text}
      </mstts:express-as>
    </prosody>
  </voice>
</speak>"""
        return ssml
    
    def cleanup(self):
        """Clean up resources"""
        self.voice_detector.cleanup()
        print("🛑 Tone Switcher stopped")

# Example usage with SSML output for TTS
if __name__ == "__main__":
    import whisper
    import pyaudio
    import wave
    
    # Setup for audio recording
    RATE = 16000
    CHUNK = 1024
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    RECORD_SECONDS = 2
    TEMP_WAV = "temp_audio.wav"
    
    # Load Whisper model for transcription
    print("⏳ Loading Whisper model...")
    whisper_model = whisper.load_model("tiny")
    
    # Initialize tone switcher
    tone_switcher = ToneSwitcher()
    tone_switcher.start()
    
    def record_audio(filename):
        """Records audio and saves to a WAV file"""
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                            input=True, frames_per_buffer=CHUNK)
        
        print("🎙️ Listening...")
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
    
    def transcribe_audio(filename):
        """Transcribes audio using Whisper"""
        result = whisper_model.transcribe(filename)
        return result["text"]
    
    try:
        print("🟢 Feel-Aware Tone Switcher Running...\n")
        print("Speak into the microphone to test different tones\n")
        
        while True:
            # Record and transcribe audio
            record_audio(TEMP_WAV)
            transcript = transcribe_audio(TEMP_WAV).strip()
            
            if not transcript:
                print("⚠️  No speech detected.\n")
                continue
            
            # Update transcript in tone switcher
            tone_switcher.update_transcript(transcript)
            
            # Get current tone and generate SSML
            current_tone = tone_switcher.get_current_tone()
            ssml = tone_switcher.generate_ssml(transcript)
            
            # Print transcript and tone information
            print(f"🗣 Transcript: \"{transcript}\"")
            print(f"🎭 Current tone: {current_tone['style']} (rate: {current_tone['rate']}, pitch: {current_tone['pitch']})")
            print(f"📢 SSML Output:\n{ssml}\n")
            
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\nStopping Feel-Aware Tone Switcher")
    finally:
        tone_switcher.cleanup()
        if os.path.exists(TEMP_WAV):
            os.remove(TEMP_WAV)
