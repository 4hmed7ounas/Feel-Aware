import os
import time
import threading
import whisper
import pyaudio
import wave
import numpy as np
import librosa
import warnings
import google.generativeai as genai
from elevenlabs import generate, save, set_api_key, Voice, VoiceSettings
from voice_emotion_detector import VoiceEmotionDetector
from text_sentiment_checker import TextSentimentChecker
from tone_switcher import ToneSwitcher

warnings.filterwarnings("ignore")

class IntegratedSystem:
    def __init__(self, gemini_api_key=None, elevenlabs_api_key=None):
        # Set API keys
        self.gemini_api_key = gemini_api_key
        self.elevenlabs_api_key = elevenlabs_api_key
        
        # Audio recording parameters
        self.RATE = 16000
        self.CHUNK = 1024
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        self.RECORD_SECONDS = 5
        self.TEMP_WAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "integrated_audio_temp.wav")
        self.RESPONSE_AUDIO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_response.wav")
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Load Whisper model for transcription
        self.whisper_model = whisper.load_model("tiny")
        
        # Initialize components
        self.voice_detector = VoiceEmotionDetector()
        self.text_checker = TextSentimentChecker()
        self.tone_switcher = ToneSwitcher()
        
        # Start the tone switcher
        self.tone_switcher.start()
        
        # Initialize Gemini if API key is provided
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        else:
            print("No Gemini API key provided. Response generation will be simulated.")
            self.gemini_model = None
        
        # Initialize ElevenLabs if API key is provided
        if self.elevenlabs_api_key:
            set_api_key(self.elevenlabs_api_key)
        else:
            print("No ElevenLabs API key provided. Voice synthesis will be simulated.")
        
        # Conversation history
        self.conversation_history = []
    
    def record_audio(self):
        """Records audio for RECORD_SECONDS and saves to a temporary WAV file."""
        try:
            stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS, 
                                    rate=self.RATE, input=True, 
                                    frames_per_buffer=self.CHUNK)
            
            print(f"Recording for {self.RECORD_SECONDS} second(s)...")
            frames = []
            for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            with wave.open(self.TEMP_WAV, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(frames))
            
            return self.TEMP_WAV
        except Exception as e:
            print(f"Error in recording audio: {str(e)}")
            return None
    
    def transcribe_audio(self, filename):
        """Transcribes audio using Whisper"""
        try:
            if not os.path.exists(filename):
                print(f"Audio file not found: {filename}")
                return ""
                        
            # Use a different approach for audio loading
            audio, sr = librosa.load(filename, sr=16000)
            
            # Get the log mel spectrogram
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio)).to(self.whisper_model.device)
            
            # Detect the spoken language
            _, probs = self.whisper_model.detect_language(mel)
            
            # Decode the audio
            options = whisper.DecodingOptions(fp16=False)
            result = whisper.decode(self.whisper_model, mel, options)
            
            return result.text
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            return ""
    
    def generate_response(self, transcript):
        """Generates a response using Gemini"""
        if not transcript or transcript.strip() == "":
            return "I didn't catch that. Could you please repeat?"
        
        try:
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": transcript})
            
            if self.gemini_model:
                # Create prompt with conversation history
                prompt = "You are a helpful and empathetic AI assistant. Respond to the following message in 1 to 2 lines:\n\n"
                
                # Add last few conversation turns for context (limit to 5 turns)
                for message in self.conversation_history[-5:]:
                    role = "User" if message["role"] == "user" else "Assistant"
                    prompt += f"{role}: {message['content']}\n"
                
                # Generate response
                response = self.gemini_model.generate_content(prompt)
                ai_response = response.text
            else:
                # Simulate response if no API key
                ai_response = f"This is a simulated response to: '{transcript}'"
            
            # Add AI response to conversation history
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            return ai_response
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I'm having trouble generating a response right now."
    
    def synthesize_voice(self, text, tone):
        try:
            if self.elevenlabs_api_key:
                try:
                    # Use a single voice ID (you can change this to your preferred voice)
                    voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice (neutral female)
                    
                    # Format text with emotion
                    # Include the emotion directly in the text
                    if tone["style"] == "calm":
                        emotion_suffix = ", they said calmly and reassuringly."
                    elif tone["style"] == "cheerful":
                        emotion_suffix = ", they said with enthusiasm and excitement!"
                    else:  # neutral
                        emotion_suffix = ", they said in a neutral tone."
                    
                    # Adjust for rate in the emotion description
                    if tone["rate"] == "slow":
                        emotion_suffix = emotion_suffix.replace("said", "said slowly")
                    elif tone["rate"] == "fast":
                        emotion_suffix = emotion_suffix.replace("said", "said quickly")
                    
                    # Adjust for pitch in the emotion description
                    if tone["pitch"] == "high":
                        emotion_suffix = emotion_suffix.replace("said", "said with a higher pitch")
                    elif tone["pitch"] == "low":
                        emotion_suffix = emotion_suffix.replace("said", "said with a lower pitch")
                    
                    print(f"Using emotion suffix: {emotion_suffix}")
                    
                    # Create voice settings optimized for emotion-based handling
                    voice_settings = VoiceSettings(
                        stability=0.3,  # Lower stability for more expressiveness
                        similarity_boost=0.75,
                        style=0.7,  # Higher style for more character
                        use_speaker_boost=True
                    )
                    
                    # Format text with emotion suffix
                    # This is the format that works best with ElevenLabs
                    modified_text = f'"{text}"{emotion_suffix}'
                    
                    # Set the API key explicitly before generating
                    set_api_key(self.elevenlabs_api_key)
                    
                    # Generate audio with emotion-based handling
                    audio = generate(
                        text=modified_text,
                        voice=Voice(
                            voice_id=voice_id,
                            settings=voice_settings
                        ),
                        model="eleven_monolingual_v1"  # Best model for this technique
                    )
                    
                    # Save audio file
                    save(audio, self.RESPONSE_AUDIO)
                    return True
                    
                except Exception as e:
                    error_message = str(e)
                    print(f"ElevenLabs API error: {error_message}")
                    
                    # Check if this is a free tier restriction error
                    if "Free Tier usage disabled" in error_message or "Unusual activity detected" in error_message:
                        print("\n⚠️ ElevenLabs free tier restriction detected. Switching to simulated voice mode.")
                        print("To use ElevenLabs voice synthesis, you may need to upgrade to a paid plan.")
                        # Disable ElevenLabs for the rest of the session
                        self.elevenlabs_disabled = True
                    else:
                        print("\n⚠️ Temporary error with ElevenLabs. Falling back to simulated voice for this response.")
            
            # Simulate voice synthesis (fallback or if API key not provided)
            print(f"\n[Simulated Voice] Speaking with {tone['style']} tone, {tone['rate']} rate, {tone['pitch']} pitch:")
            print(f"'{text}'")
            return False
            
        except Exception as e:
            print(f"Error in voice synthesis fallback: {str(e)}")
            return False
    
    def play_audio(self, filename):
        """Plays audio file"""
        if not os.path.exists(filename):
            print(f"Audio file not found: {filename}")
            return
        
        try:
            # Open the audio file
            wf = wave.open(filename, 'rb')
            
            # Create stream
            stream = self.audio.open(
                format=self.audio.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            
            # Read data
            data = wf.readframes(self.CHUNK)
            
            # Play audio
            while len(data) > 0:
                stream.write(data)
                data = wf.readframes(self.CHUNK)
            
            # Close stream
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Error playing audio: {str(e)}")
    
    def process_interaction(self):
        """Process a single interaction"""
        # Record audio
        audio_file = self.record_audio()
        if not audio_file or not os.path.exists(audio_file):
            print("❌ Failed to record audio")
            return
        
        # Detect voice emotion
        Voice_emotion, Voice_confidence = self.voice_detector.detect_emotion_from_file(audio_file)
        print(f"Voice Emotion: {Voice_emotion.upper()} (confidence: {Voice_confidence:.2f})")
        
        # Transcribe audio
        transcript = self.transcribe_audio(audio_file).strip()
        
        if not transcript:
            print("⚠️ No speech detected or transcription failed")
            return
        
        print(f"📝 Transcript: \"{transcript}\"")
        
        # Analyze text sentiment
        sentiment_score, sentiment_label = self.text_checker.analyze_transcript(transcript)
        
        if sentiment_score is not None:
            print(f"Text Sentiment: {sentiment_label.upper()} (score: {sentiment_score:.2f})")
        else:
            print("⚠️ Could not analyze text sentiment")
        
        # Update tone switcher with both inputs
        self.tone_switcher.update_transcript(transcript)
        
        # Get current tone
        current_tone = self.tone_switcher.get_current_tone()
        
        # Generate response
        response_text = self.generate_response(transcript)
        print(f"AI Response: \"{response_text}\"")
        
        # Generate SSML
        ssml = self.tone_switcher.generate_ssml(response_text)
        
        print(f"Selected tone: {current_tone['style']} (rate: {current_tone['rate']}, pitch: {current_tone['pitch']})")
        
        # Synthesize voice
        voice_success = self.synthesize_voice(response_text, current_tone)
        
        # Play audio response if synthesis was successful
        if voice_success:
            self.play_audio(self.RESPONSE_AUDIO)
        
        # Clean up
        if os.path.exists(audio_file):
            os.remove(audio_file)
    
    def start(self):
        """Start the integrated system"""
        print("Speak into the microphone when prompted.")
        print("Press Ctrl+C to exit.")
        
        try:
            while True:
                print("\n----- New Interaction -----")
                self.process_interaction()
                print("\nReady for next interaction in 2 seconds...")
                time.sleep(2)
        
        except KeyboardInterrupt:
            print("\n\n✅ System stopped. Exiting...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.audio.terminate()
        self.tone_switcher.cleanup()
        if os.path.exists(self.TEMP_WAV):
            os.remove(self.TEMP_WAV)
        if os.path.exists(self.RESPONSE_AUDIO):
            os.remove(self.RESPONSE_AUDIO)

if __name__ == "__main__":
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("⚠️ dotenv package not found. Using environment variables directly.")
    
    # Get API keys from environment variables
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
    
    system = IntegratedSystem(gemini_api_key, elevenlabs_api_key)
    system.start()
