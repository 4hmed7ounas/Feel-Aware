import os
import time
import threading
import json
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import base64
import io
from dotenv import load_dotenv
from integrated_system import IntegratedSystem

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'feel-aware-tone-switcher'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
integrated_system = None
is_running = False
processing_thread = None
conversation_history = []

# Initialize the integrated system
def initialize_integrated_system():
    global integrated_system
    
    print("⏳ Initializing Integrated Feel-Aware System...")
    
    # Get API keys from environment variables
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
    
    # Initialize the integrated system
    integrated_system = IntegratedSystem(gemini_api_key, elevenlabs_api_key)
    
    print("✅ Integrated Feel-Aware System initialized")

# Process audio data
def process_audio(audio_data):
    global conversation_history
    
    # Save audio data to a temporary file
    temp_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_audio_temp.wav")
    
    # Convert base64 to audio file
    try:
        # Extract the base64 data and decode
        audio_format = audio_data.split(';')[0].split(':')[1]
        print(f"Received audio format: {audio_format}")
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        
        # Save the original audio file with appropriate extension
        original_ext = audio_format.split('/')[1].split(';')[0]
        if ';' in original_ext:
            original_ext = original_ext.split(';')[0]
        
        temp_original = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"web_audio_temp.{original_ext}")
        with open(temp_original, 'wb') as f:
            f.write(audio_bytes)
        
        print(f"Saved original audio as: {temp_original}")
        
        # Convert to WAV format for processing if needed
        if original_ext != 'wav':
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(temp_original, format=original_ext)
                audio = audio.set_frame_rate(16000).set_channels(1)  # Convert to 16kHz mono for Whisper
                audio.export(temp_wav, format="wav")
                print(f"Converted audio to WAV format: {temp_wav}")
            except Exception as e:
                print(f"Error converting audio: {str(e)}")
                # If conversion fails, just use the original file
                with open(temp_wav, 'wb') as f:
                    f.write(audio_bytes)
        else:
            # Already WAV, just copy
            with open(temp_wav, 'wb') as f:
                f.write(audio_bytes)
    except Exception as e:
        print(f"Error processing audio data: {str(e)}")
        import traceback
        traceback.print_exc()
        # Fallback to direct save
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        with open(temp_wav, 'wb') as f:
            f.write(audio_bytes)
    
    # Process with integrated system
    try:
        # Detect voice emotion
        voice_emotion, voice_confidence = integrated_system.detect_voice_emotion(temp_wav)
        
        # Transcribe audio
        transcript = integrated_system.transcribe_audio(temp_wav)
        
        # If no transcript, return early
        if not transcript:
            return {
                "voice_emotion": voice_emotion,
                "voice_confidence": voice_confidence,
                "transcript": "No speech detected",
                "text_sentiment": "neutral",
                "text_sentiment_score": 0.0,
                "ai_response": "I didn't catch that. Could you please speak more clearly?",
                "selected_tone": {"style": "neutral", "rate": "medium", "pitch": "medium"}
            }
        
        # Add user message to conversation history
        conversation_history.append({"role": "user", "content": transcript})
        
        # Analyze text sentiment
        text_sentiment_score, text_sentiment = integrated_system.text_checker.analyze_transcript(transcript)
        
        # Update tone switcher
        integrated_system.tone_switcher.update_transcript(transcript)
        
        # Get current tone
        current_tone = integrated_system.tone_switcher.get_current_tone()
        
        # Generate AI response
        ai_response = integrated_system.generate_response(transcript)
        
        # Add AI response to conversation history
        conversation_history.append({"role": "assistant", "content": ai_response})
        
        # Limit conversation history to last 10 messages
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        
        # Synthesize voice (this will use the fallback if ElevenLabs is unavailable)
        voice_synthesized = integrated_system.synthesize_voice(ai_response, current_tone)
        
        # Clean up
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        
        # Return results
        return {
            "voice_emotion": voice_emotion,
            "voice_confidence": voice_confidence,
            "transcript": transcript,
            "text_sentiment": text_sentiment,
            "text_sentiment_score": text_sentiment_score,
            "ai_response": ai_response,
            "selected_tone": current_tone,
            "voice_synthesized": voice_synthesized
        }
    
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "voice_emotion": "error",
            "voice_confidence": 0,
            "transcript": f"Error: {str(e)}",
            "text_sentiment": "neutral",
            "text_sentiment_score": 0.0,
            "ai_response": "Sorry, there was an error processing your request.",
            "selected_tone": {"style": "neutral", "rate": "medium", "pitch": "medium"},
            "voice_synthesized": False
        }

# Routes
@app.route('/')
def index():
    return render_template('integrated_index.html')

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.emit('status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_system')
def handle_start_system():
    global is_running
    
    if not is_running:
        # Initialize integrated system if not already initialized
        if integrated_system is None:
            initialize_integrated_system()
        
        is_running = True
        socketio.emit('system_status', {'running': True})
        print("🟢 Integrated Feel-Aware System started")

@socketio.on('stop_system')
def handle_stop_system():
    global is_running
    
    if is_running:
        is_running = False
        
        # Clean up resources
        if integrated_system:
            integrated_system.cleanup()
        
        socketio.emit('system_status', {'running': False})
        print("🛑 Integrated Feel-Aware System stopped")

@socketio.on('process_audio')
def handle_audio(data):
    if not is_running:
        socketio.emit('error', {'message': 'System not running'})
        return
    
    # Process audio data
    result = process_audio(data['audio'])
    
    # Send results back to client
    socketio.emit('processing_result', result)

@socketio.on('get_conversation_history')
def handle_get_conversation_history():
    socketio.emit('conversation_history', {'history': conversation_history})

# Create templates directory if it doesn't exist
def ensure_templates_dir():
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    return templates_dir

# Main function
if __name__ == '__main__':
    # Ensure templates directory exists
    templates_dir = ensure_templates_dir()
    
    # Initialize integrated system
    initialize_integrated_system()
    
    # Run the Flask app
    print("🌐 Starting integrated web server at http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
