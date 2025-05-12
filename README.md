# Integrated Feel-Aware System

An advanced AI system that combines voice emotion detection, text sentiment analysis, and dynamic tone adjustment using ElevenLabs voice synthesis. The system provides a more empathetic and context-aware conversation experience through real-time emotional analysis and adaptive voice responses.

## System Architecture

The system is now integrated into a single cohesive application with these key components:

1. **Integrated System** (`integrated_system.py`)
   - Unified system that combines all functionality
   - Real-time voice emotion detection
   - Text sentiment analysis using Gemini
   - Dynamic tone adjustment based on emotional context
   - Voice synthesis with ElevenLabs
   - Fallback to simulated voice when ElevenLabs is unavailable

2. **Web Interface** (`integrated_web_app.py` and `templates/integrated_index.html`)
   - Modern web interface for real-time interaction
   - Real-time audio recording and processing
   - Visual feedback for detected emotions and sentiment
   - Conversation history tracking
   - Voice synthesis status monitoring

## Features

- Real-time voice emotion detection
- Text sentiment analysis using Gemini AI
- Dynamic voice tone adjustment
- ElevenLabs voice synthesis with emotion handling
- Fallback to simulated voice when ElevenLabs is unavailable
- Web-based interface with real-time feedback
- Conversation history tracking
- Detailed logging and debugging information

## Installation

1. Install Python 3.8 or higher
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
GEMINI_API_KEY=your_gemini_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

## Usage

### Command Line
Run the integrated system directly:
```bash
python integrated_system.py
```

## System Flow

1. Audio Recording
   - Records audio in 5-second chunks
   - Real-time volume monitoring
   - Automatic format conversion for processing

2. Emotion Detection
   - Voice emotion detection from audio
   - Text sentiment analysis from transcription
   - Combined emotional context analysis

3. Tone Adjustment
   - Dynamic tone selection based on emotional context
   - Adjustable parameters: style, rate, pitch
   - Smooth transitions between tones

4. Voice Synthesis
   - ElevenLabs voice synthesis with emotion handling
   - Fallback to simulated voice when needed
   - Real-time voice playback

## Tone Mapping

The system maps detected emotions and sentiments to appropriate voice styles:

- **Angry/Very Negative**: Calm, slow rate, low pitch
- **Sad/Negative**: Gentle, medium-slow rate, medium-low pitch
- **Happy/Positive**: Cheerful/Friendly, medium rate, medium-high pitch
- **Neutral**: Neutral, medium rate, medium pitch

## Requirements

- Python 3.8+
- ElevenLabs API key (for voice synthesis)
- Gemini API key (for text analysis)
- FFmpeg (for audio processing)
- A modern web browser (for web interface)

## Troubleshooting

1. If you encounter "No speech detected" errors:
   - Ensure your microphone is working
   - Check that the audio volume is sufficient
   - Verify that the audio format is supported

2. If ElevenLabs voice synthesis fails:
   - Check your API key
   - Verify your subscription status
   - The system will automatically fall back to simulated voice

3. If you encounter any other issues:
   - Check the logs in both the browser console and server terminal
   - The system provides detailed debugging information for troubleshooting

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.