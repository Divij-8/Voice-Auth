# Voice Guardian: Speaker Authentication System

Problem Statement: 3B - Voice Guardian  
Task: Single-speaker binary classification (Target vs. Non-Target)

## Features

- **Speaker Enrollment**: Enroll a target speaker using 10-20 audio samples
- **Voice Verification**: Verify if a new voice sample belongs to the enrolled speaker
- **Web Interface**: User-friendly Streamlit interface with audio recording
- **Real-time Authentication**: Get instant verification results with confidence scores

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. **Enroll Target Speaker**:
   - Place 10-20 `.wav` files of the target speaker in the `target_audio_files/` directory
   - Click "Enroll Speaker" in the sidebar
   - Wait for enrollment to complete

3. **Verify Speaker**:
   - Click the microphone icon to record your voice
   - Click "Verify My Voice" to authenticate
   - View the authentication result and confidence score

### Using the VoiceGuardian Class Directly

```python
from voice_guardian import VoiceGuardian

# Initialize
guardian = VoiceGuardian(threshold=0.85)

# Enroll target speaker
target_files = ["path/to/sample1.wav", "path/to/sample2.wav", ...]
guardian.enroll_target_speaker(target_files)

# Verify a speaker
result = guardian.verify_speaker("path/to/test_audio.wav")
print(result)  # {"status": "Authenticated", "confidence": 0.92}
```

## Technical Details

- **Model**: SpeechBrain ECAPA-TDNN (pre-trained speaker recognition model)
- **Audio Format**: 16kHz sample rate, mono WAV files
- **Similarity Metric**: Cosine similarity between speaker embeddings
- **Default Threshold**: 0.8 (configurable)

## File Structure

```
Voice Auth/
├── app.py                      # Streamlit web application
├── voice_guardian.py           # VoiceGuardian class implementation
├── requirements.txt            # Python dependencies
├── target_audio_files/         # Directory for target speaker samples
└── pretrained_models/          # Downloaded model files (auto-created)
```

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## Notes

- The first run will download the ECAPA-TDNN model (~100MB)
- Audio samples should be clear speech recordings
- More enrollment samples generally improve accuracy
- Adjust the threshold parameter to balance security vs. convenience
