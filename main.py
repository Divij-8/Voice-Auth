"""
FastAPI Backend for Voice Guardian Authentication System (Enhanced)
Uses Cross-Encoder for improved accuracy at 0.75-0.85 threshold
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
from pathlib import Path
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from io import BytesIO

# Import the enhanced VoiceGuardian with Cross-Encoder
from voice_guardian_enhanced import VoiceGuardianEnhanced

app = FastAPI(title="Voice Guardian API - Enhanced")

# Enable CORS with explicit configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# Initialize the Enhanced Voice Guardian with Cross-Encoder
# Threshold optimized to 0.80 for better accuracy with cross-encoder
guardian = VoiceGuardianEnhanced(
    threshold=0.80,  # Optimized threshold for cross-encoder (was 0.75)
    use_cross_encoder=True  # Enable cross-encoder for better accuracy
)

# Global flag to track enrollment status
is_enrolled = False


@app.on_event("startup")
async def startup_event():
    """
    Auto-enroll the target speaker on startup.
    Place your training audio files in a folder called 'training_data'
    """
    global is_enrolled
    
    training_dir = Path("training_data")
    
    if training_dir.exists():
        # Get all audio files from training directory
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        target_files = [
            str(f) for f in sorted(training_dir.iterdir())
            if f.suffix.lower() in audio_extensions
        ]
        
        if target_files:
            try:
                print("\n" + "="*70)
                print("ENROLLING TARGET SPEAKER WITH CROSS-ENCODER ENHANCEMENT")
                print("="*70)
                print(f"Found {len(target_files)} audio files for enrollment")
                guardian.enroll_target_speaker(target_files)
                is_enrolled = True
                print("\n✅ System ready for authentication with enhanced accuracy!")
                print("="*70 + "\n")
            except Exception as e:
                print(f"\n❌ Enrollment failed: {e}\n")
                is_enrolled = False
        else:
            print("\n⚠️  No audio files found in 'training_data' directory")
            is_enrolled = False
    else:
        print("\n⚠️  'training_data' directory not found. Please create it and add your voice samples.")
        is_enrolled = False


@app.options("/{full_path:path}")
async def preflight_handler(full_path: str):
    """Handle CORS preflight requests"""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
    )

@app.get("/")
async def root():
    """Health check endpoint"""
    stats = guardian.get_stats()
    return {
        "status": "online",
        "message": "Voice Guardian API - Enhanced with Cross-Encoder",
        "enrolled": is_enrolled,
        "cross_encoder": stats.get('cross_encoder_enabled', False),
        "threshold": stats.get('threshold', 0.75)
    }


@app.get("/api/status")
async def get_status():
    """Get system status"""
    try:
        stats = guardian.get_stats()
        return {
            "enrolled": is_enrolled,
            "stats": stats,
            "enhancement": "Cross-Encoder Enabled" if stats.get('cross_encoder_enabled') else "Baseline"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


def convert_audio_to_wav(audio_blob: bytes, filename: str) -> str:
    """
    Convert audio blob from browser (webm, mp3, etc.) to standard WAV format.
    
    Args:
        audio_blob: Raw audio bytes from browser
        filename: Original filename to detect format
        
    Returns:
        str: Path to converted WAV file (mono, 16kHz, 16-bit PCM)
        
    Raises:
        Exception: If conversion fails
    """
    temp_input = None
    try:
        # Detect audio format from filename
        file_ext = Path(filename).suffix.lower() if filename else '.webm'
        if not file_ext:
            file_ext = '.webm'
        
        # Save blob to temporary file with detected extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(audio_blob)
            temp_input = temp_file.name
        
        print(f"[DEBUG] Input file: {temp_input}, extension: {file_ext}")
        
        # Load audio with pydub - try multiple formats
        audio = None
        format_ext = file_ext.lstrip('.').lower()
        
        # List of formats to try in order
        formats_to_try = []
        if format_ext and format_ext != 'wav':
            formats_to_try.append(format_ext)
        formats_to_try.extend(['webm', 'ogg', 'mp3', 'wav', 'm4a', 'aac'])
        
        # Try each format
        for fmt in formats_to_try:
            try:
                print(f"[DEBUG] Trying format: {fmt}")
                audio = AudioSegment.from_file(temp_input, format=fmt)
                print(f"[DEBUG] Successfully loaded as {fmt}")
                break
            except Exception as e:
                print(f"[DEBUG] Format {fmt} failed: {str(e)[:50]}")
                continue
        
        if audio is None:
            raise Exception(f"Could not detect audio format. Tried: {', '.join(formats_to_try)}")
        
        print(f"[DEBUG] Audio info - channels: {audio.channels}, frame_rate: {audio.frame_rate}, duration: {len(audio)}ms")
        
        # Convert to mono if stereo
        if audio.channels > 1:
            print(f"[DEBUG] Converting from {audio.channels} channels to mono")
            audio = audio.set_channels(1)
        
        # Resample to 16kHz if necessary
        if audio.frame_rate != 16000:
            print(f"[DEBUG] Resampling from {audio.frame_rate}Hz to 16000Hz")
            audio = audio.set_frame_rate(16000)
        
        # Create output WAV file
        output_path = tempfile.mktemp(suffix='.wav')
        print(f"[DEBUG] Exporting to: {output_path}")
        
        audio.export(
            output_path,
            format='wav',
            parameters=['-acodec', 'pcm_s16le']  # 16-bit PCM
        )
        
        # Verify output file exists and has content
        if not os.path.exists(output_path):
            raise Exception("Output WAV file was not created")
        
        file_size = os.path.getsize(output_path)
        if file_size < 1000:
            raise Exception(f"Output WAV file is too small ({file_size} bytes)")
        
        print(f"[DEBUG] Successfully created WAV file: {output_path} ({file_size} bytes)")
        return output_path
        
    except Exception as e:
        print(f"[DEBUG] Audio conversion error: {str(e)}")
        raise
    finally:
        # Clean up input temp file
        if temp_input and os.path.exists(temp_input):
            try:
                os.unlink(temp_input)
            except:
                pass


@app.post("/api/verify")
async def verify_voice(audio: UploadFile = File(...)):
    """
    Verify if the uploaded audio matches the enrolled speaker.
    Converts browser audio blob to standard WAV format (mono, 16kHz, 16-bit PCM).
    Uses Cross-Encoder for enhanced accuracy.
    
    Returns:
        - authenticated: boolean
        - confidence: float (0-100)
        - similarity: float (combined score)
        - cosine_similarity: float (baseline cosine score)
        - cross_encoder_score: float (enhanced score)
        - message: string
    """
    if not is_enrolled:
        raise HTTPException(
            status_code=400, 
            detail="System not enrolled. Please add training data and restart."
        )
    
    converted_path = None
    try:
        # Read the audio file content
        content = await audio.read()
        
        if not content or len(content) < 100:
            raise HTTPException(
                status_code=400,
                detail="Audio file is empty or too small. Please record at least 2-3 seconds."
            )
        
        # Convert browser audio blob to standard WAV (mono, 16kHz, 16-bit PCM)
        try:
            converted_path = convert_audio_to_wav(content, audio.filename)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process audio format: {str(e)}. Please ensure your browser supports audio recording."
            )
        
        # Verify the speaker with enhanced cross-encoder
        result = guardian.verify_speaker(converted_path, return_details=True)
        
        # Convert confidence to percentage
        confidence_percentage = result['confidence'] * 100
        
        # Prepare detailed response
        response = {
            "authenticated": result['status'] == "Authenticated",
            "confidence": round(confidence_percentage, 2),
            "similarity": round(result['similarity'], 4),
            "cosine_similarity": round(result['cosine_similarity'], 4),
            "threshold": guardian.threshold,
            "scoring_method": result.get('scoring_method', 'Unknown'),
            "message": (
                f"✅ Voice Authenticated! Confidence: {confidence_percentage:.1f}%" 
                if result['status'] == "Authenticated" 
                else f"❌ Voice Rejected. Confidence: {confidence_percentage:.1f}%"
            )
        }
        
        # Add cross-encoder score if available
        if 'cross_encoder_score' in result:
            response['cross_encoder_score'] = round(result['cross_encoder_score'], 4)
            response['improvement'] = round(
                (result['similarity'] - result['cosine_similarity']) * 100, 2
            )
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Verification failed: {str(e)}"
        )
    
    finally:
        # Clean up converted audio file
        if converted_path and os.path.exists(converted_path):
            try:
                os.unlink(converted_path)
            except:
                pass


@app.post("/api/enroll")
async def enroll_speaker(audio_files: list[UploadFile] = File(...)):
    """
    Enroll a new speaker with multiple audio samples.
    Requires 10-20 audio files for best results.
    Trains cross-encoder for improved accuracy.
    """
    global is_enrolled
    
    if len(audio_files) < 5:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least 5 audio samples for enrollment (10-20 recommended for cross-encoder training)"
        )
    
    temp_files = []
    try:
        # Save all uploaded files temporarily
        for audio in audio_files:
            suffix = Path(audio.filename).suffix or '.wav'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                content = await audio.read()
                temp_file.write(content)
                temp_files.append(temp_file.name)
        
        # Enroll the speaker with cross-encoder training
        guardian.enroll_target_speaker(temp_files)
        is_enrolled = True
        
        stats = guardian.get_stats()
        
        return {
            "success": True,
            "message": "Speaker enrolled successfully with cross-encoder optimization!",
            "enrollment_count": stats['enrollment_count'],
            "cross_encoder_enabled": stats['cross_encoder_enabled'],
            "threshold": stats['threshold']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")
    
    finally:
        # Clean up temp files
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass


@app.post("/api/adjust-threshold")
async def adjust_threshold(new_threshold: float):
    """
    Adjust the verification threshold.
    
    Args:
        new_threshold: New threshold value (0.0 to 1.0)
    """
    if not 0.0 <= new_threshold <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Threshold must be between 0.0 and 1.0"
        )
    
    try:
        old_threshold = guardian.threshold
        guardian.threshold = new_threshold
        
        return {
            "success": True,
            "old_threshold": old_threshold,
            "new_threshold": new_threshold,
            "message": f"Threshold adjusted from {old_threshold:.2f} to {new_threshold:.2f}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Threshold adjustment failed: {str(e)}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("VOICE GUARDIAN - ENHANCED API SERVER WITH CROSS-ENCODER")
    print("="*70)
    print("\nEnhancements:")
    print("✓ Cross-Encoder neural network for better similarity scoring")
    print("✓ Improved accuracy at 0.75-0.85 threshold range")
    print("✓ Hybrid scoring: 40% Cosine + 60% Cross-Encoder")
    print("✓ Automatic optimization during enrollment")
    print("\nIntegration:")
    print("✓ FastAPI backend running on http://localhost:8000")
    print("✓ Frontend accessible via index.html")
    print("✓ CORS enabled for cross-origin requests")
    print("\nMake sure you have:")
    print("1. Created a 'training_data' folder")
    print("2. Added 10-20 audio samples of your voice (more is better)")
    print("3. Installed dependencies: pip install -r requirements.txt")
    print("\nEndpoints:")
    print("  GET  /                    - Health check")
    print("  GET  /api/status          - System status")
    print("  POST /api/verify          - Verify voice")
    print("  POST /api/enroll          - Enroll speaker")
    print("  POST /api/adjust-threshold - Adjust threshold")
    print("\nStarting server on http://localhost:8000")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)