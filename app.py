"""
Streamlit Web App for Voice Guardian: Speaker Authentication
Problem Statement: 3B - Voice Guardian
"""

import streamlit as st
import os
import glob
import soundfile as sf
import numpy as np
from voice_guardian import VoiceGuardian
from audio_recorder_streamlit import audio_recorder
from audio_storage import audio_storage
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Voice Guardian: Speaker Authentication",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .authenticated {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .rejected {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for recording management
if "current_recording" not in st.session_state:
    st.session_state.current_recording = None
if "previous_recording" not in st.session_state:
    st.session_state.previous_recording = None
if "recording_key" not in st.session_state:
    st.session_state.recording_key = 0
if "guardian" not in st.session_state:
    st.session_state.guardian = None
if "enrollment_complete" not in st.session_state:
    st.session_state.enrollment_complete = False
if "last_audio_data" not in st.session_state:
    st.session_state.last_audio_data = None
if "rec_session" not in st.session_state:
    st.session_state.rec_session = 0

# Load the model ONCE using Streamlit's cache
@st.cache_resource
def load_guardian():
    """Load the VoiceGuardian model (cached for performance)."""
    return VoiceGuardian(threshold=0.80)

st.session_state.guardian = load_guardian()
guardian = st.session_state.guardian

# Define the directory for target speaker samples
TARGET_AUDIO_DIR = "target_audio_files/"
os.makedirs(TARGET_AUDIO_DIR, exist_ok=True)

# ==================== SIDEBAR: ENROLLMENT SECTION ====================
with st.sidebar:
    st.header("🎤 Enrollment")
    st.divider()
    
    # Threshold adjustment
    new_threshold = st.slider(
        "Verification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=guardian.threshold,
        step=0.05,
        help="Higher = stricter authentication. Recommended: 0.80-0.90"
    )
    guardian.threshold = new_threshold
    
    st.divider()
    st.subheader("📁 Upload Audio Files")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload 10-20 .wav files for enrollment",
        type=['wav'],
        accept_multiple_files=True,
        help="Upload multiple .wav files (10-20 recommended)"
    )
    
    # Display upload status
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) selected")
        for file in uploaded_files:
            st.caption(f"📄 {file.name}")
    
    st.divider()
    
    # Enrollment button
    if st.button("🚀 Enroll Speaker", use_container_width=True):
        if not uploaded_files:
            st.error("❌ Please upload at least one audio file")
        else:
            all_audio_files = []
            
            # Save uploaded files temporarily
            with st.spinner("Processing uploaded files..."):
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(TARGET_AUDIO_DIR, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    all_audio_files.append(temp_path)
            
            # Perform enrollment
            with st.spinner("Enrolling speaker... This may take a moment."):
                try:
                    guardian.enroll_target_speaker(all_audio_files)
                    st.session_state.enrollment_complete = True
                    
                    # Save to enrollment storage
                    for i, file_path in enumerate(all_audio_files, 1):
                        try:
                            audio_data, sr = sf.read(file_path)
                            audio_storage.save_enrollment_recording(
                                audio_data,
                                sample_rate=sr,
                                speaker_id="target_speaker",
                                sample_num=i
                            )
                        except Exception as e:
                            st.warning(f"Could not store enrollment recording {i}: {str(e)}")
                    
                    st.success(f"✅ Enrollment complete with {len(all_audio_files)} files!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Enrollment failed: {str(e)}")
    
    st.divider()
    
    # System stats
    stats = guardian.get_stats()
    st.subheader("📊 System Status")
    st.metric("Status", "✅ Ready" if stats['enrolled'] else "⏳ Waiting")
    st.metric("Enrolled Samples", stats['enrollment_count'])
    st.metric("Threshold", f"{stats['threshold']:.2f}")
    st.metric("Device", stats['device'].upper())

# ==================== MAIN PAGE: VERIFICATION SECTION ====================
st.title("🎤 Voice Guardian Authentication System")

if not st.session_state.enrollment_complete or guardian.target_voiceprint is None:
    st.warning("⚠️ **Please enroll the target speaker first using the sidebar.**")
    st.info("""
    **How to enroll:**
    1. Prepare 10-20 audio samples (.wav files) of the target speaker
    2. Upload them using the sidebar
    3. Click "Enroll Speaker"
    4. Wait for enrollment to complete
    """)
else:
    st.success("✅ **System Ready. Record your voice for verification.**")
    st.divider()
    
    # Recording section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎙️ Record Your Voice")
        st.info("💡 **Tip:** Click record, speak clearly for 3-5 seconds, then wait for auto-stop (or click stop).")
        
        # Audio recorder with session-specific key
        audio_data = audio_recorder(
            pause_threshold=1.5,  # Shorter pause for better detection
            sample_rate=16000,
            text="🎤 Click to record",
            recording_color="#e74c3c",
            neutral_color="#1f77b4",
            icon_name="microphone",
            icon_size="3x",
            key=f"recorder_session_{st.session_state.rec_session}"
        )
        
        # Add control buttons
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🔄 New Recording", help="Clear and start fresh", use_container_width=True):
                st.session_state.rec_session += 1
                st.session_state.last_audio_data = None
                st.rerun()
        with col_b:
            st.caption(f"Session: {st.session_state.rec_session}")
        with col_c:
            # Debug mode toggle
            if "debug_mode" not in st.session_state:
                st.session_state.debug_mode = False
            if st.button("🔍 Debug", help="Show debug info", use_container_width=True):
                st.session_state.debug_mode = not st.session_state.debug_mode
    
    with col2:
        st.subheader("📋 Guidelines")
        st.info("""
        - Record **3-5 seconds**
        - Speak clearly and naturally
        - Use same microphone as enrollment
        - Minimize background noise
        - Avoid interruptions
        """)
    
    # Debug information (if enabled)
    if st.session_state.debug_mode:
        with st.expander("🔍 Debug Information", expanded=True):
            st.write(f"**Audio data type:** {type(audio_data)}")
            st.write(f"**Audio data is None:** {audio_data is None}")
            if audio_data is not None:
                st.write(f"**Audio data length:** {len(audio_data)} bytes")
                st.write(f"**First 20 bytes:** {audio_data[:20] if len(audio_data) > 20 else audio_data}")
                st.write(f"**Last audio data same:** {audio_data == st.session_state.last_audio_data}")
    
    st.divider()
    
    # Check if we have audio data (new or existing from this session)
    if audio_data is not None and len(audio_data) > 100:
        # Show audio player and download for ANY recorded audio
        st.subheader("🎧 Recorded Audio")
        
        col_audio1, col_audio2 = st.columns([2, 1])
        
        with col_audio1:
            # Play audio
            st.audio(audio_data, format="audio/wav")
        
        with col_audio2:
            # Download button
            st.download_button(
                label="📥 Download",
                data=audio_data,
                file_name=f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
                mime="audio/wav",
                use_container_width=True
            )
        
        # Show duration info
        try:
            import io
            import librosa
            # Load from bytes
            audio_signal, sr = sf.read(io.BytesIO(audio_data))
            duration = len(audio_signal) / sr
            
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.caption(f"🎵 Duration: **{duration:.2f}s**")
            with col_info2:
                st.caption(f"📦 Size: **{len(audio_data)} bytes**")
            with col_info3:
                if duration < 0.5:
                    st.caption("⚠️ Too short!")
                elif duration < 2.0:
                    st.caption("⚠️ Short recording")
                else:
                    st.caption("✅ Good length")
            
            if duration < 0.5:
                st.warning(f"⚠️ Recording too short ({duration:.2f}s). Please record at least 3-5 seconds for better accuracy.")
            elif duration < 2.0:
                st.info(f"ℹ️ Recording is short ({duration:.2f}s). For best results, record 3-5 seconds.")
                
        except Exception as e:
            st.caption(f"🎵 Recording captured ({len(audio_data)} bytes)")
    
    st.divider()
    
    # Process recording - Check if we have NEW audio data for verification
    if audio_data is not None and audio_data != st.session_state.last_audio_data:
        st.session_state.last_audio_data = audio_data
        
        if len(audio_data) > 100:  # At least 100 bytes
            try:
                # Save WAV bytes to temp file
                temp_wav_path = "temp_recording.wav"
                with open(temp_wav_path, "wb") as f:
                    f.write(audio_data)
                
                st.divider()
                
                # Verification button
                st.subheader("🔐 Verify This Recording")
                if st.button("🔐 Verify Speaker", use_container_width=True, type="primary"):
                    with st.spinner("Verifying... Extracting embeddings and comparing..."):
                        try:
                            result = guardian.verify_speaker(temp_wav_path, return_details=True)
                            
                            # Save verification recording
                            saved_filepath = audio_storage.save_verification_recording(
                                audio_data,
                                sample_rate=16000
                            )
                            
                            # Display results
                            st.divider()
                            
                            # Main result display
                            result_col1, result_col2, result_col3 = st.columns(3)
                            
                            with result_col1:
                                if result["status"] == "Authenticated":
                                    st.markdown(
                                        f"""
                                        <div class="authenticated">
                                            <h3>✅ AUTHENTICATED</h3>
                                            <p>Access Granted</p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        f"""
                                        <div class="rejected">
                                            <h3>❌ REJECTED</h3>
                                            <p>Access Denied</p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                            
                            with result_col2:
                                st.metric("Confidence", f"{result['confidence']:.2%}")
                            
                            with result_col3:
                                st.metric("Similarity", f"{result['similarity']:.4f}")
                            
                            # Detailed breakdown
                            st.subheader("📊 Verification Details")
                            details_col1, details_col2, details_col3, details_col4 = st.columns(4)
                            
                            with details_col1:
                                st.metric("Status", result["status"])
                            with details_col2:
                                st.metric("Threshold", f"{result['threshold']:.2f}")
                            with details_col3:
                                st.metric("Decision", result["decision"])
                            with details_col4:
                                st.metric("Margin", f"{abs(result['similarity'] - result['threshold']):.4f}")
                            
                            # Visual comparison
                            st.subheader("📈 Score Visualization")
                            comparison_data = {
                                "Similarity": result["similarity"],
                                "Threshold": result["threshold"],
                                "Confidence": result["confidence"]
                            }
                            st.bar_chart(comparison_data)
                            
                            # Recording history
                            st.subheader("📚 Recent Verifications")
                            recent_recordings = audio_storage.get_verification_recordings(limit=5)
                            if recent_recordings:
                                for i, rec in enumerate(recent_recordings, 1):
                                    st.caption(f"{i}. {os.path.basename(rec)}")
                            
                            # Storage stats
                            stats = audio_storage.get_storage_stats()
                            st.divider()
                            st.caption(
                                f"💾 Storage: {stats['total_files']} recordings "
                                f"({stats['total_size_mb']} MB)"
                            )
                            
                            # Cleanup
                            if os.path.exists(temp_wav_path):
                                os.remove(temp_wav_path)
                            
                        except Exception as e:
                            st.error(f"❌ Verification failed: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            except Exception as e:
                st.error(f"❌ Error processing audio: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning(f"⚠️ Recording too small ({len(audio_data)} bytes). Please try recording again.")
    elif audio_data is None:
        st.info("🎤 Click the microphone button above to record your voice for verification")

st.divider()
st.caption(
    "🔐 Voice Guardian - Speaker Authentication System | "
    "SpeechBrain ECAPA-TDNN | Problem Statement 3B"
)

# ==================== TEST SECTION ====================
st.divider()
st.header("🧪 Test Recognition")
st.info("""
**Test Mode:** Upload a previously downloaded recording to verify the system recognizes it.
This is useful for testing if the same recording is accepted by the enrolled speaker model.
""")

test_col1, test_col2 = st.columns(2)

with test_col1:
    st.subheader("📤 Upload Test Recording")
    test_file = st.file_uploader(
        "Upload a .wav file to test",
        type=['wav'],
        key='test_upload',
        help="Upload a previously downloaded recording or any test audio"
    )
    
    if test_file:
        st.success(f"✅ Test file selected: {test_file.name}")
        
        # Show test file details
        test_audio_data = test_file.read()
        st.caption(f"File size: {len(test_audio_data)} bytes")
        st.audio(test_audio_data, format="audio/wav")

with test_col2:
    st.subheader("🎯 Test Recognition")
    if test_file:
        if st.button("🔬 Test Speaker Recognition", use_container_width=True, type="secondary"):
            with st.spinner("Testing... Processing audio..."):
                try:
                    # Save test file temporarily
                    test_temp_path = "test_temp_recording.wav"
                    with open(test_temp_path, "wb") as f:
                        f.write(test_audio_data)
                    
                    # Verify
                    test_result = guardian.verify_speaker(test_temp_path, return_details=True)
                    
                    # Display test results
                    st.divider()
                    st.subheader("✅ Test Results")
                    
                    test_result_col1, test_result_col2, test_result_col3 = st.columns(3)
                    
                    with test_result_col1:
                        if test_result["status"] == "Authenticated":
                            st.markdown(
                                """
                                <div class="authenticated">
                                    <h3>✅ RECOGNIZED</h3>
                                    <p>System accepted as enrolled speaker</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                """
                                <div class="rejected">
                                    <h3>❌ NOT RECOGNIZED</h3>
                                    <p>System rejected voice</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    
                    with test_result_col2:
                        st.metric("Confidence", f"{test_result['confidence']:.2%}")
                    
                    with test_result_col3:
                        st.metric("Similarity", f"{test_result['similarity']:.4f}")
                    
                    # Detailed test metrics
                    st.subheader("📊 Test Metrics")
                    test_metrics_col1, test_metrics_col2, test_metrics_col3, test_metrics_col4 = st.columns(4)
                    
                    with test_metrics_col1:
                        st.metric("Recognition", test_result["status"])
                    with test_metrics_col2:
                        st.metric("Threshold", f"{test_result['threshold']:.2f}")
                    with test_metrics_col3:
                        st.metric("Decision", test_result["decision"])
                    with test_metrics_col4:
                        st.metric("Margin", f"{abs(test_result['similarity'] - test_result['threshold']):.4f}")
                    
                    # Test visualization
                    st.subheader("📈 Recognition Score Visualization")
                    test_comparison_data = {
                        "Similarity": test_result["similarity"],
                        "Threshold": test_result["threshold"],
                        "Confidence": test_result["confidence"]
                    }
                    st.bar_chart(test_comparison_data)
                    
                    # Interpretation
                    st.subheader("🔍 Interpretation")
                    if test_result["status"] == "Authenticated":
                        st.success(
                            f"✅ **The system recognized this voice as the enrolled speaker!**\n\n"
                            f"- Similarity score: {test_result['similarity']:.4f}\n"
                            f"- This is {abs(test_result['similarity'] - test_result['threshold']):.4f} points "
                            f"above the threshold ({test_result['threshold']:.2f})\n"
                            f"- Confidence level: {test_result['confidence']:.2%}"
                        )
                    else:
                        st.error(
                            f"❌ **The system rejected this voice.**\n\n"
                            f"- Similarity score: {test_result['similarity']:.4f}\n"
                            f"- This is {abs(test_result['similarity'] - test_result['threshold']):.4f} points "
                            f"below the threshold ({test_result['threshold']:.2f})\n"
                            f"- The voice is {abs(test_result['similarity'] - test_result['threshold']):.4f} points "
                            f"away from recognition\n"
                            f"- Consider: re-enrolling with more samples or adjusting the threshold"
                        )
                    
                    # Cleanup
                    if os.path.exists(test_temp_path):
                        os.remove(test_temp_path)
                    
                except Exception as e:
                    st.error(f"❌ Test failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("📤 Upload a test file first to begin testing")