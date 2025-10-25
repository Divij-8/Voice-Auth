"""
Audio Storage Utilities for Voice Guardian
Handles saving and managing .wav audio recordings
"""

import os
import soundfile as sf
from datetime import datetime
import uuid


class AudioStorage:
    """
    A utility class for storing and managing audio recordings
    """

    def __init__(self, base_dir="recordings"):
        """
        Initialize the audio storage system

        Args:
            base_dir (str): Base directory for storing recordings
        """
        self.base_dir = base_dir
        self.verification_dir = os.path.join(base_dir, "verification")
        self.enrollment_dir = os.path.join(base_dir, "enrollment")

        # Create directories if they don't exist
        os.makedirs(self.verification_dir, exist_ok=True)
        os.makedirs(self.enrollment_dir, exist_ok=True)

    def save_verification_recording(self, audio_data, sample_rate=16000, prefix="verification"):
        """
        Save a verification recording with timestamp
        
        Args:
            audio_data: Audio data (can be WAV bytes, 1D or 2D numpy array)
            sample_rate (int): Sample rate of the audio
            prefix (str): Prefix for the filename
            
        Returns:
            str: Path to the saved file
        """
        import numpy as np
        
        # Handle WAV bytes - read them as audio
        if isinstance(audio_data, bytes):
            import io
            import soundfile as sf
            # Read from bytes
            audio_array, sr = sf.read(io.BytesIO(audio_data))
            audio_data = audio_array
            sample_rate = sr
        
        # Ensure audio_data is a numpy array
        audio_data = np.array(audio_data)
        
        # Handle different audio data formats
        if audio_data.ndim == 0:
            raise ValueError("Audio data is empty (0 dimensions)")
        elif audio_data.ndim == 1:
            # 1D array (mono) - soundfile can handle this
            pass
        elif audio_data.ndim == 2:
            # 2D array - check if it's (samples, channels) or (channels, samples)
            if audio_data.shape[0] > audio_data.shape[1]:
                # Likely (samples, channels) - this is correct for soundfile
                pass
            else:
                # Likely (channels, samples) - transpose to (samples, channels)
                audio_data = audio_data.T
        else:
            raise ValueError(f"Unsupported audio data dimensions: {audio_data.ndim}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{prefix}_{timestamp}_{unique_id}.wav"
        filepath = os.path.join(self.verification_dir, filename)
        
        sf.write(filepath, audio_data, sample_rate)
        return filepath

    def save_enrollment_recording(self, audio_data, sample_rate=16000, speaker_id="unknown", sample_num=1):
        """
        Save an enrollment recording for a specific speaker
        
        Args:
            audio_data: Audio data array (can be 1D or 2D)
            sample_rate (int): Sample rate of the audio
            speaker_id (str): Identifier for the speaker
            sample_num (int): Sample number for this speaker
            
        Returns:
            str: Path to the saved file
        """
        import numpy as np
        
        # Ensure audio_data is a numpy array
        audio_data = np.array(audio_data)
        
        # Handle different audio data formats
        if audio_data.ndim == 1:
            # 1D array (mono) - soundfile can handle this
            pass
        elif audio_data.ndim == 2:
            # 2D array - check if it's (samples, channels) or (channels, samples)
            if audio_data.shape[0] > audio_data.shape[1]:
                # Likely (samples, channels) - this is correct for soundfile
                pass
            else:
                # Likely (channels, samples) - transpose to (samples, channels)
                audio_data = audio_data.T
        else:
            raise ValueError(f"Unsupported audio data dimensions: {audio_data.ndim}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enrollment_{speaker_id}_sample{sample_num}_{timestamp}.wav"
        filepath = os.path.join(self.enrollment_dir, filename)
        
        sf.write(filepath, audio_data, sample_rate)
        return filepath

    def get_verification_recordings(self, limit=None):
        """
        Get list of verification recording files

        Args:
            limit (int): Maximum number of files to return (most recent first)

        Returns:
            list: List of file paths
        """
        files = [os.path.join(self.verification_dir, f)
                for f in os.listdir(self.verification_dir)
                if f.endswith('.wav')]
        files.sort(key=os.path.getmtime, reverse=True)

        if limit:
            files = files[:limit]

        return files

    def get_enrollment_recordings(self, speaker_id=None):
        """
        Get list of enrollment recording files

        Args:
            speaker_id (str): Filter by speaker ID, if None returns all

        Returns:
            list: List of file paths
        """
        files = [os.path.join(self.enrollment_dir, f)
                for f in os.listdir(self.enrollment_dir)
                if f.endswith('.wav')]

        if speaker_id:
            files = [f for f in files if f"_speaker_{speaker_id}_" in os.path.basename(f)]

        files.sort(key=os.path.getmtime, reverse=True)
        return files

    def cleanup_old_recordings(self, days_to_keep=30, recording_type="verification"):
        """
        Clean up old recordings

        Args:
            days_to_keep (int): Number of days of recordings to keep
            recording_type (str): Type of recordings to clean ("verification" or "enrollment")
        """
        import time
        from datetime import datetime, timedelta

        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)

        if recording_type == "verification":
            target_dir = self.verification_dir
        elif recording_type == "enrollment":
            target_dir = self.enrollment_dir
        else:
            raise ValueError("recording_type must be 'verification' or 'enrollment'")

        deleted_count = 0
        for filename in os.listdir(target_dir):
            if filename.endswith('.wav'):
                filepath = os.path.join(target_dir, filename)
                if os.path.getmtime(filepath) < cutoff_time:
                    os.remove(filepath)
                    deleted_count += 1

        return deleted_count

    def get_storage_stats(self):
        """
        Get statistics about stored recordings

        Returns:
            dict: Statistics about recordings
        """
        verification_files = self.get_verification_recordings()
        enrollment_files = self.get_enrollment_recordings()

        total_size = 0
        for files in [verification_files, enrollment_files]:
            for filepath in files:
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)

        return {
            "verification_count": len(verification_files),
            "enrollment_count": len(enrollment_files),
            "total_files": len(verification_files) + len(enrollment_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }


# Global instance for easy access
audio_storage = AudioStorage()


if __name__ == "__main__":
    # Example usage
    storage = AudioStorage()

    # Print current stats
    stats = storage.get_storage_stats()
    print("Audio Storage Statistics:")
    print(f"Verification recordings: {stats['verification_count']}")
    print(f"Enrollment recordings: {stats['enrollment_count']}")
    print(f"Total files: {stats['total_files']}")
    print(f"Total size: {stats['total_size_mb']} MB")

    # Clean up old recordings (older than 7 days for demo)
    deleted = storage.cleanup_old_recordings(days_to_keep=7)
    print(f"Cleaned up {deleted} old verification recordings")
