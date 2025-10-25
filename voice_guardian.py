"""
Voice Guardian: Speaker Authentication System
Problem Statement: 3B - Voice Guardian
Single-speaker binary classification (Target vs. Non-Target)

Uses SpeechBrain ECAPA-TDNN for speaker embeddings and cosine similarity for verification.
"""

import torch
import librosa
import torch.nn.functional as F
import numpy as np
import os

# Patch torchaudio to handle compatibility with speechbrain
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
if not hasattr(torchaudio, 'get_audio_backend'):
    torchaudio.get_audio_backend = lambda: "soundfile"

from speechbrain.inference import SpeakerRecognition


class VoiceGuardian:
    """
    A voice authentication system that uses speaker embeddings
    to verify whether a speaker is the enrolled target speaker.
    
    Process:
    1. Enrollment: Process 10-20 target speaker samples to create embeddings
    2. Aggregation: Average all embeddings to create a "master voiceprint"
    3. Verification: Compare new audio against the master voiceprint using cosine similarity
    """

    def __init__(self, threshold=0.85, model_name="speechbrain/spkrec-ecapa-voxceleb"):
        """
        Initialize the VoiceGuardian system.
        
        Args:
            threshold (float): Cosine similarity threshold for authentication (default: 0.85)
            model_name (str): Pre-trained model name from SpeechBrain
        """
        self.threshold = threshold
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading SpeechBrain model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load the pre-trained ECAPA-TDNN model for speaker recognition
        self.classifier = SpeakerRecognition.from_hparams(
            source=model_name,
            savedir="pretrained_models/ecapa-tdnn"
        )
        
        self.target_voiceprint = None
        self.enrollment_count = 0
        self.embeddings_history = []

    def _load_audio(self, file_path, target_sr=16000, min_duration=0.5):
        """
        Load audio file and convert to tensor.
        
        Args:
            file_path (str): Path to the audio file
            target_sr (int): Target sample rate (default: 16000 Hz)
            min_duration (float): Minimum duration in seconds (default: 0.5)
            
        Returns:
            torch.Tensor: Audio signal tensor (1D)
            
        Raises:
            RuntimeError: If audio is too short
        """
        try:
            # Load audio at target sample rate
            audio_signal, sr = librosa.load(file_path, sr=target_sr)
            
            # Check minimum duration
            duration = len(audio_signal) / target_sr
            if (duration < min_duration):
                raise RuntimeError(
                    f"Audio too short: {duration:.2f}s (minimum: {min_duration}s). "
                    f"Please record at least {min_duration} seconds."
                )
            
            # Ensure it's a numpy array
            if not isinstance(audio_signal, np.ndarray):
                audio_signal = np.array(audio_signal)
            
            # Convert to torch tensor and ensure it's 1D
            audio_tensor = torch.from_numpy(audio_signal).float()
            
            if audio_tensor.ndim > 1:
                # If stereo, convert to mono by averaging channels
                audio_tensor = audio_tensor.mean(dim=0)
            
            return audio_tensor
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load audio from {file_path}: {str(e)}")

    def _get_embedding(self, audio_tensor):
        """
        Extract speaker embedding from audio tensor.
        
        Args:
            audio_tensor (torch.Tensor): Audio signal tensor (1D)
            
        Returns:
            torch.Tensor: Speaker embedding vector (1D)
        """
        try:
            # Ensure tensor is on correct device
            audio_tensor = audio_tensor.to(self.device)
            
            # The model expects shape (batch, samples) = (1, num_samples)
            batch_input = audio_tensor.unsqueeze(0)
            
            # Get embedding from the model
            with torch.no_grad():
                embedding = self.classifier.encode_batch(batch_input)
            
            # embedding shape is (batch, 1, embedding_dim) = (1, 1, 192)
            # Remove batch and channel dimensions to get (embedding_dim,)
            embedding = embedding.squeeze()
            
            # Move to CPU for storage
            embedding = embedding.cpu()
            
            return embedding
        except Exception as e:
            raise RuntimeError(f"Failed to extract embedding: {str(e)}")

    def enroll_target_speaker(self, target_file_paths):
        """
        Enroll the target speaker using multiple utterances (10-20 samples).
        
        Creates a master voiceprint by averaging embeddings from all target samples.
        [cite: 148] Model trained only on target speaker's samples.
        
        Args:
            target_file_paths (list): List of paths to target speaker audio files
            
        Raises:
            ValueError: If no valid audio files provided
        """
        if not target_file_paths:
            raise ValueError("No target audio files provided")
        
        all_embeddings = []
        successful_enrollments = 0
        
        print(f"\nEnrolling target speaker with {len(target_file_paths)} audio files...")
        
        # Process each target audio file
        for i, file_path in enumerate(target_file_paths, 1):
            try:
                print(f"  Processing {i}/{len(target_file_paths)}: {os.path.basename(file_path)}")
                
                # Load audio
                audio = self._load_audio(file_path)
                
                # Get embedding
                embedding = self._get_embedding(audio)
                
                # Store embedding
                all_embeddings.append(embedding)
                self.embeddings_history.append(embedding)
                successful_enrollments += 1
                
            except Exception as e:
                print(f"  ⚠️  Skipped {os.path.basename(file_path)}: {str(e)}")
        
        if not all_embeddings:
            raise ValueError("No valid embeddings could be extracted from target files")
        
        # Stack all embeddings into a tensor
        stacked_embeddings = torch.stack(all_embeddings)
        
        # Calculate the average embedding (the "master voiceprint")
        self.target_voiceprint = torch.mean(stacked_embeddings, dim=0)
        self.enrollment_count = successful_enrollments
        
        print(f"\n✅ Enrollment complete!")
        print(f"   Successfully processed: {successful_enrollments}/{len(target_file_paths)} files")
        print(f"   Voiceprint shape: {self.target_voiceprint.shape}")
        print(f"   Threshold set to: {self.threshold}")

    def verify_speaker(self, new_audio_file, return_details=False):
        """
        Verify if a new audio sample belongs to the enrolled target speaker.
        
        [cite: 146] Input is a short voice clip.
        [cite: 147] Output "Authenticated" or "Rejected" with confidence score.
        
        Args:
            new_audio_file (str): Path to the audio file to verify
            return_details (bool): If True, return additional details
            
        Returns:
            dict: {
                "status": "Authenticated" or "Rejected",
                "confidence": float (0-1),
                "similarity": float (cosine similarity score),
                "threshold": float,
                "decision": "Above" or "Below" threshold
            }
            
        Raises:
            Exception: If no target speaker enrolled or file cannot be loaded
        """
        # Check if a target speaker has been enrolled
        if self.target_voiceprint is None:
            raise Exception(
                "No target speaker enrolled. "
                "Please enroll a speaker first using enroll_target_speaker()."
            )
        
        try:
            # Load the new audio file
            new_audio = self._load_audio(new_audio_file)
            
            # Get its embedding
            new_embedding = self._get_embedding(new_audio)
            
        except Exception as e:
            raise RuntimeError(f"Failed to process verification audio: {str(e)}")
        
        # Calculate cosine similarity between target voiceprint and new embedding
        # Cosine similarity ranges from -1 to 1, with 1 being identical
        similarity = F.cosine_similarity(
            self.target_voiceprint.unsqueeze(0),
            new_embedding.unsqueeze(0),
            dim=1
        ).item()
        
        # Normalize similarity to 0-1 range for confidence score
        confidence = (similarity + 1) / 2
        
        # Determine if authenticated based on threshold
        is_authenticated = similarity > self.threshold
        
        result = {
            "status": "Authenticated" if is_authenticated else "Rejected",
            "confidence": confidence,
            "similarity": similarity,
            "threshold": self.threshold,
            "decision": "Above" if is_authenticated else "Below"
        }
        
        if return_details:
            result["file"] = os.path.basename(new_audio_file)
        
        return result

    def get_stats(self):
        """Get statistics about the enrolled speaker."""
        return {
            "enrolled": self.target_voiceprint is not None,
            "enrollment_count": self.enrollment_count,
            "threshold": self.threshold,
            "device": self.device,
            "model": self.model_name
        }


if __name__ == "__main__":
    # Example usage
    target_files = [
        "path/to/target/sample1.wav",
        "path/to/target/sample2.wav"
    ]
    
    impostor_file = "path/to/impostor.wav"
    real_speaker_file = "path/to/target/sample_test.wav"
    
    try:
        # Initialize
        guardian = VoiceGuardian(threshold=0.85)
        
        # Enroll
        guardian.enroll_target_speaker(target_files)
        
        # Verify impostor
        result_impostor = guardian.verify_speaker(impostor_file, return_details=True)
        print(f"\nImpostor Test: {result_impostor}")
        
        # Verify real speaker
        result_real = guardian.verify_speaker(real_speaker_file, return_details=True)
        print(f"Real Speaker Test: {result_real}")
        
    except Exception as e:
        print(f"Error: {e}")
