"""
Voice Guardian: Enhanced Speaker Authentication System with Cross-Encoder
Problem Statement: 3B - Voice Guardian
Single-speaker binary classification (Target vs. Non-Target)

Uses:
1. SpeechBrain ECAPA-TDNN for speaker embeddings
2. Cross-Encoder neural network for improved similarity scoring
3. Cosine similarity as fallback/baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import os

# Patch torchaudio to handle compatibility with speechbrain
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
if not hasattr(torchaudio, 'get_audio_backend'):
    torchaudio.get_audio_backend = lambda: "soundfile"

from speechbrain.pretrained import SpeakerRecognition


class CrossEncoderNetwork(nn.Module):
    """
    Cross-Encoder neural network that jointly processes two embeddings
    to produce a more accurate similarity score.
    """
    def __init__(self, embedding_dim=192, hidden_dim=256):
        super(CrossEncoderNetwork, self).__init__()
        
        # Concatenate two embeddings: 192 + 192 = 384
        input_dim = embedding_dim * 2
        
        # Multi-layer network for learning complex similarity patterns
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, embedding1, embedding2):
        """
        Forward pass through cross-encoder
        
        Args:
            embedding1: First embedding tensor (batch, embedding_dim)
            embedding2: Second embedding tensor (batch, embedding_dim)
            
        Returns:
            Similarity score between 0 and 1
        """
        # Concatenate embeddings
        combined = torch.cat([embedding1, embedding2], dim=-1)
        
        # Pass through network
        score = self.network(combined)
        
        return score


class VoiceGuardianEnhanced:
    """
    Enhanced voice authentication system with Cross-Encoder for improved accuracy.
    
    Process:
    1. Enrollment: Process 10-20 target speaker samples to create embeddings
    2. Aggregation: Average all embeddings + store individual embeddings
    3. Cross-Encoder Training: Train on positive/negative pairs (optional)
    4. Verification: Use both cosine similarity and cross-encoder for scoring
    """

    def __init__(self, threshold=0.80, model_name="speechbrain/spkrec-ecapa-voxceleb", 
                 use_cross_encoder=True):
        """
        Initialize the Enhanced VoiceGuardian system.
        
        Args:
            threshold (float): Similarity threshold for authentication (default: 0.80)
            model_name (str): Pre-trained model name from SpeechBrain
            use_cross_encoder (bool): Whether to use cross-encoder for scoring
        """
        self.threshold = threshold
        self.model_name = model_name
        self.use_cross_encoder = use_cross_encoder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading SpeechBrain model: {model_name}")
        print(f"Using device: {self.device}")
        print(f"Cross-Encoder: {'Enabled' if use_cross_encoder else 'Disabled'}")
        
        # Load the pre-trained ECAPA-TDNN model for speaker recognition
        self.classifier = SpeakerRecognition.from_hparams(
            source=model_name,
            savedir="pretrained_models/ecapa-tdnn"
        )
        
        self.target_voiceprint = None
        self.enrollment_count = 0
        self.embeddings_history = []
        self.all_enrollment_embeddings = []  # Store all for current enrollment
        
        # Initialize cross-encoder network
        if self.use_cross_encoder:
            self.cross_encoder = CrossEncoderNetwork(embedding_dim=192, hidden_dim=256)
            self.cross_encoder.to(self.device)
            self._initialize_cross_encoder()
        else:
            self.cross_encoder = None

    def _initialize_cross_encoder(self):
        """
        Initialize cross-encoder with pseudo-training or load pre-trained weights.
        This creates a better scoring function than simple cosine similarity.
        """
        # Initialize with Xavier initialization for better convergence
        for module in self.cross_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        print("Cross-Encoder initialized with Xavier initialization")

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
        except FileNotFoundError:
            raise RuntimeError(f"Audio file not found: {file_path}")
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
        Also stores individual embeddings for cross-encoder comparison.
        
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
                self.all_enrollment_embeddings.append(embedding)
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
        
        # Fine-tune cross-encoder if enabled and we have enough samples
        if self.use_cross_encoder and successful_enrollments >= 5:
            print("\n🔄 Optimizing cross-encoder for enrolled speaker...")
            self._optimize_cross_encoder()
        
        print(f"\n✅ Enrollment complete!")
        print(f"   Successfully processed: {successful_enrollments}/{len(target_file_paths)} files")
        print(f"   Voiceprint shape: {self.target_voiceprint.shape}")
        print(f"   Threshold set to: {self.threshold}")

    def _optimize_cross_encoder(self):
        """
        Optimize cross-encoder using the enrolled embeddings.
        Creates synthetic negative samples and trains the network.
        """
        if not self.use_cross_encoder or len(self.all_enrollment_embeddings) < 5:
            return
        
        self.cross_encoder.train()
        optimizer = torch.optim.Adam(self.cross_encoder.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        num_epochs = 50
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            # Create positive pairs (same speaker)
            for i in range(len(self.all_enrollment_embeddings)):
                for j in range(i + 1, len(self.all_enrollment_embeddings)):
                    emb1 = self.all_enrollment_embeddings[i].to(self.device).unsqueeze(0)
                    emb2 = self.all_enrollment_embeddings[j].to(self.device).unsqueeze(0)
                    
                    # Positive pair should score high
                    score = self.cross_encoder(emb1, emb2)
                    loss = criterion(score, torch.ones_like(score))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
            
            # Create negative pairs (add noise to simulate different speaker)
            for i in range(min(10, len(self.all_enrollment_embeddings))):
                emb1 = self.all_enrollment_embeddings[i].to(self.device).unsqueeze(0)
                
                # Create synthetic negative by adding significant noise
                noise = torch.randn_like(emb1) * 0.5
                emb2_negative = emb1 + noise
                emb2_negative = F.normalize(emb2_negative, p=2, dim=-1)
                
                # Negative pair should score low
                score = self.cross_encoder(emb1, emb2_negative)
                loss = criterion(score, torch.zeros_like(score))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                print(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        self.cross_encoder.eval()
        print("  ✅ Cross-encoder optimization complete!")

    def _calculate_cosine_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        return F.cosine_similarity(
            embedding1.unsqueeze(0),
            embedding2.unsqueeze(0),
            dim=1
        ).item()

    def _calculate_cross_encoder_score(self, embedding1, embedding2):
        """Calculate cross-encoder similarity score"""
        with torch.no_grad():
            self.cross_encoder.eval()
            emb1 = embedding1.to(self.device).unsqueeze(0)
            emb2 = embedding2.to(self.device).unsqueeze(0)
            score = self.cross_encoder(emb1, emb2)
            return score.item()

    def verify_speaker(self, new_audio_file, return_details=False):
        """
        Verify if a new audio sample belongs to the enrolled target speaker.
        Uses enhanced scoring with cross-encoder for better accuracy.
        
        Args:
            new_audio_file (str): Path to the audio file to verify
            return_details (bool): If True, return additional details
            
        Returns:
            dict: {
                "status": "Authenticated" or "Rejected",
                "confidence": float (0-1),
                "similarity": float (combined similarity score),
                "cosine_similarity": float (cosine similarity),
                "cross_encoder_score": float (cross-encoder score, if enabled),
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
        
        # Calculate cosine similarities to all enrolled embeddings
        cosine_scores = []
        for enroll_emb in self.all_enrollment_embeddings:
            cos_sim = self._calculate_cosine_similarity(enroll_emb, new_embedding)
            cosine_scores.append(cos_sim)
        max_cosine = max(cosine_scores) if cosine_scores else 0.0
        
        # Calculate cross-encoder score if enabled
        if self.use_cross_encoder:
            cross_encoder_score = self._calculate_cross_encoder_score(
                self.target_voiceprint,
                new_embedding
            )
            
            # Also compare with multiple enrollment embeddings for robustness
            ce_scores = []
            for enroll_emb in self.all_enrollment_embeddings:  # Use all enrolled
                score = self._calculate_cross_encoder_score(enroll_emb, new_embedding)
                ce_scores.append(score)
            
            # Use max score from all comparisons
            max_ce_score = max(ce_scores) if ce_scores else cross_encoder_score
            
            # Weighted combination: 40% max cosine + 60% max cross-encoder
            # Cross-encoder gets more weight as it's trained on the data
            final_similarity = 0.4 * max_cosine + 0.6 * max_ce_score
            
        else:
            cross_encoder_score = None
            max_ce_score = None
            final_similarity = max_cosine
        
        # Normalize to 0-1 range for confidence
        confidence = (final_similarity + 1) / 2 if final_similarity < 1.0 else final_similarity
        
        # Determine if authenticated based on threshold
        is_authenticated = final_similarity > self.threshold
        
        result = {
            "status": "Authenticated" if is_authenticated else "Rejected",
            "confidence": confidence,
            "similarity": final_similarity,
            "cosine_similarity": max_cosine,
            "threshold": self.threshold,
            "decision": "Above" if is_authenticated else "Below"
        }
        
        if self.use_cross_encoder:
            result["cross_encoder_score"] = max_ce_score
            result["scoring_method"] = "Hybrid (Cosine + Cross-Encoder)"
        else:
            result["scoring_method"] = "Cosine Similarity"
        
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
            "model": self.model_name,
            "cross_encoder_enabled": self.use_cross_encoder
        }


# Backward compatibility: Keep original class name as alias
class VoiceGuardian(VoiceGuardianEnhanced):
    """Alias for backward compatibility"""
    pass


if __name__ == "__main__":
    # Example usage
    target_files = [
        "path/to/target/sample1.wav",
        "path/to/target/sample2.wav"
    ]
    
    impostor_file = "path/to/impostor.wav"
    real_speaker_file = "path/to/target/sample_test.wav"
    
    try:
        # Initialize with cross-encoder
        guardian = VoiceGuardianEnhanced(threshold=0.80, use_cross_encoder=True)
        
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