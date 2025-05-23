import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Audio
from typing import List, Optional, Tuple
import functools
import librosa
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


class UrbanSoundDataset(Dataset):
    """
    A PyTorch Dataset for the UrbanSound8K dataset from Hugging Face.
    
    This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes:
    air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, 
    gun_shot, jackhammer, siren, and street_music.
    """
    
    def __init__(
        self,
        fold: int,  # Only fold is required now
        sample_rate: int = 22050,
        max_length: Optional[int] = None,
        target_length: Optional[int] = None,
        augment: bool = False,
        num_augmentations: int = 0,  # Number of augmented copies to generate (0 means no augmentation)
        cache_size: int = 1024,  # Reduced default cache size to save memory
        prefetch_factor: int = 2,  # Controls how many samples to prefetch per worker
    ):
        """
        Initialize the UrbanSound8K dataset for a specific fold.
        Args:
            fold: Only includes data from this fold (1-10)
            sample_rate: Target sample rate for audio
            max_length: Maximum number of samples to include
            target_length: If provided, pad/trim all audio to this length
            augment: Whether to apply data augmentation
            num_augmentations: Number of augmented copies to generate per sample (0 means no augmentation)
            cache_size: Maximum number of samples to cache in memory
            prefetch_factor: Controls how many samples to prefetch per worker
        """
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.augment = augment
        self.num_augmentations = num_augmentations
        self._cache = {}  # Initialize cache dictionary
        self._cache_size = cache_size
        self._cache_keys = []  # LRU tracking
        self.prefetch_factor = prefetch_factor
        self.fold = fold
        # Load only a small subset if max_length is set, for speed
        if max_length is not None:
            self.dataset = load_dataset("danavery/urbansound8K", split=f"train[:{max_length}]")
        else:
            self.dataset = load_dataset("danavery/urbansound8K", split="train")
        # Add audio with resampling
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
        # Only use the specified fold
        self.dataset = self.dataset.filter(lambda x: x["fold"] == fold)
        # Create label to index mapping
        self.class_names = sorted(list(set(self.dataset["classID"])))
        self.class_to_idx = {cls_id: i for i, cls_id in enumerate(self.class_names)}
        # Flag to determine if we're returning augmented variants
        self.use_multiple_augmentations = augment and num_augmentations > 0
        # Calculate actual dataset length based on augmentations
        self._effective_length = len(self.dataset)
        if self.use_multiple_augmentations:
            self._effective_length = len(self.dataset) * (1 + num_augmentations)
    
    def __len__(self):
        return self._effective_length
    
    def _get_original_index_and_aug_id(self, idx):
        """
        Map global index to original dataset index and augmentation ID.
        
        For example, if num_augmentations=2:
        - Global indices 0, 1, 2 map to original sample 0 with aug_ids 0, 1, 2
        - Global indices 3, 4, 5 map to original sample 1 with aug_ids 0, 1, 2
        - etc.
        
        Where aug_id 0 means "no augmentation" (original sample)
        """
        if not self.use_multiple_augmentations:
            return idx, 0
        
        # Each original sample has (1 + num_augmentations) entries
        items_per_original = 1 + self.num_augmentations
        original_idx = idx // items_per_original
        aug_id = idx % items_per_original  # 0 means original, 1+ means augmented
        
        return original_idx, aug_id

    def __getitem__(self, idx):
        # Map to original index and augmentation ID
        original_idx, aug_id = self._get_original_index_and_aug_id(idx)
        
        # Check if item is in cache
        cache_key = (original_idx, aug_id)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get the original item data
        item = self.dataset[original_idx]
        
        # Load audio and convert to tensor (always use float32)
        audio_data = item["audio"]["array"]
        waveform = torch.tensor(audio_data, dtype=torch.float32)
        
        # Handle stereo to mono conversion if needed
        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        
        # Apply target length constraint if specified
        if self.target_length is not None:
            if waveform.size(1) < self.target_length:
                # Pad
                padding = self.target_length - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif waveform.size(1) > self.target_length:
                # Trim
                if aug_id > 0:  # Take a random segment for augmented samples
                    start = torch.randint(0, waveform.size(1) - self.target_length + 1, (1,)).item()
                    waveform = waveform[:, start:start + self.target_length]
                else:  # Take the beginning for original samples (deterministic)
                    waveform = waveform[:, :self.target_length]
        
        # Apply augmentation for non-zero aug_id if using multiple augmentations
        if self.use_multiple_augmentations and aug_id > 0:
            # Set a unique seed for each (sample_idx, aug_id) combination for reproducibility
            # But still have different augmentations for each aug_id
            seed = hash((original_idx, aug_id)) % (2**32)
            torch.manual_seed(seed)
            waveform = self._augment_audio(waveform)
        elif self.augment and not self.use_multiple_augmentations:
            waveform = self._augment_audio(waveform)
        
        # Compute spectrogram using librosa (Mel spectrogram)
        waveform_np = waveform.squeeze(0).cpu().numpy()
        mel_spec = librosa.feature.melspectrogram(
            y=waveform_np,
            sr=self.sample_rate,
            n_fft=512,
            hop_length=256,
            n_mels=128,
            power=2.0
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        spectrogram = torch.tensor(mel_spec_db, dtype=torch.float32)

        # Convert spectrogram to image (RGB numpy array)
        fig, ax = plt.subplots(figsize=(2, 2), dpi=64)
        ax.axis('off')
        img = ax.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='magma')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGB')
        spectrogram_image = np.array(pil_img)
        buf.close()

        # Get label as descriptive string
        label = f"the spectrogram of a {item['class']}"

        # Create a minimal result dictionary to save memory
        result = {
            "spectrogram": spectrogram,
            "spectrogram_image": spectrogram_image,  # Add image here
            "label": label,
        }
        
        # Only add these fields if not in training/eval loop (for analysis/debugging)
        if not hasattr(torch.utils.data, '_DataLoader__initialized'):
            result.update({
                "sample_rate": self.sample_rate,
                "class_name": item["class"],
                "is_augmented": aug_id > 0,
                "original_index": original_idx
            })
        
        # Cache management with memory limits
        if len(self._cache) < self._cache_size:
            self._cache[cache_key] = result
            self._cache_keys.append(cache_key)
        elif self._cache_keys:  # If cache is full, remove oldest item
            old_key = self._cache_keys.pop(0)
            del self._cache[old_key]
            self._cache[cache_key] = result
            self._cache_keys.append(cache_key)
        
        return result
    
    def _augment_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to the audio waveform more efficiently."""
        # Apply a single augmentation type based on random choice
        # This is faster than checking each augmentation separately
        aug_type = torch.randint(0, 3, (1,)).item()
        
        if aug_type == 0:  # Random gain adjustment (volume)
            gain = 0.5 + torch.rand(1) * 1.0  # Random gain between 0.5 and 1.5
            waveform = waveform * gain
        elif aug_type == 1:  # Random time shift
            shift_amount = int(waveform.shape[1] * 0.1 * torch.rand(1))  # Up to 10% shift
            direction = 1 if torch.rand(1) > 0.5 else -1  # Left or right shift
            waveform = torch.roll(waveform, shifts=shift_amount * direction, dims=1)
        elif aug_type == 2:  # Random noise
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
            
        # Ensure values are in the valid range
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        return waveform

    def clear_cache(self):
        """Clear the sample cache to free memory"""
        self._cache = {}
        self._cache_keys = []


def get_datasets(
    sample_rate: int = 22050,
    max_length: Optional[int] = None,
    train_folds: list = [1,2,3,4,5,6,7,8],
    test_folds: list = [9,10],
    target_length: Optional[int] = None,
    num_augmentations: int = 0,
):
    """
    Get train and test datasets for the UrbanSound8K dataset using folds only.
    Args:
        sample_rate: Target sample rate for audio
        max_length: Maximum number of samples to include in each dataset
        train_folds: List of folds for training
        test_folds: List of folds for testing
        target_length: If provided, pad/trim all audio to this length
        num_augmentations: Number of augmented copies to generate per sample (0 = no augmentation)
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_datasets = [UrbanSoundDataset(
        fold=fold,
        sample_rate=sample_rate,
        max_length=max_length,
        target_length=target_length,
        augment=num_augmentations > 0,
        num_augmentations=num_augmentations
    ) for fold in train_folds]
    test_datasets = [UrbanSoundDataset(
        fold=fold,
        sample_rate=sample_rate,
        max_length=max_length,
        target_length=target_length,
        augment=False,
        num_augmentations=0
    ) for fold in test_folds]
    # Optionally, concatenate datasets if needed
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    test_dataset = ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]
    return train_dataset, test_dataset