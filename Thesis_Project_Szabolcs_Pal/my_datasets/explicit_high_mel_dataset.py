"""
dataset for contrastive training on high mel-band features.

it samples positive/negative speaker pairs from per-speaker .npy folders and
returns two normalized 11-dim vectors plus a target (+1 same, -1 different).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class Explicit_High_Mel_PairDataset(Dataset):
    """
    samples pairs for contrastive/siamese loss.

    expects:
      - root_dir with one subfolder per speaker, each containing .npy vectors
      - global_mean/global_std: per-feature stats (shape 11,)
    returns:
      (x1, x2, label) with label ∈ {+1.0, -1.0}
    """
    def __init__(self, root_dir, data_fraction=1.0, global_mean=None, global_std=None):
        """index per-speaker files; keep speakers with ≥2 files and optional per-speaker subsample."""
        self.root_dir = root_dir
        self.global_mean = global_mean
        self.global_std = global_std
        self.speaker_files = {}

        # recursively scan speaker folders and collect .npy files
        for speaker_folder in os.listdir(root_dir):
            speaker_path = os.path.join(root_dir, speaker_folder)
            if not os.path.isdir(speaker_path):
                continue

            files = [
                os.path.join(speaker_path, f)
                for f in os.listdir(speaker_path)
                if f.endswith(".npy")
            ]

            if len(files) < 2:
                continue  # skip speaker if not enough files

            k = min(len(files), max(2, int(len(files) * data_fraction)))
            self.speaker_files[speaker_folder] = random.sample(files, k)

        self.speakers = list(self.speaker_files.keys())
        print(f"✅ Loaded {len(self.speakers)} speakers with at least 2 files each.")

    def __len__(self):
        """large fixed length to enable on-the-fly random pairing."""
        return 100000  # Can be adjusted or parameterized

    def __getitem__(self, idx):
        """draw one pair (50% same / 50% different), load + normalize, return tensors and label."""
        # 50% chance to pick positive or negative pair
        if random.random() < 0.5:
            # Positive pair
            speaker = random.choice(self.speakers)
            x1_path, x2_path = random.sample(self.speaker_files[speaker], 2)
            label = 1.0
        else:
            speaker1, speaker2 = random.sample(self.speakers, 2)
            x1_path = random.choice(self.speaker_files[speaker1])
            x2_path = random.choice(self.speaker_files[speaker2])
            label = -1.0

        x1 = self.transform_and_normalize(np.load(x1_path))
        x2 = self.transform_and_normalize(np.load(x2_path))

        return (
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )

    def transform_and_normalize(self, vec):
        """
        clean and normalize one vector:
          - assert 11 dims; replace nan/inf
          - light transforms (log on 1:4, scale 4:7)
          - normalize with global_mean/std; zero any residual non-finite
        """
        vec = vec.astype(np.float32).reshape(-1)
        assert vec.shape[0] == 11, f"Expected 11 features, got {vec.shape}"

        # Clean raw input: replace NaNs and Infs
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        # Feature-specific preprocessing
        vec[1:4] = np.log(np.clip(vec[1:4], a_min=1e-3, a_max=None))  # log-scale large features
        vec[4:7] = vec[4:7] * 1000                                    # scale up tiny values

        # Clean mean and std
        safe_mean = np.nan_to_num(self.global_mean, nan=0.0, posinf=0.0, neginf=0.0)
        safe_std = np.where(
            (np.isnan(self.global_std)) | (np.isinf(self.global_std)) | (self.global_std < 1e-6),
            1.0,
            self.global_std
        )

        # Normalize
        normalized_vec = (vec - safe_mean) / (safe_std + 1e-8)

        # Final cleanup: replace anything unexpected after norm
        normalized_vec[~np.isfinite(normalized_vec)] = 0.0

        return normalized_vec


