"""
dataset for contrastive training on mid mel-band features.

it samples positive/negative speaker pairs from per-speaker .npy folders and
returns two normalized 6-dim vectors plus a target (+1 same, -1 different).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class Explicit_Mid_Mel_PairDataset(Dataset):
    """
    samples pairs for contrastive/siamese loss.

    expects:
      - root_dir with one subfolder per speaker, each containing .npy vectors
      - global_mean/global_std: per-feature stats (shape 6,)
    returns:
      (x1, x2, label) with label ∈ {+1.0, -1.0}
    """
    def __init__(self, root_dir, data_fraction=1.0, global_mean=None, global_std=None):
        """index per-speaker files; keep speakers with ≥2 files and optional per-speaker subsample."""
        self.root_dir = root_dir
        self.global_mean = global_mean
        self.global_std = global_std
        self.speaker_files = {}

        # Recursively scan speaker folders
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
