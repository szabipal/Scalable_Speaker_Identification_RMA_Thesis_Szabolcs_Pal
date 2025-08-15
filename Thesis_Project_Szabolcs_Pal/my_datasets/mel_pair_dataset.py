"""
constrastive dataset for cnn training on mel features.

it builds pairs of .npy mel arrays from per-speaker folders and labels them:
0 for different-speaker pairs, 1 for same-speaker pairs.
"""

from pathlib import Path
import numpy as np
import random
from torch.utils.data import Dataset


class CNNMelDataset(Dataset):
    """
    samples mel pairs for cnn models.

    expects:
      - data_path: directory with one subfolder per speaker, each containing .npy files
    behavior:
      - caps per-speaker files at 20 for pairing
      - optionally subsamples with `data_fraction` (0 < f ≤ 1)
      - generates:
          * all cross-speaker pairs (label=0)
          * all within-speaker distinct pairs (label=1)
    returns:
      (mel_a, mel_b, label) where mel_* are float32 numpy arrays, label ∈ {0, 1}
    """
    def __init__(self, data_path, data_fraction=1.0):
        """index speaker dirs and precompute pair list with optional per-speaker subsampling."""
        self.data_path = Path(data_path)
        self.speaker_dirs = sorted([x for x in self.data_path.iterdir() if x.is_dir()])
        self.samples = []

        for i in range(len(self.speaker_dirs)):
            for j in range(len(self.speaker_dirs)):
                if i == j:
                    continue

                MAX_FILES = 20
                files_a = list(self.speaker_dirs[i].glob("*.npy"))[:MAX_FILES]
                files_b = list(self.speaker_dirs[j].glob("*.npy"))[:MAX_FILES]

                if data_fraction < 1.0:
                    files_a = random.sample(files_a, max(1, int(len(files_a) * data_fraction)))
                    files_b = random.sample(files_b, max(1, int(len(files_b) * data_fraction)))

                # different-speaker pairs (label 0)
                for f1 in files_a:
                    for f2 in files_b:
                        self.samples.append((f1, f2, 0))  # different speaker

                # same-speaker pairs (distinct files, label 1)
                for f1 in files_a:
                    for f2 in files_a:
                        if f1 != f2:
                            self.samples.append((f1, f2, 1))  # same speaker

    def __len__(self):
        """number of precomputed pairs."""
        return len(self.samples)

    def __getitem__(self, idx):
        """load npy arrays for a pair and return (a, b, label)."""
        f1, f2, label = self.samples[idx]
        a = np.load(f1).astype(np.float32)
        b = np.load(f2).astype(np.float32)
        return a, b, label
