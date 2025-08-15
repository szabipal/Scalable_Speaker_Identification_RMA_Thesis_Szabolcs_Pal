
from pathlib import Path
import numpy as np
import random
from torch.utils.data import Dataset

class MFCCDataset(Dataset):
    def __init__(self, data_dir, mode='train', max_files_per_speaker=None, data_fraction=1.0):
        '''
        Args:
            data_dir (str or Path): Path to folder containing MFCC features per speaker.
            mode (str): 'train', 'val', or 'test'
            max_files_per_speaker (int): Optional limit for files per speaker.
            data_fraction (float): Proportion of data to use (0 < fraction <= 1.0)
        '''
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.samples = []

        speaker_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        random.seed(42)
        random.shuffle(speaker_dirs)

        # Split speakers
        total = len(speaker_dirs)
        n_train = int(0.8 * total)
        n_val = int(0.1 * total)

        if mode == 'train':
            selected_speakers = speaker_dirs[:n_train]
        elif mode == 'val':
            selected_speakers = speaker_dirs[n_train:n_train + n_val]
        else:
            selected_speakers = speaker_dirs[n_train + n_val:]

        for speaker_dir in selected_speakers:
            files = sorted(speaker_dir.glob("*.npy"))
            if max_files_per_speaker:
                files = files[:max_files_per_speaker]
            for f in files:
                self.samples.append((f, speaker_dir.name))

        if data_fraction < 1.0:
            keep_n = int(len(self.samples) * data_fraction)
            self.samples = random.sample(self.samples, keep_n)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_path, anchor_speaker = self.samples[idx]
        anchor = np.load(anchor_path)

        # Pick a positive or negative sample
        is_positive = random.random() > 0.5
        other = None

        if is_positive:
            # Find another sample from the same speaker
            same_class_samples = [s for s in self.samples if s[1] == anchor_speaker and s[0] != anchor_path]
            if not same_class_samples:
                same_class_samples = [s for s in self.samples if s[1] == anchor_speaker]
            other_path, _ = random.choice(same_class_samples)
            label = 1
        else:
            # Find a sample from a different speaker
            different_class_samples = [s for s in self.samples if s[1] != anchor_speaker]
            other_path, _ = random.choice(different_class_samples)
            label = -1

        other = np.load(other_path)
        return anchor.astype(np.float32), other.astype(np.float32), label
