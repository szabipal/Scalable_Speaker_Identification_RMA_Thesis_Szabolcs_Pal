import torch
from torch.utils.data import Dataset
import random
import ast
import numpy as np
import pandas as pd

class SpeakerPairDataset(Dataset):
    def __init__(self, csv_path, num_pairs=10000):
        self.df = pd.read_csv(csv_path)
        self.df["embedding"] = self.df["embedding"].apply(ast.literal_eval).apply(np.array)

        self.embeddings_by_speaker = {}
        for _, row in self.df.iterrows():
            self.embeddings_by_speaker.setdefault(row["speaker_id"], []).append(row["embedding"])

        self.pairs = self._generate_pairs(num_pairs)

    def _generate_pairs(self, num_pairs):
        pairs = []
        speaker_ids = list(self.embeddings_by_speaker.keys())
        for _ in range(num_pairs):
            if random.random() < 0.5:
                # Positive pair
                speaker = random.choice(speaker_ids)
                emb_list = self.embeddings_by_speaker[speaker]
                if len(emb_list) < 2:
                    continue
                a, b = random.sample(emb_list, 2)
                label = 1
            else:
                # Negative pair
                spk1, spk2 = random.sample(speaker_ids, 2)
                a = random.choice(self.embeddings_by_speaker[spk1])
                b = random.choice(self.embeddings_by_speaker[spk2])
                label = 0
            pairs.append((a, b, label))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b, label = self.pairs[idx]
        return (
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(b, dtype=torch.float32),
            torch.tensor([label], dtype=torch.float32)
        )
