import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1, 1),  # use the pairwise distance
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        emb1 = self.encoder(x1)
        emb2 = self.encoder(x2)
        dist = F.pairwise_distance(emb1, emb2, p=2).unsqueeze(1)  # Euclidean distance
        return self.classifier(dist)  # probability of match
