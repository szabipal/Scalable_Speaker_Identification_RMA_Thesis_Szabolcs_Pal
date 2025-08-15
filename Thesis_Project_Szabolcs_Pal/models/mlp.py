import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, embedding_dim=64, hidden_layers=2):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, 128))  # widened from 64 to 128
            layers.append(nn.ReLU())
            in_dim = 128
        layers.append(nn.Linear(in_dim, embedding_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, D)
        if x.dim() == 3:
            x = x.mean(dim=1)
        elif x.dim() == 4:
            x = torch.mean(x, dim=[2, 3]) 
        x = self.network(x)
        return F.normalize(x, p=2, dim=1)  # normalized embeddings
