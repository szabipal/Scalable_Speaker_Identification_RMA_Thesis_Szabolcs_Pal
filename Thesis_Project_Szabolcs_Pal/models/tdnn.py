import torch
import torch.nn as nn
import torch.nn.functional as F

class TDNN(nn.Module):
    def __init__(self, input_dim, embedding_dim=64, hidden_layers=2, context=[5, 2]):
        super(TDNN, self).__init__()
        self.context = context

        layers = []
        in_dim = input_dim

        # TDNN layers with temporal context
        for i, c in enumerate(context):
            layers.append(nn.Conv1d(in_dim, 64, kernel_size=c, stride=1))
            layers.append(nn.ReLU())
            in_dim = 64

        self.tdnn = nn.Sequential(*layers)

        # Fully connected layers after temporal encoding
        fc_input_dim = 128  # because stats_pooling returns 2 * 64
        fc_layers = []
        for _ in range(hidden_layers):
            fc_layers.append(nn.Linear(fc_input_dim, 64))
            fc_layers.append(nn.ReLU())
            fc_input_dim = 64  # for next layer input
        self.hidden = nn.Sequential(*fc_layers)

        self.embedding = nn.Linear(64, embedding_dim)

    def forward(self, x):  # x: (B, T, D)
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = self.tdnn(x)
        x = self.stats_pooling(x)
        x = self.hidden(x)
        return F.normalize(self.embedding(x), p=2, dim=1)  # normalized embeddings

    def stats_pooling(self, x):  # x: (B, C, T)
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        return torch.cat([mean, std], dim=1)  # (B, 2*C)
