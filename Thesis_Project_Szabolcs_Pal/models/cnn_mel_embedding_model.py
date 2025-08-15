
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNMelEmbeddingModel(nn.Module):
    def __init__(self, input_channels=1, input_height=40, input_width=200, embedding_dim=128, num_classes=None):
        super(CNNMelEmbeddingModel, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output shape will be [batch, channels, 1, 1]

        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes) if num_classes else None

        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  

        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))

        x = self.dropout(x)
        x = self.adaptive_pool(x)  # [B, C, 1, 1]
        x = x.view(x.size(0), -1)  # [B, C]

        x = self.embedding(x)  # [B, embedding_dim]

        if self.classifier:
            return self.classifier(x), x
        return x
