import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys

from models.mlp import MLP
from models.tdnn import TDNN
from my_datasets.explicit_high_mel_dataset import Explicit_High_Mel_PairDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("explicit_embeddings", exist_ok=True)

    # Load global mean and std for feature normalization
    global_mean = np.load("global_high_mel_feature_mean_per_feature.npy")
    global_std = np.load("global_high_mel_feature_std_per_feature.npy")

    # Load dataset
    dataset = Explicit_High_Mel_PairDataset(
        root_dir="data/processed/mel_complement_features/high/",
        data_fraction=config["data_fraction"],
        global_mean=global_mean,
        global_std=global_std
    )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Get input dimension from one sample
    anchor, _, _ = next(iter(dataloader))
    print(anchor.shape)
    input_dim = anchor.shape[-1]

    # Create model
    model_class = TDNN if config["model_type"] == "TDNN" else MLP
    model = model_class(
        input_dim=input_dim,
        embedding_dim=config["embedding_dim"],
        hidden_layers=config["hidden_layers"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CosineEmbeddingLoss(margin=0.5)

    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for x1, x2, label in pbar:
            x1 = x1.to(device)
            x2 = x2.to(device)
            label = label.to(device)

            emb1 = model(x1)
            emb2 = model(x2)

            loss = criterion(emb1, emb2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

    # Save model and config
    name = f"{config['model_type']}_h{config['hidden_layers']}_df{int(config['data_fraction']*100)}"
    torch.save(model.state_dict(), f"explicit_models/high_mel_based/{name}.pt")
    with open(f"explicit_embeddings/{name}_config.yaml", "w") as f:
        yaml.dump(config, f)

    print(f"\n Saved: {name}.pt and config")
