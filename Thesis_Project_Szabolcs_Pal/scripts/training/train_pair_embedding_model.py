
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from models.mlp import MLP
from models.tdnn import TDNN
from datasets.general_pair_dataset import GeneralPairDataset
# from datasets.general_triplet_dataset import GeneralTripletDataset

from datasets.formant_pair_dataset import FormantPairDataset
from datasets.voice_quality_pair_dataset import VoiceQualityPairDataset
from datasets.cnn_mel_dataset import CNNSpectrogramDataset
from models.tdnn import TDNN
from models.mlp import MLP
from models.cnn_mel_embedding_model import CNNMelEmbeddingModel

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f"{config['feature_type']}_pair_embedding", exist_ok=True)
    dataset_class = globals()[config["dataset"]]

    if config["dataset"] == "GeneralTripletDataset":
        dataset = dataset_class(
            root_dir=config["data_path"],
            feature_type=config["feature_type"],
            segment_len=config["segment_len"],
            moving_window=config["moving_window"],
            data_fraction=config["data_fraction"]
        )
    else:
        dataset = dataset_class(config["data_path"])

    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    sample_x1, _, _ = next(iter(dataloader))

    if config["model_type"] == "TDNN":
        model_class = TDNN
    elif config["model_type"] == "MLP":
        model_class = MLP
    elif config["model_type"] == "CNNMelEmbeddingModel":
        model_class = CNNMelEmbeddingModel
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    if config["model_type"] == "CNNMelEmbeddingModel":
        input_height = sample_x1.shape[-2]
        input_width = sample_x1.shape[-1]
        model = model_class(
            input_channels=1,
            input_height=input_height,
            input_width=input_width,
            embedding_dim=config["embedding_dim"]
        ).to(device)
    else:
        input_dim = sample_x1.shape[1] if len(sample_x1.shape) == 2 else sample_x1.shape[-1]
        model = model_class(
            input_dim=input_dim,
            embedding_dim=config["embedding_dim"],
            hidden_layers=config["hidden_layers"]
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CosineEmbeddingLoss(margin=0.5)

    
    loss_values = []

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for x1, x2, label in pbar:
            x1 = x1.to(device)
            x2 = x2.to(device)
            label = label.to(device)

            emb1 = nn.functional.normalize(model(x1), p=2, dim=1)
            emb2 = nn.functional.normalize(model(x2), p=2, dim=1)

            loss = criterion(emb1, emb2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        loss_values.append(total_loss / len(dataloader))

    save_path = os.path.join(f"{config['feature_type']}_pair_embedding",
                             f"{config['model_type']}_embed_{config['embedding_dim']}_layers{config['hidden_layers']}.pt")
    # Model saving is now handled externally.
    # torch.save(model.state_dict(), save_path)
    return model, loss_values
