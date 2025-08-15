"""
trainer for speaker embedding models (spectral + phonetic) using siamese/contrastive losses.

it picks a dataset based on config (mfcc/mel vs explicit low/mid/high), builds the right model
(mlp/tdnn/cnn), trains with cosine embedding loss, and saves weights. includes quick test
routines for both spectral and phonetic representations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] 
import numpy as np

# === Dataset imports ===
# Spectral
from my_datasets.mfcc_pair_dataset import MFCCDataset
from my_datasets.mel_pair_dataset import CNNMelDataset

# Phonetic
from my_datasets.explicit_mfcc_dataset import ExplicitPairDataset
from my_datasets.explicit_low_mel_dataset import Explicit_Low_Mel_PairDataset
from my_datasets.explicit_mid_mel_dataset import Explicit_Mid_Mel_PairDataset
from my_datasets.explicit_high_mel_dataset import Explicit_High_Mel_PairDataset

# === Model imports ===
from models.mlp import MLP
from models.tdnn import TDNN
from models.cnn_mel_embedding_model import CNNMelEmbeddingModel

# === Util import ===
from scripts.preprocessing.compute_global_std_mean import compute_global_mean_std 


class SpeakerEmbeddingModelTrainer:
    """
    orchestrates dataset selection, model init, training loop, and saving.

    config keys (string/int/float):
      - representation_type: 'spectral' | 'phonetic'
      - feature_type: depends on representation
          spectral: 'mfcc' | 'low' | 'mid' | 'high'
          phonetic: 'mfcc' | 'low' | 'mid' | 'high'
      - model_type: 'mlp' | 'tdnn' | 'cnn'
      - output_base_dir: root folder that contains feature subtrees
      - embedding_dim, hidden_layers, lr, epochs, batch_size, data_fraction

    notes
    - cosine embedding loss expects targets in {+1, -1}; all datasets follow that for siamese pairs.
    - for phonetic features, global mean/std are computed (or loaded) before dataset construction.
    """
    def __init__(self, config):
        """store config, pick device, and build the dataset that will back the dataloader."""
        self.config = config
        self.representation_type = config["representation_type"]  # 'spectral' or 'phonetic'
        self.feature_type = config["feature_type"]  # 'mfcc', 'low', 'mid', 'high'
        self.model_type = config["model_type"]  # 'mlp', 'tdnn', 'cnn'
        self.embedding_dim = config.get("embedding_dim", 128)
        self.hidden_layers = config.get("hidden_layers", 2)
        self.lr = config.get("lr", 0.0003)
        self.epochs = config.get("epochs", 10)
        self.batch_size = config.get("batch_size", 64)
        self.data_fraction = config.get("data_fraction", 1.0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = self._load_dataset()
       
    def _load_dataset(self):
        """choose and construct the dataset instance for the given representation/feature."""
        output_base_dir = Path(self.config["output_base_dir"])

        if self.representation_type == "spectral":
            root_dir = output_base_dir / "features" / "spectral" / self.feature_type

            if self.feature_type == "mfcc":
                return MFCCDataset(
                    data_dir=str(root_dir),
                    mode="train",
                    data_fraction=self.data_fraction
                )
            elif self.feature_type in ["low", "mid", "high"]:
                return CNNMelDataset(
                    data_path=str(root_dir),
                    data_fraction=self.data_fraction
                )
            else:
                raise ValueError(f"Unsupported spectral feature type: {self.feature_type}")

        elif self.representation_type == "phonetic":
            root_dir = output_base_dir / "features" / "phonetic" / self.feature_type

            # compute or load global stats for normalization
            stats_dir = Path("global_stats") / self.feature_type
            stats_dir.mkdir(parents=True, exist_ok=True)

            mean_path = stats_dir / "mean.npy"
            std_path = stats_dir / "std.npy"

            if not mean_path.exists() or not std_path.exists():
                print(f"📊 Computing global stats for: {self.feature_type}")
                global_mean, global_std = compute_global_mean_std(
                    feature_dir=root_dir,
                    mean_save_path=mean_path,
                    std_save_path=std_path,
                    verbose=True
                )
            else:
                global_mean = np.load(mean_path)
                global_std = np.load(std_path)
                print(f" Loaded global stats for: {self.feature_type}")

            if self.feature_type == "mfcc":
                return ExplicitPairDataset(
                    root_dir=root_dir,
                    data_fraction=self.data_fraction,
                    global_mean=global_mean,
                    global_std=global_std
                )
            elif self.feature_type == "low":
                return Explicit_Low_Mel_PairDataset(
                    root_dir=root_dir,
                    data_fraction=self.data_fraction,
                    global_mean=global_mean,
                    global_std=global_std
                )
            elif self.feature_type == "mid":
                return Explicit_Mid_Mel_PairDataset(
                    root_dir=root_dir,
                    data_fraction=self.data_fraction,
                    global_mean=global_mean,
                    global_std=global_std
                )
            elif self.feature_type == "high":
                return Explicit_High_Mel_PairDataset(
                    root_dir=root_dir,
                    data_fraction=self.data_fraction,
                    global_mean=global_mean,
                    global_std=global_std
                )
        else:
            raise ValueError(f"Unknown representation type: {self.representation_type}")


    def _init_model(self, x_sample):
        """
        infer input shape from a sample batch and create the selected model.

        rules
        - mlp/tdnn: last-dim treated as input_dim
        - cnn: expects (N, C=1, H, W) after adding a channel dim if needed
        """
        if len(x_sample.shape) == 3:
            x_sample = x_sample.unsqueeze(1)  # add channel dimension

        input_shape = x_sample.shape
        if len(x_sample.shape) == 3:
            x_sample = x_sample.unsqueeze(1)  # add channel dimension
        input_shape = x_sample.shape
        if self.model_type == "mlp":
            input_dim = input_shape[-1]
            return MLP(input_dim=input_dim, embedding_dim=self.embedding_dim, hidden_layers=self.hidden_layers)

        elif self.model_type == "tdnn":
            input_dim = input_shape[-1]
            return TDNN(input_dim=input_dim, embedding_dim=self.embedding_dim, hidden_layers=self.hidden_layers)

        elif self.model_type == "cnn":
            return CNNMelEmbeddingModel(
                input_channels=1,
                input_height=input_shape[2],
                input_width=input_shape[3],
                embedding_dim=self.embedding_dim
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self):
        """standard training loop with cosine embedding loss on pair batches."""
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        if len(dataloader) == 0:
            raise ValueError("DataLoader is empty. No data found in the dataset!")

        first_batch = next(iter(dataloader))
        x_sample = first_batch[0]  # <--- THIS LINE NEEDS TO COME BEFORE INIT_MODEL

        # if len(x_sample.shape) == 3:
        #     x_sample = x_sample.unsqueeze(1)
        print(type(x_sample))
        self.model = self._init_model(x_sample).to(self.device) 

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CosineEmbeddingLoss(margin=0.5)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for x1, x2, label in dataloader:
                x1, x2, label = x1.to(self.device), x2.to(self.device), label.to(self.device).float()
                emb1 = self.model(x1)
                emb2 = self.model(x2)
                loss = criterion(emb1, emb2, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"[Epoch {epoch+1}/{self.epochs}] Loss: {total_loss:.4f}")

    def save(self):
        """save model weights under explicit_models/<rep>_<feature>_based/ with a simple name schema."""
        save_dir = Path("explicit_models") / f"{self.representation_type}_{self.feature_type}_based"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_name = f"{self.model_type}_h{self.hidden_layers}_df{int(self.data_fraction * 100)}.pt"
        torch.save(self.model.state_dict(), save_dir / model_name)
        print(f"Model saved to: {save_dir / model_name}")

    def run(self):
        """one-shot convenience: train then save."""
        self.train()
        self.save()


def test_spectral_model_training():
    """smoke test for spectral models across feature types (short epochs, small fraction)."""
    feature_types = ["mfcc", "low", "mid", "high"]
    for feature_type in feature_types:
        print(f"\nTraining spectral model with feature type: {feature_type}")
        modeltype = None
        if feature_type == 'mfcc':
            modeltype = "tdnn"
        elif feature_type in [ "low", "mid", "high"] :
            modeltype = 'cnn'
        config = {
            "representation_type": "spectral",
            "feature_type": feature_type,
            "model_type": modeltype,              # CNN for spectral
            "data_fraction": 0.05,
            "embedding_dim": 128,
            "hidden_layers": 1,
            "lr": 0.001,
            "epochs": 1,
            "batch_size": 16
        }

        trainer = SpeakerEmbeddingModelTrainer(config)
        trainer.run()  # Trains and saves


def test_phonetic_model_training():
    """smoke test for phonetic models across feature types (short epochs, small fraction)."""
    feature_types = ["mfcc", "low", "mid", "high"]
    for feature_type in feature_types:
        print(f"\nTraining phonetic model with feature type: {feature_type}")
        config = {
            "representation_type": "phonetic",
            "feature_type": feature_type,
            "model_type": "mlp",              # MLP or TDNN for phonetic
            "data_fraction": 0.05,
            "embedding_dim": 128,
            "hidden_layers": 1,
            "lr": 0.001,
            "epochs": 1,
            "batch_size": 16
        }

        trainer = SpeakerEmbeddingModelTrainer(config)
        trainer.run()  # Trains and saves


if __name__ == "__main__":
    test_spectral_model_training()
    test_phonetic_model_training()
