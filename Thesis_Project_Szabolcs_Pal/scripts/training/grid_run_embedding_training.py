
import os
import sys
import itertools
import json
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_pair_embedding_model import train  # updated train version

OUTPUT_DIR = Path("trained_models")

# === 1. MFCC + TDNN Grid Search ===
mfcc_grid = itertools.product(
    ["mfcc"], ["TDNN"], [100], [False], [4]
)

base_config = {
    "embedding_dim": 32,
    "lr": 0.001,
    "batch_size": 10,
    "epochs": 50,
    "data_fraction": 1.0,
    "loss": "cosine"
}

feature_path_map = {
    "mfcc": "data/mfcc",
    "formant": "data/formant_data",
    "mel_band_low": "data/processed/mel_features_banded/low",
    "mel_band_mid": "data/processed/mel_features_banded/mid",
    "mel_band_high": "data/processed/mel_features_banded/high"
}

# === Run MFCC + TDNN models ===
for feature_type, model_type, segment_len, moving_window, hidden_layers in mfcc_grid:
    config = base_config.copy()
    config.update({
        "feature_type": feature_type,
        "model_type": model_type,
        "segment_len": segment_len,
        "moving_window": moving_window,
        "hidden_layers": hidden_layers,
        "dataset": "GeneralPairDataset",
        "data_path": feature_path_map[feature_type]
    })

    model_subdir = f"{model_type}_len{segment_len}_win{moving_window}_layers{hidden_layers}"
    model_output_dir = OUTPUT_DIR / feature_type / model_subdir
    model_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n Training TDNN on {feature_type.upper()} features...")
    model, loss_values = train(config)

    torch.save(model.state_dict(), model_output_dir / "model.pt")
    with open(model_output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    with open(model_output_dir / "loss.json", "w") as f:
        json.dump(loss_values, f, indent=4)

# === 2. Formant + MLP (Fixed setup) ===
formant_config = base_config.copy()
formant_config.update({
    "feature_type": "formant",
    "dataset": "FormantPairDataset",
    "data_path": feature_path_map["formant"],
    "model_type": "MLP",
    "input_dim": 7,
    "segment_len": 1,
    "moving_window": False,
    "hidden_layers": 3
})

model_output_dir = OUTPUT_DIR / "formant" / "MLP_len1_winFalse_layers3"
model_output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n Training MLP on FORMANT features...")
model, loss_values = train(formant_config)

torch.save(model.state_dict(), model_output_dir / "model.pt")
with open(model_output_dir / "config.json", "w") as f:
    json.dump(formant_config, f, indent=4)
with open(model_output_dir / "loss.json", "w") as f:
    json.dump(loss_values, f, indent=4)

# === 3. CNN on MEL BANDS (LOW / MID / HIGH treated as separate models) ===
mel_bands = ["low", "mid", "high"]
for band in mel_bands:
    config = base_config.copy()
    config.update({
        "feature_type": f"mel_band_{band}",
        "dataset": "CNNSpectrogramDataset",
        "data_path": feature_path_map[f"mel_band_{band}"],
        "model_type": "CNNMelEmbeddingModel",
        "input_dim": 40,
        "segment_len": 100,
        "moving_window": False,
        "hidden_layers": 1
    })

    model_output_dir = OUTPUT_DIR / f"mel_band_{band}" / "CNN_len100_winFalse_layers1"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n Training CNN on MEL BAND: {band.upper()}")
    model, loss_values = train(config)

    torch.save(model.state_dict(), model_output_dir / "model.pt")
    with open(model_output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    with open(model_output_dir / "loss.json", "w") as f:
        json.dump(loss_values, f, indent=4)
