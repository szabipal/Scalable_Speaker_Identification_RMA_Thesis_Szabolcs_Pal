"""
compute spectral embeddings with trained models (tdnn/mlp/cnn).

what this script does
- loads model configs from a trained models directory.
- figures out the right input folder for each model (mfcc/mel banded).
- runs the model over *.npy feature files and saves per-file embeddings.
- optionally filters which files to process via a grid json (same schema you use elsewhere).

notes
- output layout mirrors the input layout, nested under `output_root[/grid_version/grid_name]`.
- tdnn expects [time, feat] (we transpose a sample to infer shape if needed).
- cnn models expect [b, 1, H, W]. mlp/tdnn expect [b, ...] vectors/sequences.
"""

import os
import sys
import glob
import json
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# make project root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.tdnn import TDNN
from models.mlp import MLP
from models.cnn_mel_embedding_model import CNNMelEmbeddingModel


def load_model(model_path, model_type, input_dim, embedding_dim, hidden_layers):
    """
    create the correct model (tdnn/mlp/cnn) and load its weights.

    parameters
    ----------
    model_path : str
        path to the .pt checkpoint.
    model_type : str
        "TDNN" | "MLP" | "CNNMelEmbeddingModel".
    input_dim : tuple[int, int] | int
        network input shape; for tdnn/mlp we flatten/infer as needed.
    embedding_dim : int
        output embedding dimension used at training time.
    hidden_layers : int
        how many hidden layers your training used.

    returns
    -------
    torch.nn.Module in eval() mode (on cpu).
    """
    if model_type in ["TDNN", "MLP"]:
        flat_dim = input_dim[0] * input_dim[1] if isinstance(input_dim, tuple) else input_dim

    if model_type == "TDNN":
        model = TDNN(input_dim=input_dim[0], embedding_dim=embedding_dim, hidden_layers=hidden_layers)

    elif model_type == 'MLP':
        model = MLP(input_dim=flat_dim, embedding_dim=embedding_dim, hidden_layers=hidden_layers)

    elif model_type == "CNNMelEmbeddingModel":
        model = CNNMelEmbeddingModel(
            input_channels=1,
            input_height=input_dim[0],
            input_width=input_dim[1],
            embedding_dim=embedding_dim
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def compute_embeddings(input_dir, output_dir, model, is_cnn=False, target_files=None):
    """
    run the model over all *.npy feature files in `input_dir` and save embeddings into `output_dir`.

    parameters
    ----------
    input_dir : str | Path
        root folder containing feature .npy files (possibly nested).
    output_dir : str | Path
        where to mirror-save the produced embeddings (.npy).
    model : torch.nn.Module
        loaded model in eval mode.
    is_cnn : bool
        if true, add [B=1, C=1, H, W] dims; else keep batch-only for mlp/tdnn.
    target_files : set[str] | None
        optional set of relative paths (with forward slashes) to restrict processing.

    behavior
    - mirrors subfolder structure under `output_dir`.
    - if `target_files` is provided, skips anything not in that allowlist.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, "**", "*.npy"), recursive=True)

    if files is None:
        print('Error')
    else:
        print(len(files), files[:5])

    if not files:
        print(f"❗ No input files found in {input_dir}")
        return

    for file_path in tqdm(files, desc=f"Processing {input_dir}"):
        relative_path = os.path.relpath(file_path, input_dir)

        # normalize rel path for matching against grid allowlist
        normalized_path = relative_path.replace(os.sep, "/")
        if target_files and normalized_path not in target_files:
            continue

        # small peek to help debug grid filtering
        if target_files and len(target_files) > 0:
            print("📋 First grid file:", list(target_files)[0])

        data = np.load(file_path)
        tensor = torch.tensor(data, dtype=torch.float32)

        if is_cnn:
            # cnn expects [B, C=1, H, W]
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
        else:
            # mlp/tdnn: [B, ...]
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            # some models might return (loss, emb); handle both
            embedding = output[1] if isinstance(output, tuple) else output
            print("📤 Embedding output shape:", embedding.shape)

        out_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        print()
        print(f'{out_path} this is the path it is saved to')
        print()
        np.save(out_path, embedding.squeeze(0).numpy())


def main(
    trained_models_dir="trained_models_spectral_large_data",
    output_root="embeddings_spectral_final",
    grid_path=None,
    input_base_path="data/processed_dev"
):
    """
    controller: iterate over trained models, find their inputs, and dump embeddings.

    parameters
    ----------
    trained_models_dir : str
        directory with model families and subfolders holding config.json + model.pt.
    output_root : str
        root folder for embeddings; grid outputs go under /<grid_version>/<grid_name>/.
    grid_path : str | Path | None
        optional json with a "speakers" → "files" list to restrict which chunks get processed.
    input_base_path : str
        base path for feature inputs (e.g., data/processed_<split>).

    flow
    - if grid json is given, collect allowed relative paths and build a grid-specific output base.
    - for each model subfolder:
        * read config.json (model_type, segment_len, data_path, etc.)
        * deduce input_dir under `input_base_path`.
        * infer input_dim from sample *.npy if needed.
        * load model and run `compute_embeddings`.
    """
    import os
    import json
    import numpy as np

    # optional allowlist from a grid json
    instance_paths = None
    if grid_path:
        with open(grid_path, "r") as gf:
            grid_data = json.load(gf)

        grid_name = os.path.splitext(os.path.basename(grid_path))[0]            # "40s_1i"
        grid_version = os.path.basename(os.path.dirname(grid_path))             # "grids_1"
        output_base_dir = os.path.join(output_root, grid_version, grid_name)   # changed to use `output_root`

        instance_paths = set()
        for speaker in grid_data["speakers"]:
            for file in speaker["files"]:
                normalized_file = file.replace("\\", "/")
                instance_paths.add(normalized_file)
    else:
        output_base_dir = output_root

    # scan model families
    for top_folder in os.listdir(trained_models_dir):
        top_path = os.path.join(trained_models_dir, top_folder)
        if not os.path.isdir(top_path):
            continue

        for subfolder in os.listdir(top_path):
            model_path = os.path.join(top_path, subfolder)
            if not os.path.isdir(model_path):
                continue

            config_path = os.path.join(model_path, "config.json")
            model_file = os.path.join(model_path, "model.pt")

            if not os.path.exists(config_path) or not os.path.exists(model_file):
                print(f"Skipping {model_path}: missing config or model file.")
                continue

            with open(config_path, "r") as f:
                config = json.load(f)

            model_type = config["model_type"]
            segment_len = config["segment_len"]
            embedding_dim = config.get("embedding_dim", 128)
            hidden_layers = config.get("hidden_layers", 2)
            feature_subdir = os.path.basename(config["data_path"].rstrip("/"))

            # choose proper input folder under input_base_path
            if "CNN" in subfolder:
                band = feature_subdir.split("_")[-1]
                input_dir = os.path.join(input_base_path, "mel_features_banded", band)
            else:
                input_dir = os.path.join(input_base_path, feature_subdir)

            print(f"→ Looking in: {input_dir}")
            output_dir = os.path.join(output_base_dir, top_folder)

            # quick sample probe to infer input_dim if config didn't specify
            sample = None
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.endswith(".npy"):
                        sample_path = os.path.join(root, file)
                        sample = np.load(sample_path)
                        if model_type == "TDNN" and sample.shape[0] > sample.shape[1]:
                            sample = sample.T
                        break
                if sample is not None:
                    break

            if sample is None:
                print(f"⚠️  No .npy files found in {input_dir} to infer input shape.")
                continue

            # tdnn: ensure [time, feat]
            if model_type == "TDNN" and sample.shape[0] > sample.shape[1]:
                sample = sample.T

            # prefer config input_dim if present; else fall back to sample shape
            if "input_dim" in config and isinstance(config["input_dim"], int):
                input_dim = (config["input_dim"], segment_len)
            elif isinstance(config.get("input_dim"), list):
                input_dim = tuple(config["input_dim"])
            else:
                input_dim = sample.shape

            print(f"→ Final input_dim for {model_type}: {input_dim}")

            model = load_model(model_file, model_type, input_dim, embedding_dim, hidden_layers)
            is_cnn = model_type == "CNNMelEmbeddingModel"

            compute_embeddings(input_dir, output_dir, model, is_cnn=is_cnn, target_files=instance_paths)




            input_width=input_dim[1],
            embedding_dim=embedding_dim
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def compute_embeddings(input_dir, output_dir, model, is_cnn=False, target_files=None):


    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, "**", "*.npy"), recursive=True)

    if files is None:
        print('Error')
    else:
        print(len(files), files[:5])

    if not files:
        print(f"❗ No input files found in {input_dir}")
        return

    for file_path in tqdm(files, desc=f"Processing {input_dir}"):
        relative_path = os.path.relpath(file_path, input_dir)

        normalized_path = relative_path.replace(os.sep, "/")
        if target_files and normalized_path not in target_files:
            continue

        if target_files and len(target_files) > 0:
            print("First grid file:", list(target_files)[0])



        data = np.load(file_path)
        tensor = torch.tensor(data, dtype=torch.float32)

        if is_cnn:
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
        else:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            embedding = output[1] if isinstance(output, tuple) else output
            print("📤 Embedding output shape:", embedding.shape)

        out_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        print()
        print(f'{out_path} this is the path it is saved to')
        print()
        np.save(out_path, embedding.squeeze(0).numpy())




# if __name__ == "__main__":
#     grid_versions=['EXPLICIT_ENSEMBLE/grids_1','EXPLICIT_ENSEMBLE/grids_2','EXPLICIT_ENSEMBLE/grids_3','EXPLICIT_ENSEMBLE/grids_4', 'EXPLICIT_ENSEMBLE/grids_5']
#     for grid in grid_versions:
#         grids=os.listdir(grid)
#         for g in grids:
#             print(f'{grid}/{g})')
#             main(grid_path=f'{grid}/{g}')
# if __name__ == "__main__":
#     grid_versions = [
#         'grids_test_1',
#         'grids_test_2',
#         'grids_test_3'
#     ]
#     for grid in grid_versions:
#         grids = os.listdir(grid)
#         for g in grids:
#             grid_path = os.path.join(grid, g)
#             print(f"🔄 Running on grid: {grid_path}")
#             main(grid_path=grid_path)








# if __name__ == "__main__":
#     grid_versions=['EXPLICIT_ENSEMBLE/grids_1','EXPLICIT_ENSEMBLE/grids_2']#,'EXPLICIT_ENSEMBLE/grids_3','EXPLICIT_ENSEMBLE/grids_4', 'EXPLICIT_ENSEMBLE/grids_5']
#     # grid_versions=['queries_test_1', 'queries_test_2', 'queries_test_3']
#     for grid in grid_versions:
#         grids=os.listdir(grid)
#         for g in grids:
#             print(f'{grid}/{g})')
#             main(grid_path=f'{grid}/{g}')
# if __name__ == "__main__":
#     grid_versions = [
#         'EXPLICIT_ENSEMBLE/queries_1',
#         'EXPLICIT_ENSEMBLE/queries_2',
#         'EXPLICIT_ENSEMBLE/queries_3',
#         'EXPLICIT_ENSEMBLE/queries_4',
#         'EXPLICIT_ENSEMBLE/queries_5'
#     ]
#     for grid in grid_versions:
#         grids = os.listdir(grid)
#         for g in grids:
#             grid_path = os.path.join(grid, g)
#             print(f"🔄 Running on grid: {grid_path}")
#             main(grid_path=grid_path)

# if __name__ == "__main__":

#     main(
#         trained_models_dir="trained_models_spectral_large_data",  
#         output_root="embeddings_dev_large_data"                       
#     )


#     grid_versions = [
#         'EXPLICIT_ENSEMBLE/unknown_queries_1',
#         'EXPLICIT_ENSEMBLE/unknown_queries_2',
#         'EXPLICIT_ENSEMBLE/unknown_queries_3',
#         'EXPLICIT_ENSEMBLE/unknown_queries_4',
#         'EXPLICIT_ENSEMBLE/unknown_queries_5'
#     ]



