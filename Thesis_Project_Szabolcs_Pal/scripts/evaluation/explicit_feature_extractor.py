"""
explicit (hand-crafted) feature → embedding extraction for phonetic models.

what it does
- wraps four extractors (mfcc, low/mid/high mel bands), applies per-feature
  normalization using precomputed global mean/std, then passes through an mlp
  to get a fixed-size embedding.
- can run on an entire folder tree of *.npy wave chunks, or on a subset defined
  by a grid json (same schema you use elsewhere).
- writes one *.npy embedding per input chunk, mirroring the input directory structure.

notes
- normalization differs per feature set (small log/scale tweaks are applied first).
- the mlp weights are loaded from disk and evaluated on cpu by default.
- grid-based filtering: when `grid_path` is provided, only the listed rel paths are processed,
  and outputs are nested under <output_root>/<grid_version>/<grid_name>/<feature>/...
- caution: `FeatureProcessor.model_name` is derived from `model_path`; make sure your path format
  yields one of {'mfcc','low','mid','high'} for the normalization branches below.
"""

import os
import numpy as np
from tqdm import tqdm
import json
import glob
import torch
import sys


# Dynamically append project root (adjust as needed)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)


from scripts.preprocessing.extract_mfcc_expl import extract_mfcc_complement_features
from scripts.preprocessing.extract_low_mel_band_expl import extract_low_melband_features
from scripts.preprocessing.extract_mid_mel_band_expl import extract_mid_melband_features
from scripts.preprocessing.extract_high_mel_band_expl import extract_high_melband_features
from models.mlp import MLP


class FeatureProcessor:
    """
    minimal wrapper: extractor → normalize → mlp → embedding.

    parameters
    ----------
    extractor_func : callable
        function(waveform, sr) → feature vector (np.ndarray).
    model_path : str
        path to the trained mlp checkpoint.
    global_mean, global_std : np.ndarray
        per-feature statistics to standardize inputs (after small transforms).
    model_class : type
        torch module class to instantiate (e.g., MLP).
    input_dim : int
        feature length expected by the mlp.

    behavior
    - loads the model on cpu, switches to eval.
    - applies feature-set specific transforms before standardization.
    """
    def __init__(self, extractor_func, model_path, global_mean, global_std, model_class, input_dim):
        self.extractor_func = extractor_func
        self.model_path = model_path
        self.global_mean = global_mean
        self.global_std = global_std
        self.model = self._load_model(model_class, input_dim, embedding_dim=128, hidden_layers=2)
        # note: this parsing assumes a path like ".../<name>_based/..." → 'mfcc'/'low'/'mid'/'high'
        folders = self.model_path.split('/')[1]
        self.model_name = folders.split('_')[0]

    def _load_model(self, model_class, input_dim, embedding_dim=64, hidden_layers=2):
        """
        build the mlp and load weights.

        returns
        -------
        torch.nn.Module in eval() mode (on cpu).
        """
        model = model_class(input_dim=input_dim, embedding_dim=embedding_dim, hidden_layers=hidden_layers)
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def transform_and_normalize(self, vec: np.ndarray) -> np.ndarray:
        """
        feature-set specific preprocessing followed by standardization.

        notes
        - cleans NaN/Inf first.
        - applies small log/scale tweaks tailored to each feature group.
        - standardizes with (x - mean) / std, guarding against tiny/invalid stds.
        """
        if self.model_name == 'mfcc':

            # vec = vec.astype(np.float32).reshape(-1)
            vec = np.nan_to_num(np.array([v if v is not None else np.nan for v in vec], dtype=np.float32),
                    nan=0.0, posinf=0.0, neginf=0.0)

            vec[0:5] = np.clip(vec[0:5], a_min=0.0, a_max=None)
            vec[0:5] = np.log(vec[0:5] + 1e-3)
            vec[5:9] = vec[5:9] * 1000

            safe_std = np.where(self.global_std < 1e-6, 1.0, self.global_std)
            normalized_vec = (vec - self.global_mean) / (safe_std + 1e-8)

            normalized_vec[~np.isfinite(normalized_vec)] = 0.0
            return normalized_vec

        if self.model_name == 'low':
            # vec = vec.astype(np.float32).reshape(-1)
            vec = np.nan_to_num(np.array([v if v is not None else np.nan for v in vec], dtype=np.float32),
                    nan=0.0, posinf=0.0, neginf=0.0)
            assert vec.shape[0] == 9, f"Expected 9 features, got {vec.shape}"

            # Replace NaNs and Infs
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

            # Feature-specific transformations
            vec[4] = np.log(vec[4] + 1e-3)                      # Log large value
            vec[0:4] = np.log(np.clip(vec[0:4], a_min=1e-3, a_max=None))  # Log remaining large values
            vec[5:9] = vec[5:9] * 1000                          # Boost tiny values

            # Safe normalization
            safe_mean = np.nan_to_num(self.global_mean, nan=0.0, posinf=0.0, neginf=0.0)
            safe_std = np.where(
                (np.isnan(self.global_std)) | (np.isinf(self.global_std)) | (self.global_std < 1e-6),
                1.0,
                self.global_std
            )

            normalized_vec = (vec - safe_mean) / (safe_std + 1e-8)

            # Final cleanup
            normalized_vec[~np.isfinite(normalized_vec)] = 0.0

            return normalized_vec

        if self.model_name == 'mid':
            # vec = vec.astype(np.float32).reshape(-1)
            vec = np.nan_to_num(np.array([v if v is not None else np.nan for v in vec], dtype=np.float32),
                    nan=0.0, posinf=0.0, neginf=0.0)
            assert vec.shape[0] == 6, f"Expected 6 features, got {vec.shape}"

            # Replace NaNs and Infs in the feature vector
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

            # Log-transform large dynamic range features
            vec[1:3] = np.log(vec[1:3] + 1e-3)

            # Prepare safe mean and std from class
            safe_mean = np.nan_to_num(self.global_mean, nan=0.0, posinf=0.0, neginf=0.0)
            safe_std = np.where(
                (np.isnan(self.global_std)) | (np.isinf(self.global_std)) | (self.global_std < 1e-6),
                1.0,
                self.global_std
            )

            # Normalize
            normalized_vec = (vec - safe_mean) / (safe_std + 1e-8)

            # Replace any remaining non-finite values
            normalized_vec[~np.isfinite(normalized_vec)] = 0.0

            return normalized_vec
        
        if self.model_name == 'high':

            # vec = vec.astype(np.float32).reshape(-1)
            vec = np.nan_to_num(np.array([v if v is not None else np.nan for v in vec], dtype=np.float32),
                    nan=0.0, posinf=0.0, neginf=0.0)
            assert vec.shape[0] == 11, f"Expected 11 features, got {vec.shape}"

            # Clean raw input: replace NaNs and Infs
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

            # Feature-specific preprocessing
            vec[1:4] = np.log(np.clip(vec[1:4], a_min=1e-3, a_max=None))  # log-scale large features
            vec[4:7] = vec[4:7] * 1000                                    # scale up tiny values

            # Clean mean and std
            safe_mean = np.nan_to_num(self.global_mean, nan=0.0, posinf=0.0, neginf=0.0)
            safe_std = np.where(
                (np.isnan(self.global_std)) | (np.isinf(self.global_std)) | (self.global_std < 1e-6),
                1.0,
                self.global_std
            )

            # Normalize
            normalized_vec = (vec - safe_mean) / (safe_std + 1e-8)

            # Final cleanup: replace anything unexpected after norm
            normalized_vec[~np.isfinite(normalized_vec)] = 0.0

            return normalized_vec

    def extract_embedding(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        run extractor → normalize → mlp to produce one embedding.

        parameters
        ----------
        waveform : np.ndarray
            audio samples as loaded from disk (matching your extractors).
        sr : int
            sampling rate fed to the extractor.

        returns
        -------
        np.ndarray
            embedding vector (1d).
        """
        features = self.extractor_func(waveform, sr)
        normed = self.transform_and_normalize(features)
        with torch.no_grad():
            tensor = torch.tensor(normed, dtype=torch.float32).unsqueeze(0)
            embedding = self.model(tensor)
        return embedding.squeeze(0).numpy()


def compute_explicit_embeddings(input_dir, output_dir, processor, target_files=None, sr=16000):
    """
    walk `input_dir` for *.npy chunks, compute embeddings, and mirror-save to `output_dir`.

    parameters
    ----------
    input_dir : str | Path
        root of wave chunk files (npy).
    output_dir : str | Path
        root where embeddings are saved (npy), preserving relative subfolders.
    processor : FeatureProcessor
        configured wrapper to compute a single embedding.
    target_files : set[str] | None
        if provided, only process these relative paths (used with grid filtering).
    sr : int
        sampling rate forwarded to the extractor.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, "**", "*.npy"), recursive=True)

    for file_path in tqdm(files, desc=f"Computing embeddings"):
        relative_path = os.path.relpath(file_path, input_dir)
        normalized_path = relative_path.replace(os.sep, "/")
        if target_files and normalized_path not in target_files:
            continue

        try:
            waveform = np.load(file_path)
            emb = processor.extract_embedding(waveform, sr=sr)
        except Exception as e:
            print(f"⚠️ Failed to process {file_path}: {e}")
            continue

        out_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, emb)


def main_explicit(processors_dict, input_dirs_dict, output_root="embeddings", grid_path=None):
    """
    driver: run explicit embedding extraction for each feature type.

    parameters
    ----------
    processors_dict : dict[str, FeatureProcessor]
        mapping {'mfcc'|'low'|'mid'|'high' -> processor}.
    input_dirs_dict : dict[str, str | Path]
        mapping feature name → input directory with wave chunks.
    output_root : str
        base folder to write outputs to.
    grid_path : str | Path | None
        optional json describing which relative chunk files to process.
        when provided, outputs nest under <output_root>/<grid_version>/<grid_name>/.

    behavior
    - when grid_path is set, loads its "speakers" → "files" list to filter inputs.
    - iterates over feature types and calls `compute_explicit_embeddings`.
    """
    instance_paths = None
    if grid_path:
        with open(grid_path, "r") as gf:
            grid_data = json.load(gf)
        grid_name = os.path.splitext(os.path.basename(grid_path))[0]           # e.g. 20s_20i
        grid_version = os.path.basename(os.path.dirname(grid_path))            # e.g. grids_1
        output_base_dir = os.path.join(output_root, grid_version, grid_name)
        instance_paths = {
            file.replace("\\", "/")
            for speaker in grid_data["speakers"]
            for file in speaker["files"]
        }
    else:
        output_base_dir = output_root

    for name, processor in processors_dict.items():
        input_dir = input_dirs_dict[name]
        output_dir = os.path.join(output_base_dir, name)
        print(f"🔍 Extracting [{name}] features from: {input_dir}")
        compute_explicit_embeddings(input_dir, output_dir, processor, target_files=instance_paths)




# Create processor dictionary
processors = {
    "mfcc": FeatureProcessor(
        extractor_func=extract_mfcc_complement_features,
        model_path="trained_models/explicit_models/mfcc_based/MLP_h2_df100.pt",
        global_mean=np.load("feature_stats/global_mfcc_feature_mean_per_feature.npy"),
        global_std=np.load("feature_stats/global_mfcc_feature_std_per_feature.npy"),
        model_class=MLP,
        input_dim=11
    ),
    "low": FeatureProcessor(
        extractor_func=extract_low_melband_features,
        model_path="trained_models/explicit_models/low_mel_based/MLP_h2_df100.pt",
        global_mean=np.load("feature_stats/global_low_mel_feature_mean_per_feature.npy"),
        global_std=np.load("feature_stats/global_low_mel_feature_std_per_feature.npy"),
        model_class=MLP,
        input_dim=9
    ),
    "mid": FeatureProcessor(
        extractor_func=extract_mid_melband_features,
        model_path="trained_models/explicit_models/mid_mel_based/MLP_h2_df100.pt",
        global_mean=np.load("feature_stats/global_mid_mel_feature_mean_per_feature.npy"),
        global_std=np.load("feature_stats/global_mid_mel_feature_mean_per_feature.npy"),  # note: path points to *mean* file; verify
        model_class=MLP,
        input_dim=6
    ),
    "high": FeatureProcessor(
        extractor_func=extract_high_melband_features,
        model_path="trained_models/explicit_models/high_mel_based/MLP_h2_df100.pt",
        global_mean=np.load("feature_stats/global_high_mel_feature_mean_per_feature.npy"),
        global_std=np.load("feature_stats/global_high_mel_feature_mean_per_feature.npy"),  # note: path points to *mean* file; verify
        model_class=MLP,
        input_dim=11
    )
}

# Specify where input waveforms are stored
input_dirs = {
    "mfcc": "data/processed_test/wave_chunks_2s",
    "low": "data/processed_test/wave_chunks_2s",
    "mid": "data/processed_test/wave_chunks_2s",
    "high": "data/processed_test/wave_chunks_2s",
}

# grid_versions = ['grids_test_1','grids_test_2','grids_test_3']
# for grid_folder in grid_versions:
#     for grid_file in os.listdir(grid_folder):
#         grid_path = os.path.join(grid_folder, grid_file)
#         print(f" Processing: {grid_path}")
#         main_explicit(processors, input_dirs, output_root="embeddings_phonetic", grid_path=grid_path)


# ['grids_test_1',
# 'grids_test_2',
# 'grids_test_3',
# 'queries_test_1',
# 'queries_test_2',
# 'queries_test_3',
# 'unknown_queries_1',
# 'unknown_queries_2',
# 'unknown_queries_3']
