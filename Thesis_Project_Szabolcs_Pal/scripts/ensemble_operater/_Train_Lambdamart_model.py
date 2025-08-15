"""
train a lightgbm lambdamart ranker on merged per-grid datasets.

what it does
- gathers csvs produced by your merge step under output_<model>_final_<split>/lambdamart_cosine_dataset_<query_type>_normedX/.
- selects feature columns by prefix (mfcc + mel bands), plus required label/qid.
- converts qids to group sizes, trains a lambdarank model, and evaluates top-1 per query.
- grid-searches a small set of hyperparameters and saves each model + a short stats file.

notes
- expects csv names: <grid_folder>/lambdamart_merged_<grid_folder>.csv
- feature prefixes are chosen per model_type; adjust if your headers differ.
- output models/stats go to EXPLICIT_ENSEMBLE_final/<model>_final_128[_new]/.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] 
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
import itertools

def train_lambdamart_model(
    model_type="spectral",
    split="train",                      # "train" | "test"
    query_type="queries",              # "queries" | "unknown_queries"
    train_folders=(1, 2, 3, 4),        # which normed sets to train on
    eval_folder=5                      # which normed set to eval on
):
    """
    train a lambdarank model from csv folds and write models + simple metrics.

    parameters
    ----------
    model_type : str
        'spectral' or 'phonetic'; controls input roots and feature prefixes.
    split : str
        which split to read (train/test).
    query_type : str
        'queries' or 'unknown_queries'; selects the normed dataset family.
    train_folders : tuple[int]
        which *normed* set ids to use for training (e.g., (1,2,3)).
    eval_folder : int
        which *normed* set id to use for evaluation (e.g., 4 or 5).

    behavior
    - loads and concatenates all train/eval csvs for the chosen sets.
    - filters feature columns by prefix, keeps label/qid, builds lightgbm groups.
    - runs a small grid over lr/leaves/bagging and logs top-1 accuracy.
    - saves each model to txt and a companion stats file with summary info.
    """
    # === roots per model/split ===
    if model_type == "spectral":
        base_path = Path(f"output_spectral_final_{split}")  # e.g., output_spectral_final_test
        output_dir = Path("EXPLICIT_ENSEMBLE_final/spectral_final_128_new")
        # feature prefixes for column filtering
        feature_prefixes = ("mfcc", "mel_band_mid", "mel_band_low", "mel_band_high")
    elif model_type == "phonetic":
        base_path = Path(f"output_phonetic_final_{split}")
        output_dir = Path("EXPLICIT_ENSEMBLE_final/phonetic_final_128")
        feature_prefixes = ("mfcc", "mel_band_mid", "mel_band_low", "mel_band_high")
    else:
        raise ValueError("Invalid model_type. Choose 'phonetic' or 'spectral'.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # === hyperparameters ===
    param_grid = {
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 63, 127],
        'feature_fraction': [1.0],
        'bagging_fraction': [1.0, 0.8],
    }

    # helper to resolve folder like: lambdamart_cosine_dataset_{query_type}_normed{k}
    def normed_folder(k: int) -> Path:
        return base_path / f"lambdamart_cosine_dataset_{query_type}_normed{k}"

    # === collect files ===
    def collect_csvs(normed_idx: int) -> list[Path]:
        root = normed_folder(normed_idx)
        csvs = []
        for grid_folder in sorted(p for p in root.iterdir() if p.is_dir()):
            grid_name = grid_folder.name  # e.g., "10s_10i"
            csv_file = grid_folder / f"lambdamart_merged_{grid_name}.csv"
            if csv_file.exists():
                csvs.append(csv_file)
        if not csvs:
            raise FileNotFoundError(f"No CSVs found under {root}. Check your paths and names.")
        return csvs

    train_files = [f for k in train_folders for f in collect_csvs(k)]
    eval_files  = collect_csvs(eval_folder)

    # === load data ===
    print("Loading training and evaluation data...")
    df_train = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
    df_eval  = pd.concat([pd.read_csv(f) for f in eval_], ignore_index=True)
