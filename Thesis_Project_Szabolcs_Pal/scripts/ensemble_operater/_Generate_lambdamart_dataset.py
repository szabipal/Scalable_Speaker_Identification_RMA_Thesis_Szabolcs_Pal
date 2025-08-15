"""
builds lambdaMART-ready datasets for spectral/phonetic models.

what it does
- loads dev embeddings to compute similarity/centroid quantiles per feature (used for normalization/scoring).
- iterates train/test grids (and query/unknown_query splits) to merge per-feature embeddings
  into lambdaMART input folders for each grid configuration.
- preserves your existing folder names and mapping between logical features and on-disk folders.
"""

import os
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] 

from EXPLICIT_ENSEMBLE.lambda_mart_dataset import create_merged_lambdamart_dataset
from scripts.evaluation.check_dev import (
    compute_similarity_pairs,
    compute_centroid_distances,
    save_quantiles_to_json,
)
from scripts.evaluation.emb_stats import load_embeddings_to_df

# ---------- Config ----------
all_configs = {
    "spectral": {
       
        "feature_types": ["mfcc", "mel_band_low", "mel_band_high", "mel_band_mid"],
        "quantile_dir": "spectral_cosine_model_quantiles",
        "embedding_root": "embeddings_spectral_final_test",   
        "train_embedding_root": "embeddings_spectral_final_train", 
        "dev_embedding_root": "embeddings_spectral_final_dev",      
        "output_root": "output_spectral_final_test",
        "train_output_root": "output_spectral_final_train",
    },
        "phonetic": {
        "feature_types": ["mfcc", "mid", "low", "high"],
        "quantile_dir": "phonetic_cosine_model_quantiles",
        "embedding_root": "embeddings_phonetic_final_test",
        "train_embedding_root": "embeddings_phonetic_final_train",
        "dev_embedding_root": "embeddings_phonetic_final_dev",
        "output_root": "output_phonetic_final_test",
        "train_output_root": "output_phonetic_final_train"
    },
}


#  sets to use
test_cases = [1, 2, 3]
train_cases = [1, 2, 3, 4, 5]

# Grid-name mapping (49 configs)
grid_map = {
    1: "10s_10i",  2: "20s_10i",  3: "2s_10i",   4: "40s_10i",  5: "5s_10i",  6: "60s_10i",  7: "80s_10i",
    8: "10s_1i",   9: "20s_1i",  10: "2s_1i",  11: "40s_1i", 12: "5s_1i", 13: "60s_1i", 14: "80s_1i",
   15: "10s_20i", 16: "20s_20i", 17: "2s_20i", 18: "40s_20i", 19: "5s_20i", 20: "60s_20i", 21: "80s_20i",
   22: "10s_40i", 23: "20s_40i", 24: "2s_40i", 25: "40s_40i", 26: "5s_40i", 27: "60s_40i", 28: "80s_40i",
   29: "10s_5i",  30: "20s_5i",  31: "2s_5i",  32: "40s_5i",  33: "5s_5i",  34: "60s_5i",  35: "80s_5i",
   36: "10s_60i", 37: "20s_60i", 38: "2s_60i", 39: "40s_60i", 40: "5s_60i", 41: "60s_60i", 42: "80s_60i",
   43: "10s_80i", 44: "20s_80i", 45: "2s_80i", 46: "40s_80i", 47: "5s_80i", 48: "60s_80i", 49: "80s_80i"
}

# Map config feature labels
FEATURE_FOLDER_MAP = {
    "mfcc": "mfcc",
    "mel_band_low": "low",
    "mel_band_mid": "mid",
    "mel_band_high": "high",
    
}

def _ensure_dir(path: str | Path):
    """create directory (and parents) if missing; no-op if it already exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

# ---------- Quantiles from DEV embeddings ----------
def generate_quantiles_from_dev_embeddings(dev_root: str, feature_types: list[str], output_dir: str):
    """
    compute and save similarity/centroid quantiles from dev embeddings.

    reads from:
      <dev_root>/<feature_folder>/
    where feature_folder is derived via FEATURE_FOLDER_MAP.

    steps
    - load per-feature embedding dataframes
    - compute intra/inter cosine similarities and centroid distances
    - write 10/90% quantiles to json files under `output_dir`
    """
    _ensure_dir(output_dir)
    for feature in feature_types:
        feature_folder = FEATURE_FOLDER_MAP.get(feature, feature)
        path = os.path.join(dev_root, feature_folder)
        if not os.path.isdir(path):
            print(f"⚠️ Skipping {path}, does not exist.")
            continue

        df = load_embeddings_to_df(path)
        if df.empty:
            print(f"⚠️ No data in {path}.")
            continue

        print(f"\n📊 Processing quantiles for: {feature} ({feature_folder})")
        intra_sim, inter_sim = compute_similarity_pairs(df["embedding"], df["speaker_id"])
        intra_centroid_dists, inter_centroid_dists = compute_centroid_distances(df["embedding"], df["speaker_id"])

        save_quantiles_to_json(
            feature_name=feature_folder, 
            intra_sim=intra_sim,
            inter_sim=inter_sim,
            intra_centroid=intra_centroid_dists,
            inter_centroid=inter_centroid_dists,
            output_dir=output_dir
        )




def _cfg_for(model_type: str):
    """fetch the config block for a model family; raises on unknown key."""
    if model_type not in all_configs:
        raise ValueError(f"Unknown model_type: {model_type}")
    return all_configs[model_type]


def build_datasets_for_model_split(
    model_type: str,
    split: str,
    query_type: str,
    *,
    cases: list[int] | None = None,
    fraction: float = 1.0,
):
    """
    build lambdaMART datasets for a single (model_type, split, query_type).

    parameters
    ----------
    model_type : 'spectral' | 'phonetic'
    split : 'train' | 'test'
    query_type : 'queries' | 'unknown_queries'
    cases : list[int] | None
        which grid ids to include; defaults to globals (train_cases/test_cases).
    fraction : float
        forwarded to `create_merged_lambdamart_dataset` for optional downsampling.

    behavior
    - ensures quantiles exist (computed from dev embeddings).
    - iterates requested grid ids and all 49 grid configs.
    - writes merged datasets under the configured output roots.
    """
    assert split in ("train", "test"), "split must be 'train' or 'test'"
    assert query_type in ("queries", "unknown_queries"), "query_type must be 'queries' or 'unknown_queries'"

    cfg = _cfg_for(model_type)

    # consulted ChatGPT for error resolution
    cfg_abs = {**cfg}
    for k in ("embedding_root", "train_embedding_root", "dev_embedding_root",
            "output_root", "train_output_root"):
        cfg_abs[k] = str((ROOT / cfg[k]).resolve())

    generate_quantiles_from_dev_embeddings(
        dev_root=cfg["dev_embedding_root"],
        feature_types=cfg["feature_types"],
        output_dir=cfg["quantile_dir"],
    )

    # 1) pick roots & cases
    if split == "train":
        base_root = cfg["train_embedding_root"]
        output_root = cfg["train_output_root"]
        default_cases = train_cases
    else:
        base_root = cfg["embedding_root"]           
        output_root = cfg["output_root"]
        default_cases = test_cases

    case_ids = cases if cases is not None else default_cases

    # 2) feature folder names on disk
    feature_folders = [FEATURE_FOLDER_MAP.get(ft, ft) for ft in cfg["feature_types"]]

    # 3) loop each case/grid set
    for case_id in case_ids:
        grid_root = os.path.join(base_root, f"grid_{case_id}")
        if query_type == "queries":
            q_root = os.path.join(base_root, f"query_{case_id}")
            subdir = f"lambdamart_cosine_dataset_queries_normed{case_id}"
        else:
            q_root = os.path.join(base_root, f"unknown_query_{case_id}"
                                  )
            subdir = f"lambdamart_cosine_dataset_unknown_queries_normed{case_id}"

        # 4) each of the 49 configs
        for cfg_id, grid_name in grid_map.items():
            out_dir = os.path.join(output_root, subdir, grid_name)
            create_merged_lambdamart_dataset(
                config_id=str(cfg_id),
                grid_name=grid_name,
                query_root=q_root,
                grid_root=grid_root,
                feature_types=feature_folders,
                quantile_dir=cfg["quantile_dir"],
                output_path=out_dir,
                fraction=fraction,
            )


def build_datasets_for_model(model_type: str, *, fraction: float = 1.0):
    """
    convenience wrapper to build all split/type combinations for a model family.

    runs:
      (train + test) × (queries + unknown_queries)
    """
    for split in ("train", "test"):
        for qtype in ("queries", "unknown_queries"):
            build_datasets_for_model_split(
                model_type=model_type,
                split=split,
                query_type=qtype,
                fraction=fraction,
            )

# ---------- Main ----------
if __name__=='__main__':
    for model_name, cfg in all_configs.items():
        print(f"\n🚀 Starting processing for model: {model_name.upper()}")

        # 1) Quantiles from DEV embeddings (new structure)
        generate_quantiles_from_dev_embeddings(
            dev_root=cfg["dev_embedding_root"],
            feature_types=cfg["feature_types"],
            output_dir=cfg["quantile_dir"],
        )

        # 2) TEST sets (new structure)

        for test_id in test_cases:
            grid_root   = os.path.join(cfg["embedding_root"], f"grid_{test_id}")
            query_root  = os.path.join(cfg["embedding_root"], f"query_{test_id}")
            unkq_root   = os.path.join(cfg["embedding_root"], f"unknown_query_{test_id}")

            # queries
            for i, grid_name in grid_map.items():
                output_path = os.path.join(
                    cfg["output_root"],
                    f"lambdamart_cosine_dataset_queries_normed{test_id}",
                    grid_name
                )
                create_merged_lambdamart_dataset(
                    config_id=str(i),
                    grid_name=grid_name,
                    query_root=query_root,          
                    grid_root=grid_root,           
                    feature_types=[FEATURE_FOLDER_MAP.get(ft, ft) for ft in cfg["feature_types"]],
                    quantile_dir=cfg["quantile_dir"],
                    output_path=output_path,
                    fraction=1.0
                )

            # unknown_queries
            for i, grid_name in grid_map.items():
                output_path = os.path.join(
                    cfg["output_root"],
                    f"lambdamart_cosine_dataset_unknown_queries_normed{test_id}",
                    grid_name
                )
                create_merged_lambdamart_dataset(
                    config_id=str(i),
                    grid_name=grid_name,
                    query_root=unkq_root,           
                    grid_root=grid_root,            
                    feature_types=[FEATURE_FOLDER_MAP.get(ft, ft) for ft in cfg["feature_types"]],
                    quantile_dir=cfg["quantile_dir"],
                    output_path=output_path,
                    fraction=1.0
                )

        # 3) TRAIN sets 
        for train_id in train_cases:
            grid_root  = os.path.join(cfg["train_embedding_root"], f"grid_{train_id}")
            query_root = os.path.join(cfg["train_embedding_root"], f"query_{train_id}")
        
            for i, grid_name in grid_map.items():
                output_path = os.path.join(
                    cfg["train_output_root"],
                    f"lambdamart_cosine_dataset_queries_train_normed{train_id}",
                    grid_name
                )
                create_merged_lambdamart_dataset(
                    config_id=str(i),
                    grid_name=grid_name,
                    query_root=query_root,         
                    grid_root=grid_root,           
                    feature_types=[FEATURE_FOLDER_MAP.get(ft, ft) for ft in cfg["feature_types"]],
                    quantile_dir=cfg["quantile_dir"],
                    output_path=output_path,
                    fraction=1.0
                )
