#!/usr/bin/env python3
"""
Create a minimal synthetic tree so the ensemble pipeline can be validated
without running preprocessing, embedding, LambdaMART, or inference.

It writes tiny placeholder artifacts for both models ("spectral", "phonetic"):
- raw audio dirs
- processed features dirs (incl. 'features' and 'feature')
- embeddings stubs
- merged LambdaMART datasets for 5 folders (1..5), 1 grid config ("10s_10i")
- fake confidences for (train|test) × (queries|unknown_queries)
- fake thresholds for each model (train-derived)
- final ensemble outputs (so error analysis can read)

Notes:
- Columns are minimal but aligned with typical usage in your previous scripts.
- Adjust paths/filenames if your project’s utils expect different strings.
"""
from pathlib import Path
import os, json
import numpy as np
import pandas as pd
import shutil

ROOT = Path(__file__).resolve().parents[2]

MODELS = ("spectral", "phonetic")
SPLITS_ALL = ("dev", "train", "test")
SPLITS_DATASETS = ("train", "test")
QUERY_TYPES = ("queries", "unknown_queries")
GRID_NAME = "10s_10i"           # one config to keep it tiny
FOLDERS = [1, 2, 3, 4, 5]       # folder ids

# ---------- helpers ----------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_csv(p: Path, df: pd.DataFrame):
    ensure_dir(p.parent)
    df.to_csv(p, index=False)

def write_json(p: Path, obj: dict):
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2))

def write_npy(p: Path, arr):
    ensure_dir(p.parent)
    np.save(p, arr)

def maybe_symlink_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        dst.symlink_to(src, target_is_directory=True)
        print(f"🔗 {dst} -> {src}")
    except Exception:
        print(f"📁 copy {src} -> {dst}")
        shutil.copytree(src, dst)

# ---------- synthetic writers ----------
def make_raw_audio():
    for split in SPLITS_ALL:
        d = ROOT / f"data/{'dev-clean' if split=='dev' else ('train-clean-100' if split=='train' else 'test-clean')}"
        ensure_dir(d)
    print("✓ raw audio dirs")

def make_processed_features():
    for split in SPLITS_ALL:
        base = ROOT / f"data/processed_{split}"
        features = ensure_dir(base / "features")
        # tiny feature stubs
        write_npy(features / "feat_000.npy", np.zeros((80, 10), dtype=np.float32))
        # ensure expected alias "feature"
        singular = base / "feature"
        maybe_symlink_or_copy(features, singular)
    print("✓ processed features dirs with stub npy + alias")

def make_embeddings_stubs():
    # Put something obvious that your generate_embeddings() would later replace
    for model in MODELS:
        for split in SPLITS_ALL:
            base = ROOT / f"embeddings_{model}_{split}"
            write_npy(base / "speaker_0001" / "seg_0001.npy", np.random.randn(128).astype(np.float32))
    print("✓ embeddings stubs")

def make_enrollment_sets_like_tree():
    # Optional: some pipelines use enrollment-like structures; tiny JSONs help downstream utilities
    for split in SPLITS_DATASETS:
        for fold in FOLDERS:
            for qtype in ("grid", "query", "unknown_query"):
                base = ROOT / "enrollment_sets" / split / f"{qtype}_{fold}"
                write_json(base / f"{GRID_NAME}.json", {
                    "speakers": [
                        {"speaker_id": "19", "files": ["19-198-0000_chunk0.npy"]},
                        {"speaker_id": "24", "files": ["24-100-0001_chunk0.npy"]},
                    ]
                })
    print("✓ enrollment_sets/* created (minimal)")

def make_lambdamart_merged():
    """
    Create both filename variants you used historically:
      1) output/lambdamart_merged1/10s_10i/lambdamart_merged_10s_10i.csv
      2) output/lambdamart_merged1/10s_10i/lambdamart_merged_1.csv
    with minimal columns (config + a couple numeric features + label).
    """
    for model in MODELS:
        for fold in FOLDERS:
            outdir = ROOT / f"output/lambdamart_merged{fold}" / GRID_NAME
            df = pd.DataFrame({
                "config": [GRID_NAME]*4,
                "query_id": [f"Q_{i}" for i in range(4)],
                "candidate_id": [f"C_{i}" for i in range(4)],
                "label": [1, 0, 1, 0],
                "dist_mean": np.random.rand(4),
                "dist_min": np.random.rand(4),
                "centroid_dist": np.random.rand(4),
                "model_type": [model]*4,
            })
            write_csv(outdir / f"lambdamart_merged_{GRID_NAME}.csv", df)
            write_csv(outdir / f"lambdamart_merged_{fold}.csv", df)
    print("✓ LambdaMART merged datasets (both naming styles)")

def make_confidences_stubs():
    """
    Confidence outputs typically used for thresholding or final combine.
    We'll create per model/split/query_type single CSV per GRID_NAME.
    Columns: config, query_id, candidate_id, score (and optional label if available).
    """
    for model in MODELS:
        for split in SPLITS_DATASETS:
            for qtype in QUERY_TYPES:
                df = pd.DataFrame({
                    "config": [GRID_NAME]*6,
                    "query_id": [f"{qtype[:1]}Q_{i}" for i in range(6)],
                    "candidate_id": [f"C_{i%3}" for i in range(6)],
                    "score": np.random.rand(6),
                    # optional ground truth marker for known queries
                    "is_known": [1 if qtype=="queries" else 0]*6,
                })
                out = ROOT / f"confidences_{model}_{split}" / qtype / GRID_NAME
                write_csv(out / f"{GRID_NAME}.csv", df)
    print("✓ confidences stubs")

def make_thresholds_stubs():
    """
    Train-derived thresholds for each model, one per config.
    """
    for model in MODELS:
        df = pd.DataFrame([{
            "config": GRID_NAME,
            "threshold": 0.5,
            "metric": "precision@cutoff",
        }])
        write_csv(ROOT / f"thresholds_{model}" / "thresholds_train.csv", df)
    print("✓ thresholds (train) stubs")

def make_final_ensemble_and_analysis_stubs():
    """
    Create final ensemble results and simple error analysis inputs so step 6 can read.
    """
    for model in MODELS:
        final_dir = ROOT / "ensemble_final" / "test" / model
        final_df = pd.DataFrame({
            "config": [GRID_NAME]*4,
            "query_id": [f"Q_{i}" for i in range(4)],
            "true_label": [1, 0, 1, 0],
            "predicted": [1, 0, 0, 1],
            "confidence": np.random.rand(4),
        })
        write_csv(final_dir / "final_results.csv", final_df)

    # a top-level “all models” summary that error analysis might aggregate
    summary = pd.DataFrame({
        "model_type": ["spectral", "phonetic"],
        "macro_precision": [0.82, 0.61],
        "macro_recall": [0.77, 0.58],
    })
    write_csv(ROOT / "ensemble_final" / "test" / "summary.csv", summary)
    print("✓ final ensemble + error-analysis stubs")

def main():
    make_raw_audio()
    make_processed_features()
    make_embeddings_stubs()
    make_enrollment_sets_like_tree()
    make_lambdamart_merged()
    make_confidences_stubs()
    make_thresholds_stubs()
    make_final_ensemble_and_analysis_stubs()
    print("\n✅ Synthetic ensemble artifacts created.")

if __name__ == "__main__":
    main()
