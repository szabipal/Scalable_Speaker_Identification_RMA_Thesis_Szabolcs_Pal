# full_ensemble_pipeline.py

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import shutil

# 0) Preprocess + features
from scripts.ensemble_operater._Preprocessor_Feat_Extractor_class import PreprocessingAndFeatureExtractor

# 1) Embeddings
from scripts.ensemble_operater._Ensemble_embedding_extractor import generate_embeddings

# 2) LambdaMART datasets
from scripts.ensemble_operater._Generate_lambdamart_dataset import (
    build_datasets_for_model_split as build_datasets
)
# If the split-based version doesn’t exist, fallback:
# from 000_Generate_lambdamart_dataset import build_datasets_for_model as build_datasets

# 3) LambdaMART training
from scripts.ensemble_operater._Train_Lambdamart_model import train_lambdamart_model

# 4) Inference → confidences
from scripts.ensemble_operater._Infer_Lambdamart_model import batch_generate_confidences

# 5) Thresholding
from scripts.ensemble_operater._Find_Thresholds_Ensemble import run_thresholding_for_model

# 6) Final ensemble + error analysis
from scripts.ensemble_operater._Final_testing_ensemble_model import run_final_ensemble_test
from scripts.ensemble_operater._Final_testing_ensemble_model import run_error_analysis_for_all_models

# -----------------------------
# Config
# -----------------------------
MODELS = ("spectral", "phonetic")
SPLITS_ALL = ("dev", "train", "test")
SPLITS_DATASETS = ("train", "test")  # datasets for LM need train/test (dev used for quantiles)
QUERY_TYPES = ("queries", "unknown_queries")  # known vs unknown

RAW_AUDIO = dict(
    dev="data/dev-clean",
    train="data/train-clean-100",
    test="data/test-clean",
)
PROCESSED_BASE = {s: f"data/processed_{s}" for s in SPLITS_ALL}

RUN = dict(
    preprocess_and_extract=True,
    generate_embeddings=True,
    build_lambdamart_datasets=True,
    train_lambdamart=True,
    infer_confidences=True,
    thresholding=True,
    final_ensemble=True,
    error_analysis=True,
)

# Train LM once per model, using TRAIN split, known queries
TRAIN_ARGS = {
    "spectral": dict(split="train", query_type="queries", train_folders=(1, 2, 3, 4), eval_folder=5),
    "phonetic": dict(split="train", query_type="queries", train_folders=(1, 2, 3, 4), eval_folder=5),
}
INFER_CUTOFF = 0.1532
THRESHOLD_VALUE = 0.7
THRESHOLD_SPLIT = "test"   # run once per model, on test

# -----------------------------
# Helpers
# -----------------------------
def _ensure_feature_alias(processed_base_dir: Path):
    """Mirror .../features -> .../feature if needed (downstream expects singular)."""
    plural = processed_base_dir / "features"
    singular = processed_base_dir / "feature"
    if plural.exists() and not singular.exists():
        try:
            singular.symlink_to(plural, target_is_directory=True)
            print(f"🔗 Symlinked: {singular} -> {plural}")
        except Exception:
            print(f"📁 Copying 'features' to 'feature' (no symlink perms).")
            shutil.copytree(plural, singular)

def _run_preprocess_and_extract(split: str, model_type: str):
    cfg = {
        "raw_audio_dir": RAW_AUDIO[split],
        "output_base_dir": PROCESSED_BASE[split],
        "model_type": model_type,
    }
    pipe = PreprocessingAndFeatureExtractor(cfg)
    pipe.run_pipeline()
    _ensure_feature_alias(Path(PROCESSED_BASE[split]))

def _build_datasets_for(model_type: str):
    """
    Ensure both query types and both splits are built.
    If your datasets module exposes build_datasets_for_model_split(model, split, query_type),
    we call that. Otherwise, build_datasets_for_model(model) should already do both.
    """
    try:
        # If we imported build_datasets_for_model_split as build_datasets:
        for split in SPLITS_DATASETS:
            for q in QUERY_TYPES:
                print(f"   → Build datasets: model={model_type}, split={split}, query_type={q}")
                build_datasets(model_type, split, q)
    except TypeError:
        # Fallback to older single-call function that handles everything internally
        print(f"   → Build datasets (single call): model={model_type}")
        build_datasets(model_type)

# -----------------------------
# Pipeline
# -----------------------------
def main():
    # 0) Preprocess + Feature extraction for DEV/TRAIN/TEST, for both models
    if RUN["preprocess_and_extract"]:
        for split in SPLITS_ALL:
            for model in MODELS:
                print(f"\n=== [STEP 0] Preprocess+Extract → split={split}, model={model} ===")
                _run_preprocess_and_extract(split, model)

    # 1) Embeddings for DEV/TRAIN/TEST, for both models
    if RUN["generate_embeddings"]:
        for model in MODELS:
            for split in SPLITS_ALL:
                print(f"\n=== [STEP 1] Embeddings → model={model}, split={split} ===")
                generate_embeddings(model_type=model, dataset_type=split)

    # 2) Build LambdaMART datasets for (train|test) × (queries|unknown_queries), per model
    if RUN["build_lambdamart_datasets"]:
        for model in MODELS:
            print(f"\n=== [STEP 2] Build LambdaMART datasets → model={model} ===")
            _build_datasets_for(model)

    # 3) Train LambdaMART once per model on TRAIN known queries
    if RUN["train_lambdamart"]:
        for model in MODELS:
            print(f"\n=== [STEP 3] Train LambdaMART → model={model} ===")
            args = TRAIN_ARGS[model]
            train_lambdamart_model(
                model_type=model,
                split=args["split"],                 # 'train'
                query_type=args["query_type"],       # 'queries' (known)
                train_folders=args["train_folders"], # (1,2,3,4)
                eval_folder=args["eval_folder"],     # 5
            )

    # 4) Inference → confidences for (train|test) × (queries|unknown_queries), both models
    if RUN["infer_confidences"]:
        print("\n=== [STEP 4] Inference → confidences (all models/splits/query_types) ===")
        batch_generate_confidences(cutoff_value=INFER_CUTOFF)

    # 5) Thresholding once per model (on test)
    if RUN["thresholding"]:
        for model in MODELS:
            print(f"\n=== [STEP 5] Thresholding → model={model}, split={THRESHOLD_SPLIT} ===")
            run_thresholding_for_model(model_type=model, dataset_type=THRESHOLD_SPLIT, threshold_value=THRESHOLD_VALUE)

    # 6a) Final ensemble once per model (on test)
    if RUN["final_ensemble"]:
        for model in MODELS:
            print(f"\n=== [STEP 6a] Final ensemble → model={model} (test) ===")
            run_final_ensemble_test(model_type=model, dataset_type="test")

    # 6b) Error analysis (reads final_results.csv from step 6a)
    if RUN["error_analysis"]:
        print("\n=== [STEP 6b] Error analysis (all models, test) ===")
        run_error_analysis_for_all_models(dataset_type="test")

    print("\n✅ Pipeline finished.")

if __name__ == "__main__":
    main()
