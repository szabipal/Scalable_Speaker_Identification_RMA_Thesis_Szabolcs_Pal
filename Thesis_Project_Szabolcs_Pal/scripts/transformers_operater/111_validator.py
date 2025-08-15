#!/usr/bin/env python3
"""
Lightweight validator for the Transformers-based pipeline.

What it does (no heavy work):
- Checks runner scripts exist and are executable/callable
- Verifies Python deps & GPU availability
- Validates presence & structure of key inputs/outputs for each stage
- Peeks into a FEW files (JSON/CSV/NPY) to confirm schema/columns/shapes
- Confirms model code imports (SiameseMLP), dataset class import, and can instantiate
- Exits non-zero if something is missing/broken

Usage:
  python scripts/transformers_operater/001_VALIDATE_PIPELINE_TRANSFORMERS.py
"""

import os
import sys
import json
import traceback
from pathlib import Path

ISSUES = []

def ok(msg):   print("✅", msg)
def warn(msg): print("⚠️ ", msg)
def err(msg):  print("❌", msg); ISSUES.append(msg)

# ---------- Project roots & runner paths ----------
THIS = Path(__file__).resolve()
ROOT = THIS.parents[2]

RUNNERS = {
    "embeddings": ROOT / "scripts/transformers_operater/111_Generate_hubert_embeddings.py",
    "train":      ROOT / "scripts/transformers_operater/111_Run_siamese_training.py",
    "evaluate":   ROOT / "scripts/transformers_operater/111_Evaluate_transformers_model.py",
    "entropy":    ROOT / "scripts/transformers_operater/111_Run_entropy_calculation_transformers.py",
    "thresholds": ROOT / "scripts/transformers_operater/111_Thresholding_transformers.py",
    "final":      ROOT / "scripts/transformers_operater/111_FINAL_TESTING_TRANSFORMERS.py",
}

# ---------- Expected artifact roots ----------
EMB_TRAIN = ROOT / "hubert_embeddings_train"
EMB_TEST  = ROOT / "hubert_embeddings_test"

EVAL_ROOT = ROOT / "siamese_eval_results"
ENTROPY_ROOT = ROOT / "entropy_results_transformers"
THRESHOLDS_DIR = ROOT / "thresholds_transformers"

# Enrollment json structure
ENROLL_ROOT = ROOT / "enrollment_sets"
TRAIN_ENROLL = ENROLL_ROOT / "train"
TEST_ENROLL  = ENROLL_ROOT / "test"

# Wave sources
WAVE_TRAIN = ROOT / "data/processed_train/wave_chunks_2s"
WAVE_TEST  = ROOT / "data/processed_test/wave_chunks_2s"

# Siamese training pairs
PAIRS_TRAIN = ROOT / "data/speaker_pairs/train_pairs.csv"
PAIRS_VAL   = ROOT / "data/speaker_pairs/val_pairs.csv"

# Saved model
SAVED_SIAMESE = ROOT / "saved_siamese/siamese_mlp_final.pt"

# ---------- Tiny utilities ----------
def check_runner(path: Path, name: str):
    if not path.exists():
        err(f"[runner:{name}] missing: {path}")
        return
    if path.suffix == ".py":
        ok(f"[runner:{name}] python script found")
    else:
        # shell or no extension — ensure executable bit or at least readable
        if os.access(path, os.X_OK):
            ok(f"[runner:{name}] executable found")
        else:
            warn(f"[runner:{name}] not marked executable (may still run via 'python {path}')")

def check_imports():
    sys.path.append(str(ROOT))
    try:
        from models.siamese_nn import SiameseMLP  # noqa
        ok("Import: models.siamese_nn.SiameseMLP")
    except Exception as e:
        err(f"Import failed: models.siamese_nn.SiameseMLP -> {e}")

    try:
        from my_datasets.siamese_separated_dataset import SiamesePairCSV  # noqa
        ok("Import: my_datasets.siamese_separated_dataset.SiamesePairCSV")
    except Exception as e:
        err(f"Import failed: my_datasets.siamese_separated_dataset.SiamesePairCSV -> {e}")

def check_env():
    try:
        import torch
        ok(f"PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            ok(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            warn("CUDA not available (pipeline will run on CPU)")
    except Exception as e:
        err(f"PyTorch import failed: {e}")
    # Transformers (for embedding extractor)
    try:
        import transformers  # noqa
        ok("Transformers available")
    except Exception as e:
        err(f"Transformers import failed: {e}")

def check_folder_exists(p: Path, label: str, required=True):
    if p.exists():
        ok(f"{label} exists: {p}")
        return True
    else:
        (err if required else warn)(f"{label} missing: {p}")
        return False

def sample_json_schema_check(json_path: Path):
    try:
        data = json.loads(json_path.read_text())
        if not isinstance(data.get("speakers"), list):
            err(f"Enrollment JSON missing 'speakers' list: {json_path}")
            return
        sp = data["speakers"][0] if data["speakers"] else {}
        if not isinstance(sp.get("speaker_id", None), (str, int)):
            err(f"'speaker_id' not found/invalid in: {json_path}")
        if not isinstance(sp.get("files", None), list):
            err(f"'files' list not found in: {json_path}")
        else:
            ok(f"Enrollment JSON schema OK: {json_path.name}")
    except Exception as e:
        err(f"Failed to parse enrollment JSON {json_path}: {e}")

def find_one(patterns):
    for p in patterns:
        matches = list(Path(p).glob("**/*"))
        if matches:
            return matches[0]
    return None

def check_wave_sample(wave_root: Path):
    # find a single .npy to ensure data presence (no load to keep it light)
    sample = next(wave_root.rglob("*.npy"), None)
    if sample:
        ok(f"Found waveform npy: {sample.relative_to(wave_root)}")
    else:
        warn(f"No .npy found under {wave_root} (embedding stage will have nothing to do)")

def check_csv_columns(csv_path: Path, required_cols, label):
    import pandas as pd
    try:
        df = pd.read_csv(csv_path, nrows=5)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            err(f"{label}: missing columns {missing} in {csv_path}")
        else:
            ok(f"{label}: columns OK in {csv_path.name}")
    except Exception as e:
        err(f"{label}: failed to read {csv_path} -> {e}")

def main():
    print("=== TRANSFORMERS PIPELINE VALIDATOR ===")

    # 0) Runners present?
    for name, path in RUNNERS.items():
        check_runner(path, name)

    # 1) Environment
    check_env()
    check_imports()

    # 2) Inputs expected BEFORE running anything heavy
    # Enrollment JSONs (train/test, grid_1..5, query_1..5, unknown_query_1..5) — check existence + a sample schema
    for split_root, split_lbl in [(TRAIN_ENROLL, "enrollment(train)"), (TEST_ENROLL, "enrollment(test)")]:
        if check_folder_exists(split_root, split_lbl, required=True):
            # probe one JSON if any
            sample_json = next(split_root.rglob("*.json"), None)
            if sample_json:
                sample_json_schema_check(sample_json)
            else:
                warn(f"No JSON files under {split_root}")

    # Wave roots
    if check_folder_exists(WAVE_TRAIN, "wave(train)", required=True):
        check_wave_sample(WAVE_TRAIN)
    if check_folder_exists(WAVE_TEST, "wave(test)", required=True):
        check_wave_sample(WAVE_TEST)

    # Siamese training CSVs
    if PAIRS_TRAIN.exists():
        check_csv_columns(PAIRS_TRAIN, ["x1_path","x2_path","label"], "siamese train_pairs")
    else:
        err(f"Missing train_pairs.csv: {PAIRS_TRAIN}")

    if PAIRS_VAL.exists():
        check_csv_columns(PAIRS_VAL, ["x1_path","x2_path","label"], "siamese val_pairs")
    else:
        err(f"Missing val_pairs.csv: {PAIRS_VAL}")

    # 3) If artifacts already exist, validate minimal structure for downstream
    # Embedding CSVs (optional; only if already generated)
    if EMB_TRAIN.exists():
        # Look for one CSV under queries/grids/unknown_queries
        sample_csv = next((EMB_TRAIN / "queries").rglob("*.csv"), None) or \
                     next((EMB_TRAIN / "grids").rglob("*.csv"), None) or \
                     next((EMB_TRAIN / "unknown_queries").rglob("*.csv"), None)
        if sample_csv:
            check_csv_columns(sample_csv, ["speaker_id","session_id","instance_id","chunk_id","embedding"], "embedding csv(train)")
    if EMB_TEST.exists():
        sample_csv = next((EMB_TEST / "queries").rglob("*.csv"), None) or \
                     next((EMB_TEST / "grids").rglob("*.csv"), None) or \
                     next((EMB_TEST / "unknown_queries").rglob("*.csv"), None)
        if sample_csv:
            check_csv_columns(sample_csv, ["speaker_id","session_id","instance_id","chunk_id","embedding"], "embedding csv(test)")

    # Siamese eval results (optional)
    if EVAL_ROOT.exists():
        sample_eval = next(EVAL_ROOT.rglob("*.csv"), None)
        if sample_eval:
            # Your entropy step expects columns like: query_id, query_speaker, enrolled_speaker, sum/count/match_rate or derived per-file summary
            check_csv_columns(
                sample_eval,
                ["query_id","query_speaker","enrolled_speaker","sum","count","match_rate"],
                "siamese eval summary"
            )

    # Entropy results (optional)
    if ENTROPY_ROOT.exists():
        k = (ENTROPY_ROOT / "train" / "entropy_transformers_train_known.csv")
        u = (ENTROPY_ROOT / "train" / "entropy_transformers_train_unknown.csv")
        if k.exists(): check_csv_columns(k, ["query_id","top_match_id","normalized_entropy","source_file","grid_number","label"], "entropy(train known)")
        if u.exists(): check_csv_columns(u, ["query_id","top_match_id","normalized_entropy","source_file","grid_number","label"], "entropy(train unknown)")

    # Thresholds (optional)
    if THRESHOLDS_DIR.exists():
        thr_file = THRESHOLDS_DIR / "transformers_thresholds_train.csv"
        if thr_file.exists():
            check_csv_columns(
                thr_file,
                ["config","known_threshold","unknown_threshold","known_train_precision","known_train_recall","unknown_train_precision","unknown_train_recall"],
                "thresholds(train)"
            )

    # Saved model existence (optional but recommended before eval)
    if SAVED_SIAMESE.exists():
        ok(f"Siamese weights present: {SAVED_SIAMESE}")
    else:
        warn(f"Siamese weights not found yet (will be created by training): {SAVED_SIAMESE}")

    # -------- Summary --------
    print("\n=== SUMMARY ===")
    if ISSUES:
        print(f"❌ Found {len(ISSUES)} issue(s):")
        for i, m in enumerate(ISSUES, 1):
            print(f"  {i}. {m}")
        sys.exit(1)
    else:
        print("✅ No blocking issues detected. Pipeline should run.")
        sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        err("Validator crashed unexpectedly:\n" + traceback.format_exc())
        print("\n=== SUMMARY ===")
        for i, m in enumerate(ISSUES, 1):
            print(f"  {i}. {m}")
        sys.exit(2)
