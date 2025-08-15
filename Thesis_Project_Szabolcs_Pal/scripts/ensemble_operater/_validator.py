#!/usr/bin/env python3
"""
Lightweight validator for the full_ensemble_pipeline wiring.

It does NOT run the pipeline. It checks:
- imports (PreprocessingAndFeatureExtractor, generate_embeddings, dataset builder, train/infer/thresh/final fns)
- presence of key directories/files (created by the synthetic script)
- minimal CSV schemas where practical

Exit codes:
  0: OK
  1: blocking issues found
  2: validator crashed
"""
import sys, traceback
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))      
import pandas as pd
ISSUES = []

def check_imports():
    
    def ok(m): print("✅", m)
    def err(m): print("❌", m); ISSUES.append(m)

    # ---- import checks: use your new module names ----
    try:
        from scripts.ensemble_operater._Preprocessor_Feat_Extractor_class import PreprocessingAndFeatureExtractor
        ok("Import: _Preprocessor_Feat_Extractor_class.PreprocessingAndFeatureExtractor")
    except Exception as e:
        err(f"Import failed: _Preprocessor_Feat_Extractor_class -> {e}")

    try:
        from scripts.ensemble_operater._Ensemble_embedding_extractor import generate_embeddings
        ok("Import: _Ensemble_embedding_extractor.generate_embeddings")
    except Exception as e:
        err(f"Import failed: _Ensemble_embedding_extractor -> {e}")

    # dataset builders (split-aware preferred, fallback to monolithic)
    try:
        from scripts.ensemble_operater._Generate_lambdamart_dataset import build_datasets_for_model_split as build_datasets
        ok("Import: _Generate_lambdamart_dataset.build_datasets_for_model_split")
    except Exception as e1:
        try:
            from scripts.ensemble_operater._Generate_lambdamart_dataset import build_datasets_for_model as build_datasets
            ok("Import: _Generate_lambdamart_dataset.build_datasets_for_model (fallback)")
        except Exception as e2:
            err(f"Import failed: _Generate_lambdamart_dataset -> {e1} | {e2}")

    try:
        from scripts.ensemble_operater._Train_Lambdamart_model import train_lambdamart_model
        ok("Import: _Train_Lambdamart_model.train_lambdamart_model")
    except Exception as e:
        err(f"Import failed: _Train_Lambdamart_model -> {e}")

    try:
        from scripts.ensemble_operater._Infer_Lambdamart_model import batch_generate_confidences
        ok("Import: _Infer_Lambdamart_model.batch_generate_confidences")
    except Exception as e:
        err(f"Import failed: _Infer_Lambdamart_model -> {e}")

    try:
        from scripts.ensemble_operater._Find_Thresholds_Ensemble import run_thresholding_for_model
        ok("Import: _Find_Thresholds_Ensemble.run_thresholding_for_model")
    except Exception as e:
        err(f"Import failed: _Find_Thresholds_Ensemble -> {e}")

    try:
        from scripts.ensemble_operater._Final_testing_ensemble_model import (
            run_final_ensemble_test,
            run_error_analysis_for_all_models,
        )
        ok("Import: _Final_testing_ensemble_model.(final+error)")
    except Exception as e:
        err(f"Import failed: _Final_testing_ensemble_model -> {e}")

def ok(m): print("✅", m)

def check_exists(p: Path, label: str, required=True):
    if p.exists():
        ok(f"{label}: {p}")
        return True
    (err if required else warn)(f"Missing {label}: {p}")
    return False

def check_csv_columns(p: Path, must_have: list, label: str):
    try:
        df = pd.read_csv(p, nrows=5)
        missing = [c for c in must_have if c not in df.columns]
        if missing:
            err(f"{label} missing columns {missing}: {p}")
        else:
            ok(f"{label} schema OK: {p.name}")
    except Exception as e:
        err(f"{label} not readable: {p} -> {e}")

def check_synthetic_tree():
    # processed features
    for split in ("dev","train","test"):
        check_exists(ROOT / f"data/processed_{split}/features", f"processed features ({split})")
        check_exists(ROOT / f"data/processed_{split}/feature", f"feature alias ({split})", required=False)

    # embeddings stubs
    for model in ("spectral","phonetic"):
        for split in ("dev","train","test"):
            check_exists(ROOT / f"embeddings_{model}_{split}", f"embeddings stub ({model},{split})", required=False)

    # merged datasets (both styles)
    any_merged = False
    for fold in (1,2,3,4,5):
        base = ROOT / f"output/lambdamart_merged{fold}/10s_10i"
        p1 = base / "lambdamart_merged_10s_10i.csv"
        p2 = base / f"lambdamart_merged_{fold}.csv"
        if p1.exists() or p2.exists():
            any_merged = True
            check_csv_columns(p1 if p1.exists() else p2,
                              ["config","query_id","candidate_id","label"],
                              f"merged dataset (fold {fold})")
    if not any_merged:
        warn("No merged datasets found (expected from synthetic generator).")

    # confidences stubs
    for model in ("spectral","phonetic"):
        for split in ("train","test"):
            for q in ("queries","unknown_queries"):
                p = ROOT / f"confidences_{model}_{split}" / q / "10s_10i" / "10s_10i.csv"
                if p.exists():
                    check_csv_columns(p, ["config","query_id","candidate_id","score"], f"confidences ({model},{split},{q})")
                else:
                    warn(f"Missing confidences ({model},{split},{q}) at {p}")

    # thresholds (train)
    for model in ("spectral","phonetic"):
        p = ROOT / f"thresholds_{model}" / "thresholds_train.csv"
        if p.exists():
            check_csv_columns(p, ["config","threshold"], f"thresholds_train ({model})")
        else:
            warn(f"Missing thresholds_train for {model} at {p}")

    # final ensemble results + summary
    for model in ("spectral","phonetic"):
        p = ROOT / "ensemble_final" / "test" / model / "final_results.csv"
        if p.exists():
            check_csv_columns(p, ["config","query_id","true_label","predicted","confidence"], f"final_results ({model})")
        else:
            warn(f"Missing final_results ({model}) at {p}")

    summary = ROOT / "ensemble_final" / "test" / "summary.csv"
    if summary.exists():
        check_csv_columns(summary, ["model_type","macro_precision","macro_recall"], "ensemble summary")
    else:
        warn(f"Missing ensemble summary at {summary}")

def main():
    print("=== ENSEMBLE PIPELINE VALIDATOR ===")
    check_imports()
    check_synthetic_tree()

    print("\n=== SUMMARY ===")
    if ISSUES:
        print(f"❌ Found {len(ISSUES)} blocking issue(s):")
        for i, m in enumerate(ISSUES, 1):
            print(f"  {i}. {m}")
        sys.exit(1)
    else:
        print("✅ No blocking issues detected. Wiring & schemas look OK.")
        sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("❌ Validator crashed:\n" + traceback.format_exc())
        sys.exit(2)
