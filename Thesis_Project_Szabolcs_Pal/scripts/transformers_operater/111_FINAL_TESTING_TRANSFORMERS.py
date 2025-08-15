#!/usr/bin/env python3
"""
Apply TRAIN-derived thresholds to TEST entropy files for the transformers model.
This DOES NOT recompute thresholds on test; it reuses train thresholds.

Inputs expected:
  entropy_results_transformers/
    test/entropy_transformers_test_known.csv
    test/entropy_transformers_test_unknown.csv

  thresholds_transformers/
    transformers_thresholds_train.csv   # <- train thresholds

Outputs:
  transformers_final_eval/
    test_using_train_thresholds/
      ... (heatmaps, CSVs)
"""

import os
import sys
from pathlib import Path

# Add project root
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Your existing final evaluation function (unchanged)
from final_entropy_eval import run_final_entropy_test

TEST_KNOWN   = ROOT_DIR / "entropy_results_transformers" / "test" / "entropy_transformers_test_known.csv"
TEST_UNKNOWN = ROOT_DIR / "entropy_results_transformers" / "test" / "entropy_transformers_test_unknown.csv"
TRAIN_THR    = ROOT_DIR / "thresholds_transformers" / "transformers_thresholds_train.csv"  # reuse train thresholds
OUT_DIR      = ROOT_DIR / "transformers_final_eval" / "test_using_train_thresholds"

def main():
    missing = [p for p in [TEST_KNOWN, TEST_UNKNOWN, TRAIN_THR] if not p.exists()]
    if missing:
        print("[ERROR] Missing required files:")
        for m in missing:
            print("  -", m)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n▶ Final TEST evaluation using TRAIN thresholds")
    print(f"   test known:   {TEST_KNOWN}")
    print(f"   test unknown: {TEST_UNKNOWN}")
    print(f"   train thr:    {TRAIN_THR}")
    print(f"   out:          {OUT_DIR}")

    run_final_entropy_test(
        known_file=str(TEST_KNOWN),
        unknown_file=str(TEST_UNKNOWN),
        threshold_csv=str(TRAIN_THR),       # <-- the key: use TRAIN thresholds
        outdir=str(OUT_DIR),
    )

    print("\n✅ Done. Results saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
