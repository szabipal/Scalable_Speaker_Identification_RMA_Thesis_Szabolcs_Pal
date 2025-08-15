#!/usr/bin/env python3
"""
Run entropy-based threshold selection for all splits (train/test) and types (known/unknown)
and write split-specific threshold CSVs.

Inputs (from your entropy runner):
  entropy_results_transformers/
    train/
      entropy_transformers_train_known.csv
      entropy_transformers_train_unknown.csv
    test/
      entropy_transformers_test_known.csv
      entropy_transformers_test_unknown.csv

Outputs:
  thresholds_transformers/
    transformers_thresholds_train.csv
    transformers_thresholds_test.csv
"""

import os
import sys
from pathlib import Path

# Add project root
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Import your existing thresholding function (unchanged)
from thresholding_transformers import threshold_transformers_single_file_v2

ENTROPY_ROOT = ROOT_DIR / "entropy_results_transformers"
OUT_ROOT = ROOT_DIR / "thresholds_transformers"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

COMBOS = {
    "train": {
        "known":   ENTROPY_ROOT / "train" / "entropy_transformers_train_known.csv",
        "unknown": ENTROPY_ROOT / "train" / "entropy_transformers_train_unknown.csv",
        "out":     OUT_ROOT / "transformers_thresholds_train.csv",
    }
    # "test": {
    #     "known":   ENTROPY_ROOT / "test" / "entropy_transformers_test_known.csv",
    #     "unknown": ENTROPY_ROOT / "test" / "entropy_transformers_test_unknown.csv",
    #     "out":     OUT_ROOT / "transformers_thresholds_test.csv",
    # },
}

def main():
    for split, paths in COMBOS.items():
        known_csv = paths["known"]
        unknown_csv = paths["unknown"]
        out_csv = paths["out"]

        if not known_csv.exists() or not unknown_csv.exists():
            print(f"[WARN] Skipping {split}: missing files:\n  {known_csv}\n  {unknown_csv}")
            continue

        print(f"\n▶ Running threshold search for split={split}")
        print(f"   known:   {known_csv}")
        print(f"   unknown: {unknown_csv}")
        print(f"   out:     {out_csv}")

        threshold_transformers_single_file_v2(
            known_csv=str(known_csv),
            unknown_csv=str(unknown_csv),
            out_csv=str(out_csv),
        )

    print("\n✅ Done. Threshold CSVs saved under:", OUT_ROOT)

if __name__ == "__main__":
    main()
