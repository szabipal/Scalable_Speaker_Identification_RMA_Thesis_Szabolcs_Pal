#!/usr/bin/env python3
"""
Compute normalized top-5 entropy for Siamese verification outputs across:
- splits: train, test
- types:  queries (known), unknown_queries (unknown)
- grids:  grid_1..grid_N (auto-detected)

Inputs (produced by your Siamese eval runner):
  siamese_eval_results/
    train/
      queries/
        grid_1/*.csv
        ...
      unknown_queries/
        grid_1/*.csv
        ...
    test/
      queries/
      unknown_queries/

Outputs:
  entropy_results_transformers/
    train/
      entropy_transformers_train_known.csv
      entropy_transformers_train_unknown.csv
    test/
      entropy_transformers_test_known.csv
      entropy_transformers_test_unknown.csv
"""

import os
import sys
from pathlib import Path

# Project root (adjust if needed)
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# --- import your existing functions without modifying them ---
# Put your entropy code into a module file (e.g., entropy_calc.py)
# containing `process_match_file` and `process_folders` exactly as you shared.
from entropy_calc import process_folders  # <- uses your code verbatim


EVAL_ROOT = ROOT_DIR / "siamese_eval_results"
OUT_ROOT  = ROOT_DIR / "entropy_results_transformers"

COMBOS = [
    # (split, query_type_folder, thresholded_label_for_entropy, output_filename)
    ("train", "queries",          "known",   "entropy_transformers_train_known.csv"),
    ("train", "unknown_queries",  "unknown", "entropy_transformers_train_unknown.csv"),
    ("test",  "queries",          "known",   "entropy_transformers_test_known.csv"),
    ("test",  "unknown_queries",  "unknown", "entropy_transformers_test_unknown.csv"),
]


def find_grid_dirs(base: Path):
    """Return all grid_* directories that contain CSV files."""
    if not base.exists():
        return []
    grid_dirs = []
    for grid_dir in sorted(base.glob("grid_*")):
        if grid_dir.is_dir() and any(grid_dir.glob("*.csv")):
            grid_dirs.append(grid_dir)
    return grid_dirs


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for split, qtype, lbl, out_name in COMBOS:
        base = EVAL_ROOT / split / qtype
        grid_dirs = find_grid_dirs(base)

        if not grid_dirs:
            print(f"[WARN] No CSVs found for {split}/{qtype} under {base}. Skipping.")
            continue

        out_dir = OUT_ROOT / split
        out_dir.mkdir(parents=True, exist_ok=True)
        output_csv = out_dir / out_name

        print(f"\n▶ Entropy: split={split} | type={qtype} | label={lbl}")
        print(f"   Grids:  {', '.join(d.name for d in grid_dirs)}")
        print(f"   Output: {output_csv}")

        # Your function merges all provided folders into a single CSV
        process_folders(
            folder_paths=[str(d) for d in grid_dirs],
            output_csv=str(output_csv),
            thresholded_label=lbl
        )

    print("\n✅ Done. Entropy CSVs saved under:", OUT_ROOT)


if __name__ == "__main__":
    main()
