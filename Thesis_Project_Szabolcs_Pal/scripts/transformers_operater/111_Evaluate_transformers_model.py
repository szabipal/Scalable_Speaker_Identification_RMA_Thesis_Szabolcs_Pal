#!/usr/bin/env python3
"""
Run Siamese verification on BOTH train and test grids,
for both known queries and unknown queries.

Results are saved separately under:
  siamese_eval_results/train/queries/grid_<i>/
  siamese_eval_results/train/unknown_queries/grid_<i>/
  siamese_eval_results/test/queries/grid_<i>/
  siamese_eval_results/test/unknown_queries/grid_<i>/

Relies on CSV embeddings laid out like:
  hubert_embeddings_<split>/queries/grid_<i>/*.csv
  hubert_embeddings_<split>/unknown_queries/grid_<i>/*.csv
  hubert_embeddings_<split>/grids/grid_<i>/*.csv
"""

import os
import sys
from pathlib import Path

# Add project root
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import torch
from evaluate_siamese_model import verify_speakers_with_folders
from models.siamese_nn import SiameseMLP


# -------- Config (edit as needed) --------
MODEL_PATH = ROOT_DIR / "saved_siamese/siamese_mlp_final.pt"
EMB_ROOTS = {
    "train": {
        "queries": ROOT_DIR / "hubert_embeddings_train" / "queries",
        "unknown_queries": ROOT_DIR / "hubert_embeddings_train" / "unknown_queries",
        "grids": ROOT_DIR / "hubert_embeddings_train" / "grids",
    },
    "test": {
        "queries": ROOT_DIR / "hubert_embeddings_test" / "queries",
        "unknown_queries": ROOT_DIR / "hubert_embeddings_test" / "unknown_queries",
        "grids": ROOT_DIR / "hubert_embeddings_test" / "grids",
    },
}
OUTPUT_ROOT = ROOT_DIR / "siamese_eval_results"
NUM_GRIDS = 5           # grid_1..grid_5
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIM = 768
HIDDEN_DIM = 256
# ------------------------------------------


def main():
    # Load model once
    model = SiameseMLP(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    for split, roots in EMB_ROOTS.items():
        grids_root = roots["grids"]

        if not grids_root.exists():
            print(f"[WARN] Skipping split '{split}' – missing grids folder: {grids_root}")
            continue

        for query_type in ["queries", "unknown_queries"]:
            queries_root = roots[query_type]

            if not queries_root.exists():
                print(f"[WARN] Skipping {split} {query_type} — missing folder: {queries_root}")
                continue

            for i in range(1, NUM_GRIDS + 1):
                query_dir = queries_root / f"grid_{i}"
                enrolled_dir = grids_root / f"grid_{i}"
                out_dir = OUTPUT_ROOT / split / query_type / f"grid_{i}"

                if not query_dir.exists() or not enrolled_dir.exists():
                    print(f"[WARN] Skipping {split} {query_type} grid_{i} — missing:\n"
                          f"  query_dir:    {query_dir}\n  enrolled_dir: {enrolled_dir}")
                    continue

                print(f"\n▶ Split={split} | {query_type} | grid_{i}\n"
                      f"   queries:  {query_dir}\n"
                      f"   enrolled: {enrolled_dir}\n"
                      f"   output:   {out_dir}")

                verify_speakers_with_folders(
                    model=model,
                    query_folder=str(query_dir),
                    enrolled_folder=str(enrolled_dir),
                    output_dir=str(out_dir),
                    threshold=THRESHOLD,
                    device=DEVICE,
                )

    print("\n✅ Done: saved summaries under", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
