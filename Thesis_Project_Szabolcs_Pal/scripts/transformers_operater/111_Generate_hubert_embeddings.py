# #!/usr/bin/env python3
# """
# Orchestrates HuBERT embedding extraction by repeatedly EXECUTING your existing
# extractor script with different settings, without modifying it.

# It will run for:
#   - splits: train, test
#   - groups: grid_1..5, query_1..5, unknown_query_1..5
# Using JSONs under:
#   enrollment_sets/<split>/<group_name>/

# Wave sources assumed:
#   - train -> data/processed_train/wave_chunks_2s
#   - test  -> data/processed_test/wave_chunks_2s

# Outputs go to:
#   hubert_embeddings_<split>/<group_type>/grid_<i>/ (CSV per JSON)
# """
# import os
# import sys
# from pathlib import Path
# import runpy
# from datetime import datetime

# # === Path to your existing extractor script (UNMODIFIED) ===
# # e.g., repository/scripts/hubert_grid_extractor.py
# EXTRACTOR_SCRIPT = Path("scripts/hubert_grid_extractor.py").resolve()

# # === Config you might tweak ===
# SAMPLING_RATE = 16000
# NUM_GROUPS = 5  # grid_1..5 / query_1..5 / unknown_query_1..5

# SPLITS = {
#     "train": {
#         "wave_dir": Path("data/processed_train/wave_chunks_2s"),
#         "output_root": Path("hubert_embeddings_train"),
#     },
#     "test": {
#         "wave_dir": Path("data/processed_test/wave_chunks_2s"),
#         "output_root": Path("hubert_embeddings_test"),
#     }
# }

# # (group_folder_prefix, output_subdir)
# GROUPS = [
#     ("grid", "grids"),
#     ("query", "queries"),
#     ("unknown_query", "unknown_queries"),
# ]

# def main():
#     if not EXTRACTOR_SCRIPT.exists():
#         raise FileNotFoundError(f"Extractor script not found: {EXTRACTOR_SCRIPT}")

#     repo_root = Path(__file__).resolve().parent
#     os.chdir(repo_root)  # consistent relative paths

#     for split, split_cfg in SPLITS.items():
#         wave_dir = split_cfg["wave_dir"]
#         out_root = split_cfg["output_root"]

#         for group_prefix, out_subdir in GROUPS:
#             for i in range(1, NUM_GROUPS + 1):
#                 # enrollment_sets/<split>/<group_prefix>_<i>/
#                 config_base = Path("enrollment_sets") / split / f"{group_prefix}_{i}"
#                 # outputs: hubert_embeddings_<split>/<out_subdir>/grid_<i>/
#                 output_base = out_root / out_subdir / f"grid_{i}"
#                 output_base.mkdir(parents=True, exist_ok=True)

#                 # Prepare globals for the extractor run
#                 # These names MUST match the globals in your existing script
#                 extractor_globals = {
#                     # your script reads these:
#                     "QUERY_CONFIG_DIR_BASE": str(config_base) + os.sep,  # keep trailing separator for base + json
#                     "WAVE_DIR": str(wave_dir),
#                     "OUTPUT_BASE": output_base,  # Path is fine (your script uses Path)
#                     "NUM_GRIDS": 1,  # we run exactly one group folder (the current one)
#                     "SAMPLING_RATE": SAMPLING_RATE,

#                     # required imports your script expects in its global scope
#                     "__name__": "__main__",  # make it behave like a script
#                 }

#                 # Logging
#                 ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 print(f"[{ts}] ▶ Split={split} | Group={group_prefix}_{i} | Wave={wave_dir} | Out={output_base}")

                
#                 runpy.run_path(str(EXTRACTOR_SCRIPT), init_globals=extractor_globals)

#     print("\n✅ All embeddings generated for train/test across grids, queries, and unknown queries.")

# if __name__ == "__main__":
#     main()


from scripts.evaluation.hubert_grid_extractor import run_extraction

def main():
    for split, split_cfg in SPLITS.items():
        wave_dir = split_cfg["wave_dir"]
        out_root = split_cfg["output_root"]

        for group_prefix, out_subdir in GROUPS:
            for i in range(1, NUM_GROUPS + 1):
                config_base = Path("enrollment_sets") / split / f"{group_prefix}_{i}"
                output_base = out_root / out_subdir / f"grid_{i}"
                output_base.mkdir(parents=True, exist_ok=True)

                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{ts}] ▶ Split={split} | Group={group_prefix}_{i} | Wave={wave_dir} | Out={output_base}")

                run_extraction(
                    query_config_base=config_base,   # base path with JSON files
                    wave_dir=wave_dir,               # where your npy wave chunks are
                    output_base=output_base,         # output folder for CSVs
                    num_grids=1,                     # only process this one
                    sampling_rate=SAMPLING_RATE
                )

    print("\n✅ all embeddings generated for train/test across grids, queries, and unknown queries.")
