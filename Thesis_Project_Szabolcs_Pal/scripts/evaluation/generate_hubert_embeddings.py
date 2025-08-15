"""
extract hubert embeddings for query grids into per-grid csvs.

what this script does
- loads a pretrained hubert-base model and feature extractor.
- reads grid jsons (speaker → list of relative files) and finds the matching .npy wave chunks.
- runs the model to get frame-level hidden states, averages over time, and saves one 768-d vector per chunk.
- writes one csv per json under OUTPUT_BASE/grid_<k>/..., each row = one chunk with ids and its embedding.

inputs
- QUERY_CONFIG_DIR_BASE: base folder prefix that contains numbered grid folders (e.g., 'grids_test_1', ...).
- WAVE_DIR: root of .npy wave chunks (relative paths in jsons are resolved against this).
- NUM_GRIDS: how many grid folders to scan (1..NUM_GRIDS).
- SAMPLING_RATE: sampling rate used to tell the extractor (should match how chunks were built).

outputs
- a csv per grid json under OUTPUT_BASE/grid_<id>/<json_name>.csv with columns:
  speaker_id, session_id, instance_id, chunk_id, embedding (list[float]).

notes
- the filename regex is used to recover session/instance/chunk metadata when available; falls back to 'unknown'.
- embeddings are last hidden state mean-pooled across time (simple, stable baseline).
- script is device-aware (cuda if available).
"""

import os
import json
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel
from pathlib import Path
import safetensors  # kept for safety with safetensors checkpoints

# small compatibility shim for older torch versions
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- configuration (adjust paths to your repo layout) ---
QUERY_CONFIG_DIR_BASE = "grids_test_"                      # base prefix; script looks for grids_test_1..NUM_GRIDS
WAVE_DIR = "data/processed_test/wave_chunks_2s"            # root where relative files in json are located
OUTPUT_BASE = Path("hubert_embeddings_test/grids")         # destination for per-grid csvs
NUM_GRIDS = 3                                              # number of grid folders to scan
SAMPLING_RATE = 16000                                      # samplerate expected by the extractor

# --- load hubert model + feature extractor (cpu/gpu automatically handled) ---
extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = AutoModel.from_pretrained(
    "facebook/hubert-base-ls960",
    output_hidden_states=True,   # needed to access hidden states
    trust_remote_code=True,
    use_safetensors=True
)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# parse metadata from filenames like: speaker-session-instance_chunkX.npy
FILENAME_REGEX = re.compile(r"(\d+)-(\d+)-(\d+)_chunk(\d+)\.npy")

# --- main extraction loop over grid folders and their jsons ---
for grid_idx in range(1, NUM_GRIDS + 1):
    config_folder = Path(f"{QUERY_CONFIG_DIR_BASE}{grid_idx}")
    output_grid_dir = OUTPUT_BASE / f"grid_{grid_idx}"
    output_grid_dir.mkdir(parents=True, exist_ok=True)

    if not config_folder.exists():
        print(f" Skipping missing config folder: {config_folder}")
        continue

    # process each grid json: collect rows and dump to csv
    for json_file in config_folder.glob("*.json"):
        with open(json_file, "r") as f:
            config = json.load(f)

        rows = []
        for speaker in tqdm(config["speakers"], desc=f"Grid {grid_idx} - {json_file.name}"):
            speaker_id = speaker["speaker_id"]

            for rel_file in speaker["files"]:
                full_path = os.path.join(WAVE_DIR, rel_file)

                try:
                    # load chunk (expects float waveform from .npy)
                    waveform = np.load(full_path)
                    if waveform.ndim > 1:  # handle [channels, time]
                        waveform = waveform[0]

                    # feature extractor → tensors on the correct device
                    inputs = extractor(waveform, sampling_rate=SAMPLING_RATE, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # forward pass; take last hidden state and mean-pool over time
                    with torch.no_grad():
                        outputs = model(**inputs)
                        last_hidden = outputs.hidden_states[-1]          # [B, T, D]
                        embedding = last_hidden.mean(dim=1).squeeze().cpu().numpy()  # [D]

                    # recover ids from filename if possible
                    filename = os.path.basename(full_path)
                    match = FILENAME_REGEX.match(filename)
                    if match:
                        session_id, instance_id, chunk_id = match.group(2), match.group(3), match.group(4)
                    else:
                        session_id = instance_id = chunk_id = "unknown"

                    rows.append({
                        "speaker_id": speaker_id,
                        "session_id": session_id,
                        "instance_id": instance_id,
                        "chunk_id": chunk_id,
                        "embedding": embedding.tolist()
                    })

                except Exception as e:
                    # keep going on bad files; log the path and error
                    print(f" Failed to process {full_path}: {e}")

        # dump one csv per json (same name, .csv suffix) under the grid output folder
        output_csv_path = output_grid_dir / json_file.with_suffix(".csv").name
        pd.DataFrame(rows).to_csv(output_csv_path, index=False)
        print(f" Saved {len(rows)} embeddings to {output_csv_path}")
