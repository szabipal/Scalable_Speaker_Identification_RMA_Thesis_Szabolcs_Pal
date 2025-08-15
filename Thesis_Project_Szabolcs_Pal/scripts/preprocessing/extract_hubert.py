import os
import re
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel
import safetensors

import torch
print("Torch version:", torch.__version__)

if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CONFIGURATION
DATA_DIR = "data/processed/wave_chunks_2s"        # Folder with speaker subfolders
OUTPUT_CSV = "hubert_embeddings.csv"
N_SAMPLES_PER_SPEAKER = 20
SAMPLING_RATE = 16000



# Load HuBERT model and feature extractor
extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = AutoModel.from_pretrained(
    "facebook/hubert-base-ls960",
    output_hidden_states=True,
    trust_remote_code=True,
    use_safetensors=True
)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Pattern: speaker-session-instance_chunk#.npy
FILENAME_REGEX = re.compile(r"(\d+)-(\d+)-(\d+)_chunk(\d+)\.npy")

# Collect .npy files per speaker
speaker_files = {}
for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".npy") and FILENAME_REGEX.match(file):
            speaker_id = FILENAME_REGEX.match(file).group(1)
            full_path = os.path.join(root, file)
            speaker_files.setdefault(speaker_id, []).append(full_path)

# Process each speaker
rows = []
for speaker_id, file_list in tqdm(speaker_files.items(), desc="Processing speakers"):
    selected_files = random.sample(file_list, min(N_SAMPLES_PER_SPEAKER, len(file_list)))

    for file_path in selected_files:
        try:
            # Load raw waveform from .npy
            waveform = np.load(file_path)
            if waveform.ndim > 1:
                waveform = waveform[0]  # Convert from shape (1, N) to (N,)

            # Preprocess audio
            inputs = extractor(waveform, sampling_rate=SAMPLING_RATE, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Run HuBERT
            with torch.no_grad():
                outputs = model(**inputs)
                last_hidden = outputs.hidden_states[-1]  # shape: (1, seq_len, hidden_dim)
                embedding = last_hidden.mean(dim=1).squeeze().cpu().numpy()  # (hidden_dim,)

            # Parse metadata from filename
            filename = os.path.basename(file_path)
            match = FILENAME_REGEX.match(filename)
            session_id = match.group(2)
            instance_id = match.group(3)
            chunk_id = match.group(4)

            rows.append({
                "speaker_id": speaker_id,
                "session_id": session_id,
                "instance_id": instance_id,
                "chunk_id": chunk_id,
                "embedding": embedding.tolist()
            })

        except Exception as e:
            print(f"⚠️ Failed to process {file_path}: {e}")

# Save DataFrame with embedding column
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Done. Saved {len(df)} rows to {OUTPUT_CSV}")
