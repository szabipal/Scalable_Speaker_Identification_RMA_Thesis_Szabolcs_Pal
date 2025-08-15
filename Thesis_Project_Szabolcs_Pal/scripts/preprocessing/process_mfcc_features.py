# # === process_mfcc_features.py ===
# from pathlib import Path
# import numpy as np
# from tqdm import tqdm
# from extract_mfcc import extract_mfcc_from_npy
# import torch

# WAVE_DIR = Path("data/processed_dev/wave_chunks_2s")
# MFCC_DIR = Path("data/processed_dev/mfcc_features_2s")
# MFCC_DIR.mkdir(parents=True, exist_ok=True)

# def normalize_mfcc(mfcc):
#     """
#     Apply per-utterance mean-variance normalization (CMVN) to MFCCs.
#     """
#     mean = np.mean(mfcc, axis=0, keepdims=True)
#     std = np.std(mfcc, axis=0, keepdims=True) + 1e-8  # avoid division by zero
#     return (mfcc - mean) / std

# for speaker_folder in tqdm(WAVE_DIR.iterdir(), desc="Extracting MFCCs"):
#     if not speaker_folder.is_dir():
#         continue
#     output_speaker_dir = MFCC_DIR / speaker_folder.name
#     output_speaker_dir.mkdir(exist_ok=True)

#     for npy_file in speaker_folder.glob("*.npy"):
#         try:
#             mfcc = extract_mfcc_from_npy(npy_file)
#             mfcc_norm = normalize_mfcc(mfcc)
#             out_path = output_speaker_dir / (npy_file.stem + ".npy")
#             np.save(out_path, mfcc_norm)
#         except Exception as e:
#             print(f"⚠️ Error with {npy_file}: {e}")

# === process_mfcc_features.py ===

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from extract_mfcc import extract_mfcc_from_npy
import torch

def normalize_mfcc(mfcc):
    """
    Apply per-utterance mean-variance normalization (CMVN) to MFCCs.
    """
    mean = np.mean(mfcc, axis=0, keepdims=True)
    std = np.std(mfcc, axis=0, keepdims=True) + 1e-8  # avoid division by zero
    return (mfcc - mean) / std

def process_mfcc_folder(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for speaker_folder in tqdm(input_dir.iterdir(), desc="Extracting MFCCs"):
        if not speaker_folder.is_dir():
            continue

        output_speaker_dir = output_dir / speaker_folder.name
        output_speaker_dir.mkdir(exist_ok=True)

        for npy_file in speaker_folder.glob("*.npy"):
            try:
                mfcc = extract_mfcc_from_npy(npy_file)
                mfcc_norm = normalize_mfcc(mfcc)
                out_path = output_speaker_dir / (npy_file.stem + ".npy")
                np.save(out_path, mfcc_norm)
            except Exception as e:
                print(f"⚠️ Error with {npy_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory of waveform .npy files")
    parser.add_argument("--output", required=True, help="Output directory to save MFCC features")
    args = parser.parse_args()

    process_mfcc_folder(args.input, args.output)
