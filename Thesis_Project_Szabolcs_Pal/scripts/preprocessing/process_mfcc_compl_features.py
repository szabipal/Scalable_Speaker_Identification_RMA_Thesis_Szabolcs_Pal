# process_mfcc_complement_features.py

import os
import numpy as np
import argparse
from tqdm import tqdm
from extract_mfcc_expl import extract_mfcc_complement_features  # expects waveform array, not path

def process_directory(input_dir, output_dir, sr=16000):
    os.makedirs(output_dir, exist_ok=True)

    all_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".npy"):
                all_files.append(os.path.join(root, file))

    for input_path in tqdm(all_files, desc="Processing files"):
        rel_path = os.path.relpath(input_path, input_dir)

        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".npy"):
                    input_path = os.path.join(root, file)
                    rel_path = os.path.relpath(input_path, input_dir)
                    output_path = os.path.join(output_dir, rel_path)

                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    try:
                        waveform = np.load(input_path)
                        features = extract_mfcc_complement_features(waveform, sr=sr)
                        # print()
                        # print(features)
                        np.save(output_path, features)
                    except Exception as e:
                        print(f"[ERROR] Skipping {input_path} due to: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory with .npy waveform chunks")
    parser.add_argument("--output", required=True, help="Output directory to save feature .npy files")
    parser.add_argument("--sr", type=int, default=16000, help="Sampling rate of the chunks (default: 16000)")
    args = parser.parse_args()

    process_directory(args.input, args.output, sr=args.sr)

# how to run: python process_mfcc_complement_features.py --input data/wavs --output data/mfcc_complement_features
