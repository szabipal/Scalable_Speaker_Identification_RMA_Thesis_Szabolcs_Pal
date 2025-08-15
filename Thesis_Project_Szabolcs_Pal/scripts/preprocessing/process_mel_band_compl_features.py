# process_mel_complement_features.py

import os
import argparse
import numpy as np
from tqdm import tqdm


from extract_low_mel_band_expl import extract_low_melband_features
from extract_mid_mel_band_expl import extract_mid_melband_features
from extract_high_mel_band_expl import extract_high_melband_features

def save_feature(output_base, subfolder, rel_path, feature_vector):
    output_path = os.path.join(output_base, subfolder, rel_path).replace(".wav", ".npy")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, feature_vector)

def process_all_mel_features(input_dir, output_dir):
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

                    try:
                        y = np.load(input_path)
                        sr = 16000  # Set your sampling rate here

                        low_feat = extract_low_melband_features(y, sr)
                        # print(f'low_feat {low_feat}')
                        save_feature(output_dir, "low", rel_path, low_feat)
                    except Exception as e:
                        print(f"[LOW] Skipping {input_path} → {e}")

                    try:
                        mid_feat = extract_mid_melband_features(y, sr)
                        # print(f'mid_feat {mid_feat}')
                        save_feature(output_dir, "mid", rel_path, mid_feat)
                    except Exception as e:
                        print(f"[MID] Skipping {input_path} → {e}")

                    try:
                        high_feat = extract_high_melband_features(y, sr)
                        # print(f'high_feat {high_feat}')
                        save_feature(output_dir, "high", rel_path, high_feat)
                    except Exception as e:
                        print(f"[HIGH] Skipping {input_path} → {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory with .wav files")
    parser.add_argument("--output", required=True, help="Output base directory")
    args = parser.parse_args()

    process_all_mel_features(args.input, args.output)

#python process_mel_complement_features.py --input data/wave_chunks --output data/mel_complement_features
