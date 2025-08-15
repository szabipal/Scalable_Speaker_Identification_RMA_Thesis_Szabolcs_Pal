

# from pathlib import Path
# import numpy as np
# from tqdm import tqdm
# from extract_mel import extract_mel_spectrogram
# from tqdm import tqdm

# def process_banded_mel_features(wave_dir, mel_base_dir):
#     WAVE_DIR = Path(wave_dir)
#     MEL_BASE_DIR = Path(mel_base_dir)

#     # Define subfolders for each band
#     low_dir = MEL_BASE_DIR / "low"
#     mid_dir = MEL_BASE_DIR / "mid"
#     high_dir = MEL_BASE_DIR / "high"

#     for band_dir in [low_dir, mid_dir, high_dir]:
#         band_dir.mkdir(parents=True, exist_ok=True)

#     for speaker_folder in tqdm(WAVE_DIR.iterdir(), desc="Extracting Mel Bands"):
#         if not speaker_folder.is_dir():
#             continue

#         # Create speaker subfolders in each band directory
#         low_speaker_dir = low_dir / speaker_folder.name
#         mid_speaker_dir = mid_dir / speaker_folder.name
#         high_speaker_dir = high_dir / speaker_folder.name

#         for d in [low_speaker_dir, mid_speaker_dir, high_speaker_dir]:
#             d.mkdir(parents=True, exist_ok=True)

#         for npy_file in speaker_folder.glob("*.npy"):
#             try:
#                 mel = extract_mel_spectrogram(npy_file)
#                 n_mels = mel.shape[0]

#                 # Split into bands
#                 low_band = mel[:n_mels // 3, :]
#                 mid_band = mel[n_mels // 3: 2 * n_mels // 3, :]
#                 high_band = mel[2 * n_mels // 3:, :]

#                 # Save each band in its respective folder
#                 np.save(low_speaker_dir / (npy_file.stem + ".npy"), low_band)
#                 np.save(mid_speaker_dir / (npy_file.stem + ".npy"), mid_band)
#                 np.save(high_speaker_dir / (npy_file.stem + ".npy"), high_band)

#             except Exception as e:
#                 print(f"⚠️ Error with {npy_file}: {e}")

# if __name__ == "__main__":
#     process_banded_mel_features("data/processed_dev/wave_chunks_2s", "data/processed_dev/mel_features_banded_2s")

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from extract_mel import extract_mel_spectrogram

def process_banded_mel_features(wave_dir, mel_base_dir):
    WAVE_DIR = Path(wave_dir)
    MEL_BASE_DIR = Path(mel_base_dir)

    # Define subfolders for each band
    low_dir = MEL_BASE_DIR / "low"
    mid_dir = MEL_BASE_DIR / "mid"
    high_dir = MEL_BASE_DIR / "high"

    for band_dir in [low_dir, mid_dir, high_dir]:
        band_dir.mkdir(parents=True, exist_ok=True)

    for speaker_folder in tqdm(WAVE_DIR.iterdir(), desc="Extracting Mel Bands"):
        if not speaker_folder.is_dir():
            continue

        # Create speaker subfolders in each band directory
        low_speaker_dir = low_dir / speaker_folder.name
        mid_speaker_dir = mid_dir / speaker_folder.name
        high_speaker_dir = high_dir / speaker_folder.name

        for d in [low_speaker_dir, mid_speaker_dir, high_speaker_dir]:
            d.mkdir(parents=True, exist_ok=True)

        for npy_file in speaker_folder.glob("*.npy"):
            try:
                mel = extract_mel_spectrogram(npy_file)
                n_mels = mel.shape[0]

                # Split into bands
                low_band = mel[:n_mels // 3, :]
                mid_band = mel[n_mels // 3: 2 * n_mels // 3, :]
                high_band = mel[2 * n_mels // 3:, :]

                # Save each band in its respective folder
                np.save(low_speaker_dir / (npy_file.stem + ".npy"), low_band)
                np.save(mid_speaker_dir / (npy_file.stem + ".npy"), mid_band)
                np.save(high_speaker_dir / (npy_file.stem + ".npy"), high_band)

            except Exception as e:
                print(f"⚠️ Error with {npy_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory of .npy waveform chunks")
    parser.add_argument("--output", required=True, help="Output base directory to save mel bands")
    args = parser.parse_args()

    process_banded_mel_features(args.input, args.output)
