# import os
# from pathlib import Path
# import librosa
# import numpy as np
# import webrtcvad
# import struct
# from tqdm import tqdm


# DATA_DIR = Path("data/dev-clean/")         
# OUTPUT_DIR = Path("data/processed_dev/wave_chunks_2s")    
# CHUNK_DURATION = 2.0  
# TARGET_SR = 16000

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# def apply_vad(y, sr, frame_duration=30, aggressiveness=2):
#     vad = webrtcvad.Vad(aggressiveness)
#     frame_size = int(sr * frame_duration / 1000)
#     bytes_per_sample = 2

#     # Ensure audio is padded for full frames
#     if len(y) % frame_size != 0:
#         padding = frame_size - (len(y) % frame_size)
#         y = np.pad(y, (0, padding), mode='constant')

#     pcm = (y * 32768).astype(np.int16).tobytes()
#     voiced_samples = []

#     for i in range(0, len(pcm), frame_size * bytes_per_sample):
#         frame = pcm[i:i + frame_size * bytes_per_sample]
#         if len(frame) < frame_size * bytes_per_sample:
#             continue
#         is_speech = vad.is_speech(frame, sr)
#         if is_speech:
#             start = i // bytes_per_sample
#             end = start + frame_size
#             voiced_samples.append(y[start:end])

#     if voiced_samples:
#         return np.concatenate(voiced_samples)
#     else:
#         return np.array([])

# def preprocess_audio(file_path, target_sr, chunk_duration):
#     y, sr = librosa.load(file_path, sr=target_sr)
#     y_voiced = apply_vad(y, sr)
#     if len(y_voiced) == 0:
#         return []

#     chunk_size = int(chunk_duration * target_sr)
#     chunks = []
#     for start in range(0, len(y_voiced), chunk_size):
#         end = start + chunk_size
#         chunk = y_voiced[start:end]
#         if len(chunk) < chunk_size:
#             continue  
#         chunks.append(chunk)
#     return chunks

# def process_all(data_dir, output_dir):
#     flac_files = list(data_dir.rglob("*.flac"))
#     print(f"Found {len(flac_files)} audio files.")

#     for file_path in tqdm(flac_files, desc="Processing audio files"):
#         filename = file_path.stem  
#         speaker_id = filename.split("-")[0]
#         speaker_folder = output_dir / speaker_id
#         speaker_folder.mkdir(parents=True, exist_ok=True)

#         try:
#             chunks = preprocess_audio(
#                 file_path,
#                 target_sr=TARGET_SR,
#                 chunk_duration=CHUNK_DURATION
#             )
#         except Exception as e:
#             print(f"⚠️ Failed to process {file_path}: {e}")
#             continue

#         for i, chunk in enumerate(chunks):
#             out_file = speaker_folder / f"{filename}_chunk{i}.npy"
#             np.save(out_file, chunk)


# if __name__ == "__main__":
#     process_all(DATA_DIR, OUTPUT_DIR)

# preprocess.py

import os
import argparse
from pathlib import Path
import librosa
import numpy as np
import webrtcvad
import struct
from tqdm import tqdm


CHUNK_DURATION = 2.0  
TARGET_SR = 16000

def apply_vad(y, sr, frame_duration=30, aggressiveness=2):
    vad = webrtcvad.Vad(aggressiveness)
    frame_size = int(sr * frame_duration / 1000)
    bytes_per_sample = 2

    if len(y) % frame_size != 0:
        padding = frame_size - (len(y) % frame_size)
        y = np.pad(y, (0, padding), mode='constant')

    pcm = (y * 32768).astype(np.int16).tobytes()
    voiced_samples = []

    for i in range(0, len(pcm), frame_size * bytes_per_sample):
        frame = pcm[i:i + frame_size * bytes_per_sample]
        if len(frame) < frame_size * bytes_per_sample:
            continue
        if vad.is_speech(frame, sr):
            start = i // bytes_per_sample
            end = start + frame_size
            voiced_samples.append(y[start:end])

    return np.concatenate(voiced_samples) if voiced_samples else np.array([])

def preprocess_audio(file_path, target_sr, chunk_duration):
    y, sr = librosa.load(file_path, sr=target_sr)
    y_voiced = apply_vad(y, sr)
    if len(y_voiced) == 0:
        return []

    chunk_size = int(chunk_duration * target_sr)
    chunks = [
        y_voiced[start:start + chunk_size]
        for start in range(0, len(y_voiced), chunk_size)
        if len(y_voiced[start:start + chunk_size]) == chunk_size
    ]
    return chunks

def process_all(data_dir, output_dir):
    flac_files = list(data_dir.rglob("*.flac"))
    print(f"Found {len(flac_files)} audio files.")

    for file_path in tqdm(flac_files, desc="Processing audio files"):
        filename = file_path.stem  
        speaker_id = filename.split("-")[0]
        speaker_folder = output_dir / speaker_id
        speaker_folder.mkdir(parents=True, exist_ok=True)

        try:
            chunks = preprocess_audio(
                file_path,
                target_sr=TARGET_SR,
                chunk_duration=CHUNK_DURATION
            )
        except Exception as e:
            print(f"⚠️ Failed to process {file_path}: {e}")
            continue

        for i, chunk in enumerate(chunks):
            out_file = speaker_folder / f"{filename}_chunk{i}.npy"
            np.save(out_file, chunk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input directory (root of LibriSpeech-style .flac files)")
    parser.add_argument("--output", type=str, required=True, help="Output directory for wave chunks")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    process_all(input_path, output_path)
