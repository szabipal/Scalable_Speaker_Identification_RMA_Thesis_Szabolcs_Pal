import os
import random
import json

# Constants

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed", "wave_chunks_2s")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "grids_cleaned_")

os.makedirs(OUTPUT_DIR, exist_ok=True)
# DATA_DIR = "../data/processed/wave_chunks_2s"
# OUTPUT_DIR = "grids_cleaned_"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SPEAKERS_LIST = [2, 5, 10, 20, 40, 60, 80]
NUM_INSTANCES_LIST = [1, 5, 10, 20, 40, 60, 80]

def generate_file_grids(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, seed=42):
    random.seed(seed)

    speaker_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    speaker_to_files = {}

    # Build mapping of speaker_id -> list of full file paths
    for speaker in speaker_folders:
        folder = os.path.join(data_dir, speaker)
        files = [f for f in os.listdir(folder) if f.endswith(".npy")]
        full_paths = [os.path.join(speaker, f) for f in files]  # save relative paths
        speaker_to_files[speaker] = full_paths

    for num_speakers in NUM_SPEAKERS_LIST:
        for num_instances in NUM_INSTANCES_LIST:
            if num_speakers > len(speaker_to_files):
                continue

            selected_speakers = random.sample(list(speaker_to_files.keys()), num_speakers)
            grid = {"speakers": []}

            for speaker in selected_speakers:
                files = speaker_to_files[speaker]
                if len(files) < num_instances:
                    break  # skip this combo if any speaker has too few files

                chosen = random.sample(files, num_instances)
                grid["speakers"].append({
                    "speaker_id": speaker,
                    "files": chosen
                })

            # Save only complete grids
            if len(grid["speakers"]) == num_speakers:
                filename = f"{num_speakers}s_{num_instances}i.json"
                with open(os.path.join(output_dir, filename), "w") as f:
                    json.dump(grid, f, indent=2)
                print(f"✅ Saved: {filename}")
            else:
                print(f"⚠️ Skipped {num_speakers}s_{num_instances}i — some speakers had < {num_instances} files.")

generate_file_grids()
