import os
import random
import json

# Constants
DATA_DIR = "data/processed_test/wave_chunks_2s"
BASE_OUTPUT_DIR = "grids_test_"  # Will be used as grids_test1, grids_test2, etc.
NUM_GRIDS = 5  # Number of full sets to generate
NUM_SPEAKERS_LIST = [2, 5, 10, 20, 40, 60, 80]
NUM_INSTANCES_LIST = [1, 5, 10, 20, 40, 60, 80]

def generate_file_grids(data_dir, output_base, num_grids, seed=42):
    for grid_idx in range(1, num_grids + 1):
        print(f"\n🔄 Generating Grid Set {grid_idx}...")
        output_dir = f"{output_base}{grid_idx}"
        os.makedirs(output_dir, exist_ok=True)

        random.seed(seed + grid_idx)  # different seed per set

        # Get all speaker folders
        speaker_folders = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        random.shuffle(speaker_folders)

        # Map speaker_id -> list of full file paths
        speaker_to_files = {}
        for speaker in speaker_folders:
            folder = os.path.join(data_dir, speaker)
            files = [f for f in os.listdir(folder) if f.endswith(".npy")]
            full_paths = sorted([os.path.join(speaker, f) for f in files])
            speaker_to_files[speaker] = full_paths

        # Initialize cumulative speakers
        cumulative_speakers = []

        saved_count = 0
        for num_speakers in NUM_SPEAKERS_LIST:
            if num_speakers > len(speaker_to_files):
                print(f"⚠️ Skipping {num_speakers} speakers, not enough total.")
                continue

            # Expand cumulative list
            needed_new = num_speakers - len(cumulative_speakers)
            if needed_new > 0:
                new_speakers = [s for s in speaker_folders if s not in cumulative_speakers][:needed_new]
                cumulative_speakers.extend(new_speakers)

            for num_instances in NUM_INSTANCES_LIST:
                grid = {"speakers": []}
                skip_grid = False

                for speaker in cumulative_speakers:
                    files = speaker_to_files[speaker]
                    if len(files) < num_instances:
                        print(f"⚠️ Skipping {num_speakers}s_{num_instances}i — speaker {speaker} has < {num_instances} files.")
                        skip_grid = True
                        break

                    selected_files = files[:num_instances]  # fixed order
                    grid["speakers"].append({
                        "speaker_id": speaker,
                        "files": selected_files
                    })

                if not skip_grid:
                    filename = f"{num_speakers}s_{num_instances}i.json"
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, "w") as f:
                        json.dump(grid, f, indent=2)
                    print(f"✅ Saved: {filepath}")
                    saved_count += 1

        print(f"📦 Total grids saved in set {grid_idx}: {saved_count}")

# Run the function
generate_file_grids(
    data_dir=DATA_DIR,
    output_base=BASE_OUTPUT_DIR,
    num_grids=5
)