import os
import json
from pathlib import Path
import random

def create_unknown_speaker_queries(
    grid_json_dir="grids_cleaned_",
    wave_chunk_dir="data/processed/wave_chunks_2s",
    output_dir="unknown_queries_cleaned_",
    num_grids=3,
    max_queries_per_speaker=5,
    seed=42
):
    random.seed(seed)
    chunk_root = Path(wave_chunk_dir)

    all_speaker_ids = {d.name for d in chunk_root.iterdir() if d.is_dir()}

    for grid_idx in range(1, num_grids + 1):
        grid_path = Path(f"{grid_json_dir}{grid_idx}")
        query_path = Path(f"{output_dir}{grid_idx}")
        query_path.mkdir(parents=True, exist_ok=True)

        if not grid_path.exists():
            print(f"[!] Missing grid folder: {grid_path}")
            continue

        for json_file in grid_path.glob("*.json"):
            with open(json_file, "r") as f:
                enrollment_data = json.load(f)

            enrolled_speakers = {entry["speaker_id"] for entry in enrollment_data.get("speakers", [])}
            unknown_speakers = list(all_speaker_ids - enrolled_speakers)

            query_speakers = []
            for speaker_id in random.sample(unknown_speakers, min(len(unknown_speakers), 10)):
                speaker_path = chunk_root / speaker_id
                if not speaker_path.exists():
                    continue

                all_files = sorted([f.name for f in speaker_path.glob("*.npy")])
                if not all_files:
                    continue

                query_files = all_files[:max_queries_per_speaker]
                full_paths = [f"{speaker_id}/{f}" for f in query_files]
                query_speakers.append({
                    "speaker_id": speaker_id,
                    "files": full_paths
                })

            out_path = query_path / json_file.name
            with open(out_path, "w") as f:
                json.dump({"speakers": query_speakers}, f, indent=2)

            print(f"[✓] Saved unknown query file {out_path} with {len(query_speakers)} speakers, {sum(len(s['files']) for s in query_speakers)} total queries.")

if __name__ == "__main__":
    create_unknown_speaker_queries()
