
import os
import json
from pathlib import Path

def create_query_from_enrollment_json(
    grid_json_dir="grids_cleaned_",
    wave_chunk_dir="data/processed/wave_chunks_2s",
    output_dir="queries_cleaned_",
    num_grids=3,
    max_queries_per_speaker=5
):
    chunk_root = Path(wave_chunk_dir)

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

            query_speakers = []
            for entry in enrollment_data.get("speakers", []):
                speaker_id = entry["speaker_id"]
                enrolled_set = set(Path(f).name for f in entry["files"])

                speaker_path = chunk_root / speaker_id
                if not speaker_path.exists():
                    print(f"[!] Missing speaker path: {speaker_path}")
                    continue

                all_files = {f.name for f in speaker_path.glob("*.npy")}
                query_files = sorted(list(all_files - enrolled_set))[:max_queries_per_speaker]

                if query_files:
                    full_paths = [f"{speaker_id}/{f}" for f in query_files]
                    query_speakers.append({
                        "speaker_id": speaker_id,
                        "files": full_paths
                    })

            out_path = query_path / json_file.name
            with open(out_path, "w") as f:
                json.dump({"speakers": query_speakers}, f, indent=2)

            print(f"[✓] Saved query file {out_path} with {len(query_speakers)} speakers, {sum(len(s['files']) for s in query_speakers)} total queries.")

if __name__ == "__main__":
    create_query_from_enrollment_json()
