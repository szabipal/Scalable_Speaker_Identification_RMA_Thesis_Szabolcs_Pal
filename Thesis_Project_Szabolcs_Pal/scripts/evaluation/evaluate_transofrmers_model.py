"""
pairwise verification between query and enrolled speaker embeddings (siamese mlp).

what it does
- loads per-grid csvs of embeddings for queries and enrolled speakers.
- for each matching filename across the two folders, computes all pairwise
  similarities with a siamese model, then aggregates a simple 'match rate'
  per (query_id, query_speaker, enrolled_speaker).
- writes one summary csv per input filename into `output_dir`.

notes
- expects an 'embedding' column containing lists (or stringified lists) and
  id columns like speaker_id/session_id/instance_id/chunk_id.
- the model should output a similarity score where higher = more similar
  (compared to `threshold` to produce 0/1 matches).
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# dynamically append project root (kept minimal; adjust if your layout changes)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from models.siamese_nn import SiameseMLP

def parse_embedding(emb):
    """
    coerce an embedding cell into a float32 numpy array.

    parameters
    ----------
    emb : list | str | np.ndarray
        embedding in native list/ndarray form, or a string like "[0.1, 0.2, ...]".

    returns
    -------
    np.ndarray (float32)
    """
    if isinstance(emb, str):
        return np.array(eval(emb)).astype(np.float32)
    return emb

def verify_speakers_with_folders(model, query_folder, enrolled_folder, output_dir, threshold=0.5, device="cpu"):
    """
    compute query↔enrolled similarities file-by-file and summarize match rates.

    parameters
    ----------
    model : torch.nn.Module
        siamese model producing a similarity score for (query, enrolled).
    query_folder : str | Path
        folder with per-grid query csvs (one csv per grid/config).
    enrolled_folder : str | Path
        folder with per-grid enrolled csvs; filenames must match `query_folder`.
    output_dir : str | Path
        where to write the per-file summary csvs.
    threshold : float
        similarity cutoff to flag a pair as a match (>= threshold → 1).
    device : str
        'cpu' or 'cuda'.

    behavior
    - for each common csv filename: loads both dataframes, parses embeddings,
      computes all pairwise sims, then aggregates:
        sum (matches), count (trials), and match_rate = sum / count
      per (query_id, query_speaker, enrolled_speaker).
    """
    model.eval()
    model.to(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_files = {f.name: f for f in Path(query_folder).glob("*.csv")}
    enrolled_files = {f.name: f for f in Path(enrolled_folder).glob("*.csv")}
    common_files = sorted(set(query_files.keys()) & set(enrolled_files.keys()))

    for filename in tqdm(common_files, desc="Processing matched files"):
        query_df = pd.read_csv(query_files[filename])
        enrolled_df = pd.read_csv(enrolled_files[filename])

        # parse embeddings from stringified lists if needed
        query_df["embedding"] = query_df["embedding"].apply(parse_embedding)
        enrolled_df["embedding"] = enrolled_df["embedding"].apply(parse_embedding)

        all_results = []

        for _, query_row in query_df.iterrows():
            query_emb = torch.tensor(query_row["embedding"]).to(device)
            query_speaker = query_row["speaker_id"]
            # compact query identifier for downstream grouping
            query_id = f'{query_speaker}_{query_row["session_id"]}_{query_row["instance_id"]}_chunk{query_row["chunk_id"]}'

            for _, enrolled_row in enrolled_df.iterrows():
                enrolled_emb = torch.tensor(enrolled_row["embedding"]).to(device)
                enrolled_speaker = enrolled_row["speaker_id"]
                # compact enrolled identifier (kept for traceability)
                enrolled_id = f'{enrolled_speaker}_{enrolled_row["session_id"]}_{enrolled_row["instance_id"]}_chunk{enrolled_row["chunk_id"]}'

                with torch.no_grad():
                    sim = model(query_emb.unsqueeze(0), enrolled_emb.unsqueeze(0)).item()

                match = int(sim >= threshold)

                all_results.append({
                    "query_id": query_id,
                    "query_speaker": query_speaker,
                    "enrolled_id": enrolled_id,
                    "enrolled_speaker": enrolled_speaker,
                    "similarity": sim,
                    "match": match
                })

        # aggregate per (query_id, query_speaker, enrolled_speaker)
        results_df = pd.DataFrame(all_results)
        grouped = results_df.groupby(["query_id", "query_speaker", "enrolled_speaker"])
        summary_df = grouped["match"].agg(["sum", "count"]).reset_index()
        summary_df["match_rate"] = summary_df["sum"] / summary_df["count"]

        # save summary under the original filename
        output_file = output_dir / f"{filename}"
        summary_df.to_csv(output_file, index=False)
        print(f"[INFO] Saved: {output_file}")


if __name__ == "__main__":
    # example load (paths/models are placeholders; adapt to your training artifacts)
    model = SiameseMLP(input_dim=768, hidden_dim=256)
    model.load_state_dict(torch.load("saved_siamese/siamese_mlp_final.pt"))
    model.eval()
