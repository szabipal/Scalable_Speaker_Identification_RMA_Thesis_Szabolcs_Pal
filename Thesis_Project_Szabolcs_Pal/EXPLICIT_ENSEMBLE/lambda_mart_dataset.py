import os
import json
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List
import numpy as np
import re

def load_embeddings(path: Path):
    """
    Load speaker-wise .npy embeddings from a given path.
    Returns a dict: {speaker_id: [(file_name, embedding), ...]}
    """
    embeddings = {}
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    for speaker_dir in path.iterdir():
        if speaker_dir.is_dir():
            speaker_id = speaker_dir.name
            embeddings[speaker_id] = []
            for file in speaker_dir.glob("*.npy"):
                emb = np.load(file)
                embeddings[speaker_id].append((file.name, emb))
    return embeddings


def compute_features_with_normalization(query_embedding, candidate_embeddings, quantiles, instances_per_speaker: int):
    from scipy.spatial.distance import cdist, cosine # Import cosine as well if you plan to use it directly

    # === Distance calculations ===
    # Change 'euclidean' to 'cosine' here
    dists = cdist([query_embedding], candidate_embeddings, metric='cosine')[0] # <<< CHANGE IS HERE

    # For cosine similarity, a value of 1 means identical, 0 means orthogonal, -1 means opposite.
    # SciPy's `cosine` metric actually computes the *cosine distance*, which is 1 - cosine_similarity.
    # So, 0 is identical, 1 is orthogonal, 2 is opposite. This is good for distance calculations.

    centroid = np.mean(candidate_embeddings, axis=0)

    # For centroid distance with cosine, it's often more appropriate to
    # calculate the cosine similarity of the query to the centroid, then convert to distance.
    # Or, normalize all embeddings to unit vectors first and then use Euclidean distance.
    # However, since you're using `cdist` for the other distances, let's stick to a `cdist`-like approach for consistency.

    # To calculate centroid distance with cosine, you could do:
    # 1. Compute centroid.
    # 2. Calculate cosine distance between query_embedding and centroid.
    #    This is generally more robust than directly applying Euclidean to cosine.
    centroid_dist_raw = cosine(query_embedding, centroid) # Use scipy.spatial.distance.cosine for single pair

    mean_dist = np.mean(dists)
    min_dist = np.min(dists)

    raw = {
        "mean_dist": mean_dist,
        "min_dist": min_dist,
        "centroid_dist": centroid_dist_raw, # Use the cosine centroid distance
    }

    # ... (rest of the normalization code remains the same) ...
    # === Hardcoded dev values ===
    n_dev_centroid = 7000  # e.g., 500 centroid distances used for quantile calculation
    n_dev_pairs = 5000    # e.g., 5000 intra-speaker pairwise distances used

    # === Use instances_per_speaker from grid name ===
    n_current = instances_per_speaker
    if n_current != 1:
        n_current_pairs = n_current * (n_current - 1) / 2 if n_current > 1 else 1
    else:
        n_current_pairs = 1
    norm = {}

    for key, value in raw.items():
        # Map to the correct quantile key
        if key == "centroid_dist":
            q_key = "centroid"
        else:
            q_key = "cosine"
        q_data = quantiles.get("intra", {}).get(q_key, {})
        q10 = q_data.get("q10")
        q90 = q_data.get("q90")

        if q10 is not None and q90 is not None and q90 != q10:
            # Choose the correct scaling method
            if key == "centroid_dist":
                scaling_factor = (n_dev_centroid / n_current) ** 0.5
            else:  # mean_dist or min_dist
                scaling_factor = (n_dev_pairs / n_current_pairs) ** 0.5

            if value > q90:
                
                norm_val = 1
                
            else:
                # print('within range')
                scaled_iqr = (q90 - q10) * scaling_factor
                norm_val = (value - q10) / scaled_iqr
                # print(f'{value} became {norm_val}')
                if norm_val < -1:
                    norm_val = 0
        else:
            # norm_val = value  # fallback if quantiles missing
            continue
        norm[f"norm_{key}"] = norm_val

    return {**norm}


def create_merged_lambdamart_dataset(
    config_id: str,
    grid_name: str,
    query_root: str,
    grid_root: str,
    feature_types: List[str],
    quantile_dir: str,
    output_path: str,
    fraction: float = 1.0
):
    all_query_embeddings = {}
    all_enrolled_embeddings = {}
    all_quantiles = {}

    
    match = re.search(r"(\d+)i", grid_name)
    if match:
        instances_per_speaker = int(match.group(1))
    else:
        raise ValueError(f"Could not extract instances per speaker from grid name: {grid_name}")

    for feat in feature_types:
        query_path = Path(query_root) / grid_name / feat
        enrolled_path = Path(grid_root) / grid_name / feat
        quantile_path = Path(quantile_dir) / f"quantiles_{feat}.json"

        all_query_embeddings[feat] = load_embeddings(query_path)
        all_enrolled_embeddings[feat] = load_embeddings(enrolled_path)
        with open(quantile_path, "r") as f:
            all_quantiles[feat] = json.load(f)

    rows = []
    base_feat = feature_types[0]

    for query_speaker, instances in tqdm(all_query_embeddings[base_feat].items(), desc=f"Processing config {config_id}"):
        sampled = random.sample(instances, k=max(1, int(len(instances) * fraction)))
        for file_name, _ in sampled:
            qid = f"{config_id}_{query_speaker}_{file_name}"

            for candidate_speaker in all_enrolled_embeddings[base_feat].keys():
                merged_features = {}
                label = 1 if candidate_speaker == query_speaker else 0

                for feat in feature_types:
                    query_embedding = dict(all_query_embeddings[feat][query_speaker])[file_name]
                    candidate_embeddings = [emb for _, emb in all_enrolled_embeddings[feat][candidate_speaker]]
                    quantiles = all_quantiles[feat]

                    feat_vals = compute_features_with_normalization(query_embedding, candidate_embeddings, quantiles, instances_per_speaker)
                    for k, v in feat_vals.items():
                        merged_features[f"{feat}_{k}"] = v

                merged_features.update({
                    "label": label,
                    "qid": qid,
                    "query_speaker": query_speaker,
                    "query_file": file_name,
                    "candidate_speaker": candidate_speaker
                })
                rows.append(merged_features)

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"lambdamart_merged_{grid_name}.csv"
    df.to_csv(out_file, index=False)
    print(f"✅ Merged dataset saved: {out_file}")
    return out_file


if __name__ == "__main__":

    grid_map = {
        1: "10s_10i",   2: "20s_10i",   3: "2s_10i",   4: "40s_10i",   5: "5s_10i",   6: "60s_10i",   7: "80s_10i",
        8: "10s_1i",    9: "20s_1i",   10: "2s_1i",   11: "40s_1i",   12: "5s_1i",   13: "60s_1i",   14: "80s_1i",
        15: "10s_20i", 16: "20s_20i",  17: "2s_20i",  18: "40s_20i",  19: "5s_20i",  20: "60s_20i",  21: "80s_20i",
        22: "10s_40i", 23: "20s_40i",  24: "2s_40i",  25: "40s_40i",  26: "5s_40i",  27: "60s_40i",  28: "80s_40i",
        29: "10s_5i",  30: "20s_5i",   31: "2s_5i",   32: "40s_5i",   33: "5s_5i",   34: "60s_5i",   35: "80s_5i",
        36: "10s_60i", 37: "20s_60i",  38: "2s_60i",  39: "40s_60i",  40: "5s_60i",  41: "60s_60i",  42: "80s_60i",
        43: "10s_80i", 44: "20s_80i",  45: "2s_80i",  46: "40s_80i",  47: "5s_80i",  48: "60s_80i",  49: "80s_80i"
    }


    for h in range(1, 4):

        for i, grid_name in grid_map.items():
            create_merged_lambdamart_dataset(
                config_id=str(i),
                grid_name=grid_name,
                query_root=f"embeddings_phonetic_test/queries/queries_test_{h}",
                grid_root=f"embeddings_phonetic/grids_test_{h}",
                # feature_types = [ "mfcc", "mel_band_mid", "mel_band_low", "mel_band_high"],
                feature_types = [ "mfcc", "mid", "low", "high"],
                quantile_dir="phonetic_cosine_model_quantiles",
                output_path=f"output_phonetic_final_test/lambdamart_cosine_dataset_known_queries_normed{h}/{grid_name}",
                fraction=1
            )