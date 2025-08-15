"""
utilities for checking similarities and centroid distances between speaker embeddings.
the functions here help you measure how similar or different embeddings are,
and then save simple summaries (quantiles) of those results.

the module is safe to import – nothing runs until you actually call a function.
"""

from pathlib import Path
import os
import json
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine


def compute_similarity_pairs(embeddings, speaker_ids, sample_size: int = 5000):
    """
    compare embeddings in random pairs and separate the results into:
    - intra-speaker similarities: pairs from the same speaker
    - inter-speaker similarities: pairs from different speakers

    parameters
    ----------
    embeddings : array-like
        the list or array of embedding vectors.
    speaker_ids : list
        the speaker id for each embedding, same length as embeddings.
    sample_size : int
        how many random pairs to check.

    returns
    -------
    intra : array
        cosine similarities for same-speaker pairs.
    inter : array
        cosine similarities for different-speaker pairs.
    """
    intra_similarities = []
    inter_similarities = []

    indices = np.arange(len(embeddings))
    np.random.shuffle(indices)

    from tqdm.auto import tqdm

    for _ in tqdm(range(sample_size), desc="computing similarities"):
        i, j = np.random.choice(indices, 2, replace=False)
        sim = 1 - cosine(embeddings[i], embeddings[j])

        if speaker_ids[i] == speaker_ids[j]:
            intra_similarities.append(sim)
        else:
            inter_similarities.append(sim)

    return np.array(intra_similarities), np.array(inter_similarities)


def compute_centroid_distances(embeddings, speaker_ids):
    """
    measure how close each embedding is to its own speaker's average vector (centroid),
    and how close speaker centroids are to each other.

    returns two arrays:
    - intra-centroid: similarities within the same speaker
    - inter-centroid: similarities between different speakers
    """
    speaker_to_embeddings = defaultdict(list)
    for emb, sid in zip(embeddings, speaker_ids):
        speaker_to_embeddings[sid].append(emb)

    centroids = {sid: np.mean(embs, axis=0) for sid, embs in speaker_to_embeddings.items()}

    intra_dists = []
    for sid, embs in speaker_to_embeddings.items():
        c = centroids[sid]
        for emb in embs:
            intra_dists.append(1 - cosine(emb, c))

    inter_dists = []
    clist = list(centroids.values())
    for i in range(len(clist)):
        for j in range(i + 1, len(clist)):
            inter_dists.append(1 - cosine(clist[i], clist[j]))

    return np.array(intra_dists), np.array(inter_dists)


def save_quantiles_to_json(feature_name,
                           intra_sim: np.ndarray,
                           inter_sim: np.ndarray,
                           intra_centroid: np.ndarray,
                           inter_centroid: np.ndarray,
                           output_dir: str):
    """
    save the 10th and 90th percentile values for each similarity type into a json file.
    the result is a small summary of the overall distribution.

    file is named like: quantiles_<feature_name>.json
    """
    os.makedirs(output_dir, exist_ok=True)

    quantiles = {
        "intra": {
            "cosine": {
                "q10": float(np.percentile(intra_sim, 10)),
                "q90": float(np.percentile(intra_sim, 90)),
            },
            "centroid": {
                "q10": float(np.percentile(intra_centroid, 10)),
                "q90": float(np.percentile(intra_centroid, 90)),
            },
        },
        "inter": {
            "cosine": {
                "q10": float(np.percentile(inter_sim, 10)),
                "q90": float(np.percentile(inter_sim, 90)),
            },
            "centroid": {
                "q10": float(np.percentile(inter_centroid, 10)),
                "q90": float(np.percentile(inter_centroid, 90)),
            },
        },
    }

    feat = str(feature_name).split("/")[-1]
    out = os.path.join(output_dir, f"quantiles_{feat}.json")

    with open(out, "w") as f:
        json.dump(quantiles, f, indent=4)

    print(f"✅ saved quantiles to {out}")
    return out


__all__ = [
    "compute_similarity_pairs",
    "compute_centroid_distances",
    "save_quantiles_to_json",
    "compute_and_save_quantiles_for_root",
]


def compute_and_save_quantiles_for_root(embedding_root: str | Path,
                                        output_dir: str | Path):
    """
    go through each feature folder inside the given embeddings directory,
    calculate similarities and centroid distances, and save quantile summaries.

    skips any folders with 'grid' or 'queries' in the name.
    """
    from scripts.evaluation.emb_stats import load_embeddings_to_df

    embedding_root = Path(embedding_root)
    if not embedding_root.exists():
        raise FileNotFoundError(f"embedding root not found: {embedding_root}")

    features = [p for p in embedding_root.iterdir() if p.is_dir()]

    for path in features:
        pstr = str(path)
        if "grid" in pstr or "queries" in pstr:
            continue

        df = load_embeddings_to_df(path)

        intra_sim, inter_sim = compute_similarity_pairs(df["embedding"], df["speaker_id"])
        intra_c, inter_c = compute_centroid_distances(df["embedding"], df["speaker_id"])

        save_quantiles_to_json(path.name, intra_sim, inter_sim, intra_c, inter_c, str(output_dir))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="compute and save quantiles for similarities in embedding sets"
    )
    parser.add_argument("--root", required=True, help="path to the embeddings root folder")
    parser.add_argument("--out", required=True, help="folder to save quantiles json files")
    args = parser.parse_args()

    compute_and_save_quantiles_for_root(args.root, args.out)
