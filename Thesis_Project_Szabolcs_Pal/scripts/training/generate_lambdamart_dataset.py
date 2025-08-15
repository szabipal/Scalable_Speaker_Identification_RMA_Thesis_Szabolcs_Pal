"""
build a multi-feature lambdamart training table from per-speaker embeddings.

what this script does
- walks a grid-style embedding layout and loads vectors for multiple feature types (e.g., mfcc, mel bands).
- for each query (picked by holding out one vector from a speaker), scores every enrolled speaker with a small
  set of distances and margins (euclidean to samples and centroid, mahalanobis, silhouette-like margin).
- standardizes distances per query so features are roughly comparable.
- concatenates features from all requested types into one row per (query, enrolled_speaker) and writes a csv.

expected layout
- embeddings/
    grids_1/
      <config_dir>/
        <speaker_id>/
          mfcc/*.npy
          mel_band_low/*.npy
          ...
    grids_2/...
    ...

inputs (cli)
- --features: names of feature folders to include (e.g. "mfcc mel_band_low").
- --output: path to write the joint dataset csv.

output
- one csv with columns:
  - speaker metadata: speaker_id, label, query_id
  - per-feature block: <feat>_mean_distance, <feat>_centroid_distance, <feat>_min_distance,
                       <feat>_boundary_margin, <feat>_mahalanobis, <feat>_silhouette_score,
                       <feat>_margin_to_avg
notes
- distances are z-scored within the current query (simple per-query normalization).
- mahalanobis uses pinv of covariance for robustness; falls back to inf if ill-conditioned.
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist


def compute_lambdamart_features(test_embedding, enrolled_clusters, true_speaker_id, intra_speaker_dist, inter_speaker_dist, feature_prefix):
    """
    build one query block: distances from a held-out query embedding to each enrolled speaker.

    params
    - test_embedding: 1d np array for the query.
    - enrolled_clusters: dict[speaker_id -> (n, d) array] enrolled vectors per speaker.
    - true_speaker_id: id string of the query speaker (for label).
    - intra_speaker_dist, inter_speaker_dist: rough averages used to form a simple margin_to_avg feature.
    - feature_prefix: short name to prefix feature columns (e.g., 'mfcc').

    returns
    - df with one row per enrolled speaker and standardized distance features for this query.
    """
    features = []
    query_id = np.random.randint(1000000)
    raw_features = []

    for speaker_id, cluster in enrolled_clusters.items():
        # basic distances to the cluster
        dists = cdist([test_embedding], cluster, metric='euclidean')[0]
        centroid = np.mean(cluster, axis=0)
        centroid_dist = np.linalg.norm(test_embedding - centroid)
        min_dist = np.min(dists)
        mean_dist = np.mean(dists)

        # mahalanobis to centroid (pinv for stability)
        try:
            cov_inv = np.linalg.pinv(np.cov(cluster.T))
            mahalanobis_dist = np.sqrt((test_embedding - centroid) @ cov_inv @ (test_embedding - centroid).T)
        except np.linalg.LinAlgError:
            mahalanobis_dist = float("inf")

        # simple boundary/margin proxy (spread within class)
        boundary_margin = mean_dist - np.max(dists)

        # silhouette-like score: separation to nearest other centroid vs own centroid distance
        closest_other = min([
            np.linalg.norm(test_embedding - np.mean(other, axis=0))
            for other_id, other in enrolled_clusters.items() if other_id != speaker_id
        ])
        silhouette_score = (closest_other - centroid_dist) / max(centroid_dist, 1e-6)

        raw_features.append({
            "speaker_id": speaker_id,
            "label": int(speaker_id == true_speaker_id),
            "query_id": query_id,
            f"{feature_prefix}_mean_distance": mean_dist,
            f"{feature_prefix}_centroid_distance": centroid_dist,
            f"{feature_prefix}_min_distance": min_dist,
            f"{feature_prefix}_boundary_margin": boundary_margin,
            f"{feature_prefix}_mahalanobis": mahalanobis_dist,
            f"{feature_prefix}_silhouette_score": silhouette_score,
            # coarse normalization using global intra/inter estimates
            f"{feature_prefix}_margin_to_avg": (mean_dist - intra_speaker_dist) / (inter_speaker_dist - intra_speaker_dist + 1e-6)
        })

    # z-score the distance-like features within this query so scales are aligned
    df_raw = pd.DataFrame(raw_features)
    distance_cols = [f"{feature_prefix}_" + col for col in [
        "mean_distance", "centroid_distance", "min_distance",
        "boundary_margin", "mahalanobis", "silhouette_score"
    ]]
    for col in distance_cols:
        mean = df_raw[col].mean()
        std = df_raw[col].std() + 1e-6
        df_raw[col] = (df_raw[col] - mean) / std

    return df_raw


def extract_embeddings_from_grid(grid_path, feature_type):
    """
    read all .npy embeddings for a given feature type under a single grid/config.

    params
    - grid_path: path to a single config directory containing per-speaker folders.
    - feature_type: subfolder name to read (e.g., 'mfcc').

    returns
    - dict[speaker_id -> (n, d) ndarray] with stacked embeddings per speaker.
    """
    enrolled_clusters = {}
    for speaker_dir in os.listdir(grid_path):
        speaker_path = os.path.join(grid_path, speaker_dir)
        if not os.path.isdir(speaker_path):
            continue
        embeddings = []
        feature_path = os.path.join(speaker_path, feature_type)
        if not os.path.exists(feature_path):
            continue
        for file in os.listdir(feature_path_
