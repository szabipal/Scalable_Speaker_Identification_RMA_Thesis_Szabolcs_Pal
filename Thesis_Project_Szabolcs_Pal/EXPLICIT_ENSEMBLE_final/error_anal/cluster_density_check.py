import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import pairwise_distances
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import combinations


RESULTS_FILE = Path("cluster_density_results.csv")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "cluster_density_metrics_results_phonetic.csv"
def compute_cluster_density(embeddings, labels):
    """Compute average cluster density as mean pairwise distance within clusters."""
    unique_labels = np.unique(labels)
    densities = []
    for lbl in unique_labels:
        cluster_points = embeddings[labels == lbl]
        if len(cluster_points) > 1:
            dists = np.linalg.norm(cluster_points[:, None] - cluster_points, axis=2)
            mean_dist = np.sum(dists) / (len(cluster_points) * (len(cluster_points) - 1))
            densities.append(mean_dist)
    return np.mean(densities) if densities else np.nan

def analyze_cluster_density(base_path):
    """Traverse structure: base/config/model/speaker/*.npy and compute density."""
    results = []
    base_path = Path(base_path)

    config_folders = [f for f in base_path.iterdir() if f.is_dir()]
    for config_folder in tqdm(config_folders, desc=f"Processing {base_path.name}"):
        try:
            num_speakers = int(config_folder.name.split("_")[0].replace("s", ""))
        except:
            num_speakers = None

        try:
            num_instances = int(config_folder.name.split("_")[1].replace("i", ""))
        except:
            num_instances = None

        model_folders = [m for m in config_folder.iterdir() if m.is_dir()]
        for model_folder in model_folders:
            embeddings_list = []
            labels_list = []

            for idx, speaker_folder in enumerate(model_folder.iterdir()):
                if not speaker_folder.is_dir():
                    continue
                for file in speaker_folder.glob("*.npy"):
                    emb = np.load(file)
                    embeddings_list.append(emb)
                    labels_list.append(idx)

            if not embeddings_list:
                continue

            embeddings = np.vstack(embeddings_list)
            labels = np.array(labels_list)

            density_score = compute_cluster_density(embeddings, labels)

            results.append({
                "model": model_folder.name,
                "config": config_folder.name,
                "num_speakers": num_speakers,
                "num_instances":num_instances,
                "avg_density": density_score
            })

    return pd.DataFrame(results)




def plot_cluster_density_sp(df):
    """Plot density trend for all models across speaker counts."""
    plt.figure(figsize=(10, 6))
    for model in df['model'].unique():
        sub = df[df['model'] == model]
        sub = sub.groupby("num_speakers")["avg_density"].mean().reset_index()
        plt.plot(sub["num_speakers"], sub["avg_density"], marker='o', label=model)

    plt.xlabel("Number of Speakers")
    plt.ylabel("Average Cluster Density")
    plt.title("Cluster Density vs Number of Speakers")
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_cluster_density_in(df):
    """Plot density trend for all models across instance counts."""
    plt.figure(figsize=(10, 6))
    for model in df['model'].unique():
        sub = df[df['model'] == model]
        sub = sub.groupby("num_instances")["avg_density"].mean().reset_index()
        plt.plot(sub["num_instances"], sub["avg_density"], marker='o', label=model)

    plt.xlabel("Number of Instances per Speaker")
    plt.ylabel("Average Cluster Density")
    plt.title("Cluster Density vs Number of Instances per Speaker")
    plt.legend()
    plt.grid(True)
    plt.show()

if RESULTS_FILE.exists():
    print(f"✅ Found existing results file: {RESULTS_FILE}, loading instead of recomputing...")
    all_results = pd.read_csv(RESULTS_FILE)
    plot_cluster_density_sp(all_results)
    plot_cluster_density_in(all_results)
else:
    print("⏳ Computing new cluster density metrics...")
    all_results = pd.DataFrame()
    # paths = [
    #     'embeddings_spectral_final/grids_test_1',
    #     'embeddings_spectral_final/grids_test_2',
    #     'embeddings_spectral_final/grids_test_3'
    # ]
    paths = [
        'embeddings_phonetic/grids_test_1',
        'embeddings_phonetic/grids_test_2',
        'embeddings_phonetic/grids_test_3'
    ]
    for path in tqdm(paths, desc="Overall progress"):
        dataframe = analyze_cluster_density(path)
        all_results = pd.concat([all_results, dataframe], ignore_index=True)

    all_results.to_csv(RESULTS_FILE, index=False)
    print(f"💾 Results saved to {RESULTS_FILE}")

