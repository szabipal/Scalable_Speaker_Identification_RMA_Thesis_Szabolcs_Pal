import os
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "cluster_metrics_results_phonetic.csv"


def load_embeddings_from_folder(folder_path):
    embeddings = []
    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            emb = np.load(os.path.join(folder_path, file))
            embeddings.append(emb)
    return np.vstack(embeddings) if embeddings else np.empty((0,))


def compute_cluster_metrics(all_speakers_embeddings):
    centroids = {spk: np.mean(embs, axis=0) for spk, embs in all_speakers_embeddings.items() if embs.size > 0}

    intra_spread = []
    for spk, embs in all_speakers_embeddings.items():
        if embs.size == 0:
            continue
        centroid = centroids[spk]
        dists = np.linalg.norm(embs - centroid, axis=1)
        intra_spread.append(np.mean(dists))
    avg_intra = np.mean(intra_spread) if intra_spread else 0

    inter_dists = []
    for (spk1, c1), (spk2, c2) in combinations(centroids.items(), 2):
        inter_dists.append(np.linalg.norm(c1 - c2))
    avg_inter = np.mean(inter_dists) if inter_dists else 0

    separation_score = avg_inter / (avg_intra + 1e-6)
    return avg_inter, avg_intra, separation_score


def analyze_cluster_separation(base_path):
    results = []
    config_folders = [f for f in Path(base_path).iterdir() if f.is_dir()]

    for config_folder in tqdm(config_folders, desc=f"Processing {base_path} configs"):
        parts = config_folder.name.split("_")
        num_speakers = int(parts[0].replace("s", "")) if len(parts) > 0 else None
        num_instances = int(parts[1].replace("i", "")) if len(parts) > 1 else None

        model_folders = [m for m in config_folder.iterdir() if m.is_dir()]
        for model_folder in tqdm(model_folders, desc=f"   {config_folder.name}", leave=False):
            all_speakers_embeddings = {}
            X, labels = [], []
            for idx, speaker_folder in enumerate(model_folder.iterdir()):
                if not speaker_folder.is_dir():
                    continue
                embs = load_embeddings_from_folder(speaker_folder)
                if embs.size > 0:
                    all_speakers_embeddings[speaker_folder.name] = embs
                    X.append(embs)
                    labels.extend([idx] * len(embs))
            if len(all_speakers_embeddings) > 1:
                inter, intra, score = compute_cluster_metrics(all_speakers_embeddings)
                X_all = np.vstack(X)

                try:
                    sil_score = silhouette_score(X_all, labels)
                except:
                    sil_score = np.nan

                centroids = {spk: np.mean(embs, axis=0) for spk, embs in all_speakers_embeddings.items()}

                # Pre-stack "other" centroids for quick nearest-other lookup
                spk_list = list(all_speakers_embeddings.keys())
                centroid_mat = np.vstack([centroids[s] for s in spk_list])  # shape [S, D]

                overlap = 0
                total_points = 0

                for s_idx, spk in enumerate(spk_list):
                    embs = all_speakers_embeddings[spk]
                    if embs.size == 0:
                        continue

                    # distances to own centroid
                    d_self = np.linalg.norm(embs - centroids[spk], axis=1)  # shape [N_spk]

                    # distances to all centroids, then mask out own centroid
                    # (vectorized; no per-point loops)
                    d_to_all = np.linalg.norm(embs[:, None, :] - centroid_mat[None, :, :], axis=2)  # [N_spk, S]
                    d_to_all[:, s_idx] = np.inf  # ignore own centroid
                    d_other = np.min(d_to_all, axis=1)  # nearest other centroid per embedding

                    # count overlaps: strictly closer to another centroid than own
                    overlap += int(np.sum(d_other < d_self))
                    total_points += embs.shape[0]

                # Sanity checks
                if total_points == 0:
                    overlap_score = 0.0
                else:
                    # This assert guarantees no score > 1 unless there's a logic bug upstream
                    assert overlap <= total_points, (
                        f"Overlap count ({overlap}) exceeds total embeddings ({total_points}). "
                        "Check for double counting or filtering mistakes."
                    )
                    overlap_score = overlap / total_points

                # Numerical safety + debugging aid
                if not (0.0 <= overlap_score <= 1.0):
                    print(
                        f"[WARN] overlap_score out of bounds: {overlap_score:.4f} "
                        f"(overlap={overlap}, total_points={total_points}, "
                        f"config={config_folder.name}, model={model_folder.name})"
                    )
                    overlap_score = max(0.0, min(1.0, overlap_score))  # clip as last resort
                # --- end overlap computation ---

                results.append({
                    "config": config_folder.name,
                    "model": model_folder.name,
                    "num_speakers": num_speakers,
                    "instances_per_speaker": num_instances,
                    "avg_inter_cluster_dist": inter,
                    "avg_intra_cluster_spread": intra,
                    "separation_score": score,
                    "overlap_score": overlap_score,
                    "silhouette_score": sil_score
                })
    return pd.DataFrame(results)


def plot_cluster_metrics_improved(df):
    metrics = ["separation_score", "overlap_score", "silhouette_score"]
    sns.set(style="whitegrid")

    # 1️⃣ Improved line plots with aggregation
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for model_name, group in df.groupby("model"):
            mean_values = group.groupby("num_speakers")[metric].mean()
            std_values = group.groupby("num_speakers")[metric].std()
            plt.plot(mean_values.index, mean_values, marker='o', label=model_name)
            plt.fill_between(mean_values.index,
                             mean_values - std_values,
                             mean_values + std_values,
                             alpha=0.2)
        if metric == "separation_score":
            plt.yscale("log")
        if metric == "silhouette_score":
            plt.axhline(0, color="red", linestyle="--", linewidth=1)
        plt.title(f"{metric.replace('_', ' ').title()} vs Number of Speakers")
        plt.xlabel("Number of Speakers")
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.tight_layout()
        plt.show()

    # scatter plot
    plt.figure(figsize=(10, 6))
    for model_name, group in df.groupby("model"):
        plt.scatter(group["separation_score"], group["overlap_score"], alpha=0.4, label=model_name)
    plt.xscale("log")
    plt.title("Cluster Overlap vs Separation Score")
    plt.xlabel("Separation Score (log scale)")
    plt.ylabel("Overlap Score")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---- MAIN EXECUTION ----
if RESULTS_FILE.exists():
    print(f"Found existing results file: {RESULTS_FILE}, loading instead of recomputing...")
    all_results = pd.read_csv(RESULTS_FILE)
else:
    print("⏳ Computing new cluster metrics...")
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
        dataframe = analyze_cluster_separation(path)
        all_results = pd.concat([all_results, dataframe], ignore_index=True)

    #  Save results
    all_results.to_csv(RESULTS_FILE, index=False)
    print(f" Results saved to {RESULTS_FILE}")

plot_cluster_metrics_improved(all_results)
