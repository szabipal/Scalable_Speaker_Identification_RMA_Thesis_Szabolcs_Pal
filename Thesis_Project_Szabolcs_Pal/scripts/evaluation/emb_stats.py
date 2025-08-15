"""
quick diagnostics for speaker-embedding quality.

what it does
- samples cosine similarities for intra-/inter-speaker pairs and prints a compact summary.
- plots similarity distributions, computes a simple eer from cosine scores.
- makes small t-sne views for two disjoint speaker sets to sanity-check clustering.
- reports clustering stats (silhouette, davies–bouldin) and centroid-based distances.
- loads nested *.npy embeddings into a tidy dataframe for downstream analysis.

notes
- cosine is used as the similarity; higher is “closer”.
- eer here is derived from roc on cosine scores (1 = same speaker, 0 = different).
- t-sne is for visualization only; don’t over-interpret distances.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score
from tqdm import tqdm
from scipy.spatial.distance import cosine
import random



# --- Compute similarity scores ---
def compute_similarity_pairs(embeddings, speaker_ids, sample_size=5000):
    """
    sample intra-/inter-speaker cosine similarities.

    parameters
    ----------
    embeddings : array-like
        list/array of embedding vectors (len = n).
    speaker_ids : array-like
        per-embedding speaker label (len = n).
    sample_size : int
        number of random pairs to draw.

    returns
    -------
    intra, inter : np.ndarray, np.ndarray
        cosine similarities for same-speaker and different-speaker pairs.

    behavior
    - randomly draws pairs without replacement per sample.
    - prints a short summary (mean, std, quantiles) for both sets.
    """
    intra_similarities = []
    inter_similarities = []

    indices = np.arange(len(embeddings))
    np.random.shuffle(indices)
    
    for _ in tqdm(range(sample_size)):
        i, j = np.random.choice(indices, 2, replace=False)
        sim = 1 - cosine(embeddings[i], embeddings[j])
        if speaker_ids[i] == speaker_ids[j]:
            intra_similarities.append(sim)
        else:
            inter_similarities.append(sim)

    # Print statistical summary
    def summarize(name, values):
        print(f"\n Stats for {name} similarities:")
        print(f"  Mean       : {np.mean(values):.4f}")
        print(f"  Std Dev    : {np.std(values):.4f}")
        print(f"  Min        : {np.min(values):.4f}")
        print(f"  25%        : {np.percentile(values, 25):.4f}")
        print(f"  Median     : {np.median(values):.4f}")
        print(f"  75%        : {np.percentile(values, 75):.4f}")
        print(f"  Max        : {np.max(values):.4f}")
        print(f"  Count      : {len(values)}")

    summarize("Intra-speaker", intra_similarities)
    summarize("Inter-speaker", inter_similarities)
    return np.array(intra_similarities), np.array(inter_similarities)

# --- Plot cosine similarity distributions ---
def plot_similarity_distributions(intra, inter):
    """
    visualize intra-/inter-speaker cosine distributions.

    parameters
    ----------
    intra, inter : array-like
        cosine similarity arrays (output of compute_similarity_pairs).
    """
    plt.figure(figsize=(8, 5))
    sns.kdeplot(intra, label='Intra-speaker', fill=True)
    sns.kdeplot(inter, label='Inter-speaker', fill=True)
    plt.title("Cosine Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()

# --- Compute EER ---
def compute_eer(intra, inter):
    """
    compute a quick eer from cosine similarities.

    parameters
    ----------
    intra, inter : array-like
        cosine similarities for positive (same) and negative (different) pairs.

    returns
    -------
    eer : float
        equal error rate.
    eer_threshold : float
        score threshold at eer intersection.

    notes
    - builds an roc with labels 1 (same) and 0 (different) on the cosine scores.
    """
    scores = np.concatenate([intra, inter])
    labels = np.concatenate([np.ones_like(intra), np.zeros_like(inter)])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    print(f"EER: {eer:.4f}")
    return eer, eer_threshold

# --- t-SNE Plot ---
def plot_tsne_two_sets(embeddings, speaker_ids, n_speakers=10):
    """
    draw two small t-sne plots for disjoint speaker subsets.

    parameters
    ----------
    embeddings : array-like
        embedding vectors.
    speaker_ids : array-like
        speaker labels aligned with embeddings.
    n_speakers : int
        number of speakers per t-sne subset.

    behavior
    - samples 2 × n_speakers unique speakers.
    - runs t-sne per subset (same settings) and plots colored points by speaker.
    """
    df = pd.DataFrame({
        'embedding': list(embeddings),
        'speaker_id': speaker_ids
    })

    all_speakers = df['speaker_id'].unique()
    if len(all_speakers) < 2 * n_speakers:
        raise ValueError("Not enough unique speakers to sample two sets.")

    # Randomly select two non-overlapping sets of speakers
    selected = random.sample(list(all_speakers), 2 * n_speakers)
    set1 = selected[:n_speakers]
    set2 = selected[n_speakers:]

    def tsne_plot_for_speakers(selected_speakers, title):
        sub_df = df[df['speaker_id'].isin(selected_speakers)].reset_index(drop=True)
        emb_matrix = np.stack(sub_df['embedding'].values)
        speaker_labels = sub_df['speaker_id'].astype(str)

        tsne = TSNE(n_components=2, perplexity=20, n_iter=500, init='pca', method='barnes_hut', random_state=42)
        reduced = tsne.fit_transform(emb_matrix)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=speaker_labels, palette='tab10', s=30, alpha=0.8)
        plt.title(title)
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()
        plt.close()

    tsne_plot_for_speakers(set1, "t-SNE of 10 Random Speakers – Set 1")
    tsne_plot_for_speakers(set2, "t-SNE of 10 Random Speakers – Set 2")


# --- Silhouette Score & DBI ---
def clustering_stats(embeddings, speaker_ids):
    """
    report simple clustering measures.

    parameters
    ----------
    embeddings : array-like
        embedding vectors (2d array).
    speaker_ids : array-like
        labels for each embedding.

    returns
    -------
    sil_score : float
    dbi_score : float

    notes
    - higher silhouette is better; lower davies–bouldin is better.
    """
    sil_score = silhouette_score(embeddings, speaker_ids)
    dbi_score = davies_bouldin_score(embeddings, speaker_ids)
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Davies–Bouldin Index: {dbi_score:.4f}")
    return sil_score, dbi_score

# --- Centroid Analysis ---
def inter_intra_centroid_distances(embeddings, speaker_ids):
    """
    compare intra-speaker scatter vs inter-speaker centroid separation.

    parameters
    ----------
    embeddings : array-like
        embedding vectors.
    speaker_ids : array-like
        per-embedding speaker labels.

    returns
    -------
    intra_mean : float
        average distance from samples to their own centroid.
    inter_mean : float
        average distance between speaker centroids.

    notes
    - prints the inter/intra ratio (higher suggests cleaner separation).
    """
    from collections import defaultdict
    speaker_to_embeddings = defaultdict(list)
    
    for emb, sid in zip(embeddings, speaker_ids):
        speaker_to_embeddings[sid].append(emb)
    
    centroids = {sid: np.mean(speaker_to_embeddings[sid], axis=0) for sid in speaker_to_embeddings}
    
    intra_dists = []
    for sid, emb_list in speaker_to_embeddings.items():
        centroid = centroids[sid]
        for emb in emb_list:
            intra_dists.append(np.linalg.norm(emb - centroid))

    inter_dists = []
    centroid_list = list(centroids.values())
    for i in range(len(centroid_list)):
        for j in range(i + 1, len(centroid_list)):
            inter_dists.append(np.linalg.norm(centroid_list[i] - centroid_list[j]))

    intra_mean = np.mean(intra_dists)
    inter_mean = np.mean(inter_dists)
    print(f"Avg Intra-speaker Distance: {intra_mean:.4f}")
    print(f"Avg Inter-speaker Distance: {inter_mean:.4f}")
    print(f"Inter/Intra Ratio: {inter_mean / intra_mean:.4f}")
    return intra_mean, inter_mean


import os
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import glob

def load_embeddings_to_df(directory: str) -> pd.DataFrame:
    """
    load nested *.npy embeddings into a dataframe.

    parameters
    ----------
    directory : str
        root directory containing speaker/session folders with *.npy files.

    expectations
    ------------
    filenames look like: speaker-session-sample_chunk.npy


    returns
    -------
    pandas.DataFrame with columns:
      ['speaker_id','session_id','sample_id','chunk_id','embedding']

    behavior
    - walks subfolders, parses ids from the filename, and loads the array.
    - skips files that don’t match the expected pattern.
    """
    data = []
    npy_files = glob.glob(os.path.join(directory, "**/*.npy"), recursive=True)

    pattern = re.compile(r"(\d+)-(\d+)-(\d+)_chunk(\d+)\.npy$")

    for fpath in tqdm(npy_files, desc="Loading embeddings"):
        fname = os.path.basename(fpath)
        match = pattern.match(fname)
        if not match:
            print(f"Skipping unrecognized file format: {fname}")
            continue

        speaker_id, session_id, sample_id, chunk_id = match.groups()
        embedding = np.load(fpath)

        data.append({
            'speaker_id': speaker_id,
            'session_id': session_id,
            'sample_id': sample_id,
            'chunk_id': int(chunk_id),
            'embedding': embedding
        })

    return pd.DataFrame(data)
