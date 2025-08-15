import numpy as np
from pathlib import Path
import os

def compute_global_mean_std(
    feature_dir,
    mean_save_path=None,
    std_save_path=None,
    verbose=True
):
    """
    Computes global mean and std of all feature vectors in a directory.

    Args:
        feature_dir (str or Path): Directory containing .npy feature files.
        mean_save_path (str): Optional file path to save the mean.
        std_save_path (str): Optional file path to save the std.
        verbose (bool): Whether to print diagnostics.

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean, std
    """
    feature_vectors = []
    max_dims = 0

    for file in Path(feature_dir).rglob("*.npy"):
        try:
            vec = np.load(file).reshape(-1)
            feature_vectors.append(vec)
            max_dims = max(max_dims, vec.shape[0])
        except Exception as e:
            if verbose:
                print(f"⚠️ Skipping {file} due to loading error: {e}")

    if not feature_vectors:
        raise ValueError("❌ No valid feature vectors found in the directory.")

    # Pad vectors to same length (NaN-fill)
    padded_vectors = []
    for vec in feature_vectors:
        if vec.shape[0] < max_dims:
            pad = np.full((max_dims - vec.shape[0],), np.nan)
            vec = np.concatenate([vec, pad])
        padded_vectors.append(vec)

    feature_matrix = np.vstack(padded_vectors)

    # NaN diagnostics
    nan_counts = np.isnan(feature_matrix).sum(axis=0)
    nan_dimensions = [i for i, count in enumerate(nan_counts) if count > 0]

    if verbose:
        if nan_dimensions:
            print(f"⚠️ NaNs found in dimensions: {nan_dimensions}")
            print(f"🔢 NaN counts per dimension:\n{nan_counts}")
        else:
            print("✅ No NaNs found in any dimension.")

    # Safe stats computation
    mean = np.nanmean(feature_matrix, axis=0)
    std = np.nanstd(feature_matrix, axis=0)

    if mean_save_path:
        np.save(mean_save_path, mean)
        if verbose:
            print(f"💾 Saved mean to {mean_save_path}")
    if std_save_path:
        np.save(std_save_path, std)
        if verbose:
            print(f"💾 Saved std to {std_save_path}")

    if verbose:
        print("✅ Global stats computed.")
        print(f"📊 Mean: {mean}")
        print(f"📈 Std: {std}")

    return mean, std
