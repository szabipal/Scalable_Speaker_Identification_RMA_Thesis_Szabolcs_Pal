
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def find_best_threshold(df, mode):
    """
    Select best threshold based on max precision where TP > 20% of total positives
    """
    label_col = 'label'
    ent_col = 'normalized_entropy'
    true_label = 1 if mode == "known" else 0

    thr_min, thr_max = df[ent_col].min(), df[ent_col].max()
    rng = np.arange(thr_max, thr_min, -0.01) if mode == "known" else np.arange(thr_min, thr_max, 0.01)

    total_true = (df[label_col] == true_label).sum()
    best_threshold = None
    best_precision = 0
    best_recall = 0

    for t in rng:
        if mode == "known":
            pred = (df[ent_col] < t).astype(int)
        else:
            pred = (df[ent_col] > t).astype(int)

        actual = (df[label_col] == true_label).astype(int)
        tp = ((pred == 1) & (actual == 1)).sum()
        fp = ((pred == 1) & (actual == 0)).sum()
        fn = ((pred == 0) & (actual == 1)).sum()

        if tp < total_true * 0.2:
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision > best_precision:
            best_threshold = t
            best_precision = precision
            best_recall = recall
            print(best_threshold)
            print(best_precision)
            print(best_recall)

    return best_threshold, best_precision, best_recall


def threshold_transformers_single_file_v2(
    known_csv,
    unknown_csv,
    out_csv="transformers_threshold.csv"
):
    known_df = pd.read_csv(known_csv)
    unknown_df = pd.read_csv(unknown_csv)

    results = []

    # Group known by source_file and then by label
    known_grouped = defaultdict(dict)
    for fname, group in known_df.groupby("source_file"):
        for label, sub in group.groupby("label"):
            known_grouped[fname][label] = sub

    # Group unknown by source_file
    for fname, group_u in unknown_df.groupby("source_file"):
        if fname not in known_grouped or 1 not in known_grouped[fname] or 0 not in known_grouped[fname]:
            continue

        # Balanced known
        known_balanced = pd.concat([
            known_grouped[fname][0].sample(min(len(known_grouped[fname][0]), len(known_grouped[fname][1])), random_state=42),
            known_grouped[fname][1].sample(min(len(known_grouped[fname][0]), len(known_grouped[fname][1])), random_state=42)
        ])
        known_balanced['label'] = known_balanced['label'].astype(int)

        # Best threshold for known
        t_k, p_k, r_k = find_best_threshold(known_balanced.copy(), mode="known")

        # Prepare unknown data
        group_u = group_u.copy()
        group_u['label'] = 1
        sample_size = min(len(group_u), len(known_grouped[fname][0]))
        known_sample = known_grouped[fname][0].sample(sample_size, random_state=42).copy()
        group_u = group_u.sample(sample_size, random_state=42).copy()
        known_sample['label'] = 0
        combined_df = pd.concat([group_u, known_sample], ignore_index=True)

        # Best threshold for unknown
        t_u, p_u, r_u = find_best_threshold(combined_df.copy(), mode="unknown")

        results.append({
            "config": fname,
            "known_threshold": round(t_k, 4) if t_k is not None else None,
            "known_train_precision": round(p_k, 3),
            "known_train_recall": round(r_k, 3),
            "unknown_threshold": round(t_u, 4) if t_u is not None else None,
            "unknown_train_precision": round(p_u, 3),
            "unknown_train_recall": round(r_u, 3)
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_csv, index=False)
    print(f"[✓] Threshold results saved to {out_csv}")


# === USAGE ===
threshold_transformers_single_file_v2(
    known_csv="TRANSFORMERS_BASED_MODEL/entropy_transformers_known_dev_new.csv",
    unknown_csv="TRANSFORMERS_BASED_MODEL/entropy_transformers_unknown_dev_new.csv",
    out_csv="TRANSFORMERS_BASED_MODEL/transformers_threshold_new2.csv"
)
