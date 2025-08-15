

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_entropy_thresholds(known_df, unknown_df, threshold_csv, return_full_df=False):
    thresholds = pd.read_csv(threshold_csv).set_index("config")
    results = []
    full_data = []

    known_df["true"] = 1
    known_df["label"] = ((known_df["query_id"].str.split('_').str[0].astype(int) == known_df["top_match_id"].astype(int))).astype(int)

    # For unknown_df: label is -1 as is
    unknown_df["true"] = -1
    unknown_df["label"] = unknown_df["true"]

    all_configs = sorted(set(known_df["source_file"].unique()) &
                         set(unknown_df["source_file"].unique()) &
                         set(thresholds.index))

    for config in all_configs:
        df_known = known_df[known_df["source_file"] == config].copy()
        df_unknown = unknown_df[unknown_df["source_file"] == config].copy()
        df = pd.concat([df_known, df_unknown], ignore_index=True)
        df["config"] = config  # explicitly add config for clarity

        known_thresh = thresholds.loc[config]["known_threshold"]
        unknown_thresh = thresholds.loc[config]["unknown_threshold"]

        def classify(ent):

            # if known_thresh > unknown_thresh:
            #     if ent > known_thresh:
            #         return 1
            #     elif ent < unknown_thresh:
            #         return -1
            #     else:
            #         return 0
            # else:
            if ent < known_thresh:
                return 1
            elif ent > unknown_thresh:
                return -1
            else:
                return 0

        df["pred"] = df["normalized_entropy"].apply(classify)

        full_data.append(df)

        tp_k = ((df["pred"] == 1) & (df["label"] == 1)).sum()
        fp_k = ((df["pred"] == 1) & (df["label"] != 1)).sum()
        fn_k = ((df["pred"] != 1) & (df["label"] == 1)).sum()

        precision_known = tp_k / (tp_k + fp_k) if (tp_k + fp_k) > 0 else 0
        recall_known = tp_k / (tp_k + fn_k) if (tp_k + fn_k) > 0 else 0

        tp_u = ((df["pred"] == -1) & (df["label"] == -1)).sum()
        fp_u = ((df["pred"] == -1) & (df["label"] != -1)).sum()
        fn_u = ((df["pred"] != -1) & (df["label"] == -1)).sum()

        precision_unknown = tp_u / (tp_u + fp_u) if (tp_u + fp_u) > 0 else 0
        recall_unknown = tp_u / (tp_u + fn_u) if (tp_u + fn_u) > 0 else 0

        abstention_ratio = (df["pred"] == 0).sum() / len(df)

        results.append({
            "config": config,
            "precision_known": precision_known,
            "recall_known": recall_known,
            "precision_unknown": precision_unknown,
            "recall_unknown": recall_unknown,
            "abstention_ratio": abstention_ratio
        })

    results_df = pd.DataFrame(results)

    if return_full_df:
        combined = pd.concat(full_data, ignore_index=True)
        return results_df, combined
    else:
        return results_df


# def evaluate_top1_known(known_df):
#     results = []
#     known_df["pred"] = known_df["label"]
#     for config in known_df["source_file"].unique():
#         df = known_df[known_df["source_file"] == config]
#
#         if "query_id" not in df.columns or "pred" not in df.columns or "label" not in df.columns:
#             continue
#
#         top_preds = df.sort_values(["query_id", "pred"], ascending=[True, False]) \
#                       .groupby("query_id").head(1)
#
#         y_true = top_preds["label"]
#         y_pred = [1] * len(top_preds)
#
#         precision = precision_score(y_true, y_pred, zero_division=0)
#         recall = recall_score(y_true, y_pred, zero_division=0)
#         f1 = f1_score(y_true, y_pred, zero_division=0)
#
#         results.append({
#             "config": config,
#             "top1_precision": precision,
#             "top1_recall": recall,
#             "top1_f1": f1,
#             'grid_number':grid_number
#         })
#     print(pd.DataFrame(results).columns)
#     return pd.DataFrame(results)
def evaluate_top1_known(known_df):
    results = []
    known_df["pred"] = known_df["label"]

    # Group by both config and grid_number
    for (config, grid_num), df in known_df.groupby(["source_file", "grid_number"]):

        if "query_id" not in df.columns or "pred" not in df.columns or "label" not in df.columns:
            continue

        top_preds = (
            df.sort_values(["query_id", "pred"], ascending=[True, False])
              .groupby("query_id")
              .head(1)
        )

        y_true = top_preds["label"]
        y_pred = [1] * len(top_preds)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            "config": config,
            "grid_number": grid_num,
            "top1_precision": precision,
            "top1_recall": recall,
            "top1_f1": f1
        })

    return pd.DataFrame(results)
# def plot_heatmap(df, metric, outdir):
#     if "source_file" in df.columns:
#         df[['speakers', 'instances']] = df["source_file"].str.extract(r'(\d+)s_(\d+)i').astype(int)
#     elif "config" in df.columns:
#         df[['speakers', 'instances']] = df["config"].str.extract(r'(\d+)s_(\d+)i').astype(int)
#     else:
#         raise ValueError("No valid column ('source_file' or 'config') found for extracting config information.")
#
#     heatmap_data = df.pivot(index="speakers", columns="instances", values=metric)
#
#     plt.figure(figsize=(10, 6))
#     sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5)
#     plt.title(metric.replace("_", " ").title())
#     plt.xlabel("Instances per Speaker")
#     plt.ylabel("Number of Speakers")
#     plt.tight_layout()
#     plt.savefig(Path(outdir) / f"{metric}_heatmap.png")
#     plt.close()
def plot_heatmap(df, metric, outdir):
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path

    os.makedirs(outdir, exist_ok=True)

    # ✅ Extract speaker/instance info from CONFIG, not grid_number
    df["config"] = df["config"].astype(str)
    extracted = df["config"].str.extract(r'(\d+)s_(\d+)i')

    # Remove rows that didn't match pattern
    extracted = extracted.dropna()
    extracted.columns = ["speakers", "instances"]
    extracted = extracted.astype(int)

    # Keep only matching rows
    df = df.loc[extracted.index].copy()
    df[["speakers", "instances"]] = extracted

    # Pivot for heatmap
    heatmap_data = df.pivot(index="speakers", columns="instances", values=metric)

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5)
    plt.title(metric.replace("_", " ").title())
    plt.xlabel("Instances per Speaker")
    plt.ylabel("Number of Speakers")
    plt.tight_layout()
    plt.savefig(Path(outdir) / f"{metric}_heatmap.png")
    plt.close()


def run_final_entropy_test(known_file, unknown_file, threshold_csv, outdir):
    os.makedirs(outdir, exist_ok=True)

    print("📥 Loading known entropy data...")
    known_df = pd.read_csv(known_file)

    print("📥 Loading unknown entropy data...")
    unknown_df = pd.read_csv(unknown_file)

    # === Threshold-based evaluation
    print("⚙️ Running threshold-based classification...")
    print(known_df.columns)
    threshold_df, full_results_df = evaluate_entropy_thresholds(
    known_df, unknown_df, threshold_csv, return_full_df=True
    )
    threshold_df.to_csv(Path(outdir) / "threshold_results.csv", index=False)
    full_results_df.to_csv(Path(outdir) / "thresholded_predictions.csv", index=False)
    print(threshold_df.columns)

    for metric in ["precision_known", "recall_known", "precision_unknown", "recall_unknown", "abstention_ratio"]:
        plot_heatmap(threshold_df, metric, outdir)

    # # === Top-1 known only evaluation
    # print("⚙️ Running raw Top-1 prediction accuracy (known only)...")
    # top1_df = evaluate_top1_known(known_df)
    # print(top1_df.columns)
    #
    # top1_df.to_csv(Path(outdir) / "top1_accuracy_known.csv", index=False)
    #
    # for metric in ["top1_precision", "top1_recall", "top1_f1"]:
    #     plot_heatmap(top1_df, metric, outdir)
    #
    # print(f"✅ All evaluation complete. Results saved to {outdir}")
        # === Top-1 known only evaluation
    # === Top-1 known only evaluation
    print("⚙️ Running raw Top-1 prediction accuracy (known only)...")
    top1_df = evaluate_top1_known(known_df)
    top1_df.to_csv(Path(outdir) / "top1_accuracy_known.csv", index=False)

    # ✅ Generate individual heatmaps per grid_number
    if "grid_number" in top1_df.columns:
        for grid_num in top1_df["grid_number"].unique():
            grid_df = top1_df[top1_df["grid_number"] == grid_num]
            for metric in ["top1_precision", "top1_recall", "top1_f1"]:
                plot_heatmap(
                    grid_df,
                    metric,
                    Path(outdir) / f"grid_{grid_num}"  # subfolder per grid
                )

    # ✅ Generate combined heatmaps (all grids together)
    for metric in ["top1_precision", "top1_recall", "top1_f1"]:
        plot_heatmap(top1_df, metric, outdir)

    print(f"✅ All evaluation complete. Results saved to {outdir}")


run_final_entropy_test(
    known_file="TRANSFORMERS_BASED_MODEL/entropy_transformers_known_test_new2.csv",
    unknown_file="TRANSFORMERS_BASED_MODEL/entropy_transformers_unknown_test_new2.csv",
    threshold_csv="TRANSFORMERS_BASED_MODEL/transformers_threshold_new2.csv",
    outdir="TRANSFORMERS_BASED_MODEL/final_eval_TEST_new3"
)
