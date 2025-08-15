import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# === CONFIG ===
model_path = "EXPLICIT_ENSEMBLE_final/phonetic_final_128/lambdamart_model_lr0.1_nl1023_ff1.0_bf0.8.txt"
eval_base_path = Path("output_phonetic_final_test")  # now loops over multiple merged folders like lambdamart_merged_unknown5
conf_out_dir = Path("EXPLICIT_ENSEMBLE_final/confidences_phonetic_unknown_test")
conf_out_dir.mkdir(parents=True, exist_ok=True)
cut_off=0.1532
# cut_off=2.7399
# === LOAD TRAINED MODEL ===
model = lgb.Booster(model_file=model_path)

# === LOOP OVER EVAL DATASETS ===
for eval_folder in eval_base_path.glob("lambdamart_cosine_dataset_unknown_queries_normed*"):
    if not eval_folder.is_dir():
        continue

    dataset_id = eval_folder.name.replace("lambdamart_cosine_dataset_unknown_queries_normed", "")  # e.g., 'unknown5'
    metrics = {}

    for grid_folder in eval_folder.iterdir():
        if not grid_folder.is_dir():
            continue

        try:
            parts = grid_folder.name.split("s_")
            s_val = int(parts[0])
            i_val = int(parts[1].replace("i", ""))
        except Exception as e:
            print(f"⚠️ Skipping {grid_folder.name} due to naming error: {e}")
            continue

        csv_path = grid_folder / f"lambdamart_merged_{grid_folder.name}.csv"
        if not csv_path.exists():
            print(f"⚠️ CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if df.empty or "label" not in df.columns or "qid" not in df.columns:
            print(f"⚠️ Invalid or empty CSV: {csv_path}")
            continue

        # Ensure all model features exist
        for col in model.feature_name():
            if col not in df.columns:
                df[col] = 0.0

        X_eval = df[model.feature_name()]
        df["pred"] = model.predict(X_eval)

        # Add normalized column using min-max normalization per qid group
        def normalize_group(group):
            scores = group["pred"].values.astype(float)
            scores = np.where(scores < cut_off, 0.01, scores)
            min_score = scores.min()
            max_score = scores.max()
            eps = 1e-8
            group["norm_pred"] = (scores - min_score) / (max_score - min_score + eps)
            return group




        # === Save confidence CSV ===
        confidence_df = df[["qid", "label", "pred", "candidate_speaker"]]

        # Ensure 'pred' is float (if it's formatted as a string)
        confidence_df["pred"] = confidence_df["pred"].astype(float)

        # Apply normalization per qid
        confidence_df = confidence_df.groupby("qid", group_keys=False).apply(normalize_group)


        out_name = f"confidences_{dataset_id}_{grid_folder.name}.csv"
        confidence_df.to_csv(conf_out_dir / out_name, index=False)

        # === Evaluation ===
        top_preds = df.sort_values(["qid", "pred"], ascending=[True, False]) \
                      .groupby("qid").head(1)

        tp = top_preds["label"].sum()
        fp = len(top_preds) - tp
        fn = df.groupby("qid")["label"].sum().sum() - tp
        tn = len(df) - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / len(df) if len(df) > 0 else 0

        metrics[(s_val, i_val)] = {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy
        }

    # === Create and Save Heatmaps + Metrics CSV ===
    if not metrics:
        continue

    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.index = pd.MultiIndex.from_tuples(metrics_df.index, names=["speakers", "instances"])
    metrics_df = metrics_df.reset_index()

    # Save metrics as CSV
    metrics_out_dir = Path(f"EXPLICIT_ENSEMBLE_final/phonetic_metrics_test_{dataset_id}")
    metrics_out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_out_dir / f"metrics_phonetic_known_{dataset_id}.csv", index=False)

    # Save heatmaps
    for metric in ["precision", "recall", "accuracy"]:
        pivot = metrics_df.pivot(index="speakers", columns="instances", values=metric)
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues")
        plt.title(f"{metric.capitalize()} Heatmap – {dataset_id}")
        plt.ylabel("Number of Speakers")
        plt.xlabel("Instances per Speaker")
        plt.tight_layout()
        plt.savefig(metrics_out_dir /f"{metric}_heatmap_phonetic_{dataset_id}.png")
        plt.close()
