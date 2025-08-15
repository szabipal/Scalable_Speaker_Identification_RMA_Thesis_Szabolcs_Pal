"""
lambdamart inference + confidence/metric export.

what it does
- loads a trained lightgbm lambdamart model and runs inference on merged
  datasets (per grid config).
- writes per-qid confidence csvs and summary metrics (precision/recall/accuracy),
  including heatmaps.
- includes a batch helper that runs all (model_type × dataset_type × query_type)
  combos and moves outputs into the new structured layout.

notes
- expects inputs under: output_<model_type>_final_<split>/lambdamart_cosine_dataset_<query_type>_normed*/
- confidence csvs are normalized per qid via min–max; scores below `cut_off`
  are floored to 0.01 before normalization.
- final structured confidences go under:
  EXPLICIT_ENSEMBLE_final/confidences/<model>/<split>/<query>/
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] 
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

def run_lambdamart_inference(config):
    """
    run lambdamart (lightgbm) inference and export confidences + metrics.

    parameters
    ----------
    config : dict
        required:
          - model_type: 'phonetic' | 'spectral'
          - dataset_type: 'train' | 'test'
          - query_type: 'queries' | 'unknown_queries'
        optional:
          - cut_off: float threshold to floor low scores before min–max (default 0.1532)
          - model_path: path to trained model file; if absent, a default is used

    outputs
    -------
    - EXPLICIT_ENSEMBLE_final/confidences_{model_type}_{query_type}_{dataset_type}/confidences_*_*.csv
    - EXPLICIT_ENSEMBLE_final/{model_type}_metrics_{query_type}_{dataset_id}/
        * metrics csv + heatmaps for precision/recall/accuracy
    """
    model_type = config["model_type"]  # "phonetic" or "spectral"
    dataset_type = config["dataset_type"]  # "train" or "test"
    query_type = config["query_type"]  # "known" or "unknown"
    cut_off = config.get("cut_off", 0.1532)

    # === PATH CONFIGURATION ===
    model_dir = f"EXPLICIT_ENSEMBLE_final/{model_type}_final_128"
    model_path = config.get("model_path", model_dir + "/lambdamart_model_lr0.1_nl1023_ff1.0_bf0.8.txt")
    eval_base_path = Path(f"output_{model_type}_final_{dataset_type}")
    conf_out_dir = Path(f"EXPLICIT_ENSEMBLE_final/confidences_{model_type}_{query_type}_{dataset_type}")
    conf_out_dir.mkdir(parents=True, exist_ok=True)

    # === LOAD TRAINED MODEL ===
    model = lgb.Booster(model_file=model_path)

    # === LOOP OVER EVAL DATASETS ===
    pattern = f"lambdamart_cosine_dataset_{query_type}_normed*"
    for eval_folder in eval_base_path.glob(pattern):
        if not eval_folder.is_dir():
            continue

        dataset_id = eval_folder.name.replace(f"lambdamart_cosine_dataset_{query_type}_normed", "")
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
            confidence_df["pred"] = confidence_df["pred"].astype(float)
            confidence_df = confidence_df.groupby("qid", group_keys=False).apply(normalize_group)

            out_name = f"confidences_{dataset_id}_{grid_folder.name}.csv"
            confidence_df.to_csv(conf_out_dir / out_name, index=False)

            # === Evaluation ===
            top_preds = df.sort_values(["qid", "pred"], ascending=[True, False]).groupby("qid").head(1)

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

        metrics_out_dir = Path(f"EXPLICIT_ENSEMBLE_final/{model_type}_metrics_{query_type}_{dataset_id}")
        metrics_out_dir.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_out_dir / f"metrics_{model_type}_{query_type}_{dataset_id}.csv", index=False)

        for metric in ["precision", "recall", "accuracy"]:
            pivot = metrics_df.pivot(index="speakers", columns="instances", values=metric)
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues")
            plt.title(f"{metric.capitalize()} Heatmap – {dataset_id}")
            plt.ylabel("Number of Speakers")
            plt.xlabel("Instances per Speaker")
            plt.tight_layout()


def batch_generate_confidences(cutoff_value=0.1532):
    model_types = ["phonetic", "spectral"]
    dataset_types = ["train", "test"]
    query_types = ["queries", "unknown_queries"]

    for model in model_types:
        for dataset in dataset_types:
            for query in query_types:
                config = {
                    "model_type": model,
                    "dataset_type": dataset,
                    "query_type": query,
                    "cut_off": cutoff_value
                }

                print(f"\n Running inference for: {model} | {dataset} | {query}")
                run_lambdamart_inference(config)

                # Move confidence files to structured directory
                source_dir = Path(f"EXPLICIT_ENSEMBLE_final/confidences_{model}_{query}_{dataset}")
                if not source_dir.exists():
                    print(f"⚠️ Source directory missing: {source_dir}")
                    continue

                target_dir = Path(f"EXPLICIT_ENSEMBLE_final/confidences/{model}/{dataset}/{query}")
                target_dir.mkdir(parents=True, exist_ok=True)

                for file in source_dir.glob("*.csv"):
                    target_path = target_dir / file.name
                    file.rename(target_path)
                    print(f"✅ Moved: {file.name} → {target_path}")

                # Optionally remove empty old source dir
                if not any(source_dir.iterdir()):
                    source_dir.rmdir()