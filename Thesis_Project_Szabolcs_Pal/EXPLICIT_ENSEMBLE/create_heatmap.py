import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import lightgbm as lgb

# === CONFIG ===
model_path = "EXPLICIT_ENSEMBLE/spectral_trained/lambdamart_model_lr0.05_nl63_ff0.8_bf1.0.txt"
eval_base_path = Path("output_spectral/lambdamart_merged5")

# === LOAD TRAINED MODEL ===
model = lgb.Booster(model_file=model_path)

# === EVALUATION STORAGE ===
metrics = {}

# === LOOP OVER GRID FOLDERS ===
for grid_folder in eval_base_path.iterdir():
    if not grid_folder.is_dir():
        continue

    try:
        s_val = int(grid_folder.name.split('s_')[0])
        i_val = int(grid_folder.name.split('s_')[1].replace('i', ''))
    except Exception as e:
        print(f"Skipping {grid_folder.name}: {e}")
        continue

    csv_path = grid_folder / f"lambdamart_merged_{grid_folder.name}.csv"
    if not csv_path.exists():
        continue

    df = pd.read_csv(csv_path)
    if df.empty or "label" not in df.columns:
        continue

    feature_cols = [col for col in df.columns if col.startswith(("mfcc", "mel_band_mid", "mel_band_low", "mel_band_high"))]
    for col in model.feature_name():
        if col not in df.columns:
            df[col] = 0.0  # fill missing features with 0

    X_eval = df[model.feature_name()]

    y_true = df["label"]
    qids = df["qid"]

    df["pred"] = model.predict(X_eval)
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

# === CREATE HEATMAPS ===
metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
metrics_df.index.names = ["speakers", "instances"]
metrics_df = metrics_df.reset_index()

for metric in ["precision", "recall", "accuracy"]:
    pivot = metrics_df.pivot(index = "speakers", columns = "instances", values = metric)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues")
    plt.title(f"{metric.capitalize()} Heatmap by Grid Configuration")
    plt.ylabel("Number of Speakers")
    plt.xlabel("Instances per Speaker")
    plt.tight_layout()
    plt.savefig(f"EXPLICIT_ENSEMBLE/{metric}_heatmap.png")
    plt.show()
