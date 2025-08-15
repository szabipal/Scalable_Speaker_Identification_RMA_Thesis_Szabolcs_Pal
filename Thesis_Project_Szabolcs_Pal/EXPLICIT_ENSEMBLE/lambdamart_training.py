
import pandas as pd
from pathlib import Path
import lightgbm as lgb
import itertools
import os

# === CONFIGURATION ===
train_fraction = 1
train_folders = [1, 2, 3, 4]
eval_folder = 5
base_path = Path("output_phonetic_new")
output_dir = Path("EXPLICIT_ENSEMBLE_final/phonetic_final_128")
output_dir.mkdir(parents=True, exist_ok=True)

# === HYPERPARAMETER GRID ===
param_grid = {
    'learning_rate': [0.05, 0.1],
    'num_leaves': [15, 31, 63, 127, 255, 511, 1023],
    # 'num_leaves': [511, 1023],
    'feature_fraction': [1.0],
    'bagging_fraction': [1.0, 0.8],
}

# === COLLECT TRAINING FILES ===
train_files = []
for x in train_folders:
    merged_folder = base_path / f"lambdamart_cosine_dataset_known_queries_normed{x}"
    for grid_folder in merged_folder.iterdir():
        if grid_folder.is_dir():
            y = grid_folder.name
            csv_file = grid_folder / f"lambdamart_merged_{y}.csv"
            if csv_file.exists():
                train_files.append(csv_file)

eval_files = []
merged_eval = base_path / f"lambdamart_cosine_dataset_known_queries_normed{eval_folder}"
for grid_folder in merged_eval.iterdir():
    if grid_folder.is_dir():
        y = grid_folder.name
        csv_file = grid_folder / f"lambdamart_merged_{y}.csv"
        if csv_file.exists():
            eval_files.append(csv_file)

# === LOAD DATA ===
print("Loading training and evaluation data...")
df_train = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
df_eval = pd.concat([pd.read_csv(f) for f in eval_files], ignore_index=True)

if train_fraction < 1.0:
    df_train = df_train.sample(frac=train_fraction, random_state=42)

# === FEATURE SELECTION ===
feature_cols = [col for col in df_train.columns if col.startswith(("mfcc", "mel_band_mid", "mel_band_low", "mel_band_high"))]
X_train, y_train, qid_train = df_train[feature_cols], df_train['label'], df_train['qid']
X_eval, y_eval, qid_eval = df_eval[feature_cols], df_eval['label'], df_eval['qid']

def get_group(qid_series):
    return qid_series.value_counts(sort=False).sort_index().values

train_group = get_group(qid_train)
eval_group = get_group(qid_eval)

# === GRID SEARCH ===
param_combinations = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())

for i, values in enumerate(param_combinations):
    # Compose parameter set
    params = dict(zip(param_names, values))
    params.update({
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'verbosity': -1
    })

    print(f"\n🔧 Training config {i+1}/{len(param_combinations)}: {params}")

    # Prepare data
    train_data = lgb.Dataset(X_train, label=y_train, group=train_group)
    eval_data = lgb.Dataset(X_eval, label=y_eval, group=eval_group, reference=train_data)

    # Train model
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, eval_data],
        valid_names=["train", "eval"],
        num_boost_round=100,
        # verbose_eval=False
    )

    # === Predict and Rank ===
    df_eval["pred"] = model.predict(X_eval)
    sorted_df = df_eval.sort_values(["qid", "pred"], ascending=[True, False])
    top_preds = sorted_df.groupby("qid").head(1)  # Top-1 per query

    # === Standard Stats ===
    top1_acc = top_preds["label"].sum() / df_eval["qid"].nunique()
    avg_conf = top_preds["pred"].mean()
    min_conf = top_preds["pred"].min()
    max_conf = top_preds["pred"].max()
    label_mean = df_eval["label"].mean()
    pred_std = df_eval["pred"].std()

    # === Quantiles for all top predictions ===
    quantiles_all = top_preds["pred"].quantile([0.10, 0.25, 0.50, 0.75, 0.90])

    # === Filter correct predictions only ===
    correct_preds = top_preds[top_preds["label"] == 1]

    # === Stats for correct top predictions ===
    correct_avg = correct_preds["pred"].mean()
    correct_quantiles = correct_preds["pred"].quantile([0.10, 0.25, 0.50, 0.75, 0.90])

    # === Print or return results ===
    print("Top-1 Accuracy:", top1_acc)
    print("Avg prediction (top-1):", avg_conf)
    print("Prediction quantiles (top-1):")
    print(quantiles_all)

    print("\nCorrect top-1 prediction stats:")
    print("Avg confidence for correct preds:", correct_avg)
    print("Quantiles for correct preds:")
    print(correct_quantiles)

    # Save model and stats
    config_name = f"lr{params['learning_rate']}_nl{params['num_leaves']}_ff{params['feature_fraction']}_bf{params['bagging_fraction']}"
    model_file = output_dir / f"lambdamart_model_{config_name}.txt"
    model.save_model(str(model_file))

    # Save stats to text file
    stats_file = output_dir / f"lambdamart_stats_{config_name}.txt"
    with open(stats_file, "w") as f:
        f.write(f"Top-1 Accuracy: {top1_acc:.4f}\n")
        f.write(f"Avg prediction (top-1): {avg_conf:.4f}\n")
        f.write(f"Min prediction (top-1): {min_conf:.4f}\n")
        f.write(f"Max prediction (top-1): {max_conf:.4f}\n")
        f.write(f"Overall label mean: {label_mean:.4f}\n")
        f.write(f"Prediction std dev: {pred_std:.4f}\n\n")

        f.write("Quantiles for all top predictions:\n")
        for q, val in quantiles_all.items():
            f.write(f"  {int(q*100)}%: {val:.4f}\n")

        f.write("\nAvg prediction for correct top-1s: {:.4f}\n".format(correct_avg))
        f.write("Quantiles for correct top-1 predictions:\n")
        for q, val in correct_quantiles.items():
            f.write(f"  {int(q*100)}%: {val:.4f}\n")
    
    print(f"✅ Saved model to {model_file}")
    print(f"📄 Saved report to {stats_file}")
    print(f"Top-1 Accuracy: {top1_acc:.4f}\n")
    print(f"Avg prediction (top-1): {avg_conf:.4f}\n")
