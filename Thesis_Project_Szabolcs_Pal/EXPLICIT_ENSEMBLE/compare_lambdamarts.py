import os
import re
import pandas as pd
from pathlib import Path

# === Configuration ===
report_dir = Path("EXPLICIT_ENSEMBLE_final/phonetic_final_128/")
report_files = sorted(report_dir.glob("lambdamart_stats_*.txt"))
# print(report_files)

results = []

# === Extract metrics from each report ===
for report_file in report_files:
    # print(report_file)
    with open(report_file, "r") as f:
        content = f.read()

    config_name = report_file.stem.replace("lambdamart_stats_", "")
    
    def extract_metric(pattern, default=-1.0):
        match = re.search(pattern, content)
        return float(match.group(1)) if match else default

    top1_acc = extract_metric(r"Top-1 Accuracy:\s*([0-9.\-]+)")
    avg_conf = extract_metric(r"Avg prediction \(top-1\):\s*([0-9.\-]+)")
    min_conf = extract_metric(r"Min prediction \(top-1\):\s*([0-9.\-]+)")
    max_conf = extract_metric(r"Max prediction \(top-1\):\s*([0-9.\-]+)")
    label_mean = extract_metric(r"Overall label mean:\s*([0-9.\-]+)")
    pred_std = extract_metric(r"Prediction std dev:\s*([0-9.\-]+)")

    results.append({
        "config": config_name,
        "top1_accuracy": top1_acc,
        "avg_confidence": avg_conf,
        "min_conf": min_conf,
        "max_conf": max_conf,
        "label_mean": label_mean,
        "pred_std": pred_std
    })


# === Build summary dataframe ===
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="top1_accuracy", ascending=False)

# === Save to CSV ===
summary_file = report_dir / "lambdamart_model_summary.csv"
results_df.to_csv(summary_file, index=False)
print(f"✅ Summary saved to: {summary_file}")

# === Print top-performing config ===
best = results_df.iloc[0]
print(f"\n Best configuration: {best['config']}")
print(f"Top-1 Accuracy: {best['top1_accuracy']:.4f}")
