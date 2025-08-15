import os
import pandas as pd
import numpy as np
from scipy.stats import entropy
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report


def calculate_entropy(prob_list):
    prob_array = np.array(prob_list)
    if prob_array.sum() == 0:
        return 0
    prob_array = prob_array / prob_array.sum()  # Normalize
    return entropy(prob_array, base=2)

def extract_s_i_from_filename(name):
    """Extract speaker and instance values from a filename like confidences_10s_1i.csv"""
    parts = name.replace("confidences_", "").replace(".csv", "").split("_")
    s_val = int(parts[0].replace("s", ""))
    i_val = int(parts[1].replace("i", ""))
    return s_val, i_val

def final_prediction_evaluation(confidence_dir, thresholds_csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load thresholds CSV, expected to have: s_val, i_val, threshold, type
    threshold_df = pd.read_csv(thresholds_csv_path)
    print(threshold_df)
    results = []
    confidence_files = list(Path(confidence_dir).glob("*.csv"))

    for file_path in tqdm(confidence_files, desc="Evaluating configurations"):
        filename = file_path.name
        s_val, i_val = extract_s_i_from_filename(filename)

        type_tag = "known" if "known" in confidence_dir.lower() else "unknown"
        thresh_row = threshold_df[
            (threshold_df["s_val"] == s_val) & 
            (threshold_df["i_val"] == i_val) & 
            (threshold_df["type"] == type_tag)
        ]

        if thresh_row.empty:
            print(f"⚠️ No threshold found for: s={s_val}, i={i_val}, type={type_tag}")
            continue

        threshold = float(thresh_row["threshold"].values[0])
        df = pd.read_csv(file_path)

        grouped = df.groupby("qid")
        predictions = []

        for qid, group in grouped:
            probs = group["pred"].tolist()
            true_label = group["label"].iloc[0] if "label" in group else "UNKNOWN"
            ent = calculate_entropy(probs)

            if ent <= threshold:
                pred_label = f"known_{np.argmax(probs)}"
            else:
                pred_label = "unknown"

            predictions.append({
                "qid": qid,
                "entropy": ent,
                "prediction": pred_label,
                "gold": true_label,
                "s_val": s_val,
                "i_val": i_val,
                "type": type_tag
            })

        result_df = pd.DataFrame(predictions)
        result_csv_path = os.path.join(output_dir, f"predictions_{s_val}s_{i_val}i_{type_tag}.csv")
        result_df.to_csv(result_csv_path, index=False)
        results.append(result_df)

    combined_results = pd.concat(results, ignore_index=True)
    print("✅ Combined results preview:")
    print(combined_results.head())
    combined_results.to_csv(os.path.join(output_dir, "all_combined_results.csv"), index=False)

    report = classification_report(combined_results['gold'], combined_results['prediction'], zero_division=0)
    print(report)

    return combined_results



if __name__ == "__main__":
    final_prediction_evaluation(
        confidence_dir="/Users/g/Documents/School_stuff/Master/Thesis/Explicit_model/expl-speaker_embedding_models/EXPLICIT_ENSMEBLE/confidences_phonetic_test_known",            # or "confidences_unknown"
        thresholds_csv_path="/Users/g/Documents/School_stuff/Master/Thesis/Explicit_model/expl-speaker_embedding_models/EXPLICIT_ENSEMBLE/entropy_thresholds_phonetic.csv",
        output_dir="EXPLICIT_ENSMEBLE/final_predictions"
    )