from collections import Counter
from pathlib import Path
import pandas as pd
def analyze_certainty_threshold(df, output_path):
    """
    Creates separate certainty threshold histograms for label=1 (known) and label=-1 (unknown).

    Parameters:
    - df (pd.DataFrame): Must include ['qid', 'label', 'pred']
    - output_path (str or Path): Folder to save the plots
    """
    from pathlib import Path
    from collections import Counter
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    df["qid_base"] = df["qid"].apply(lambda x: "_".join(x.split("_")[:-1]))

    for label_value in [-1, 1]:
        results = []
        df_label = df[df["label"] == label_value]

        if df_label.empty:
            continue

        for base, group in df_label.groupby("qid_base"):
            true_label = group["label"].iloc[0]
            preds = group["pred"].tolist()
            pred_counts = Counter(preds)
            majority_label = pred_counts.most_common(1)[0][0]
            majority_vote_correct = (majority_label == true_label)
            correct_chunks = sum(group["label"] == group["pred"])
            total_chunks = len(group)
            certainty_required = total_chunks if not majority_vote_correct else correct_chunks

            results.append({
                "qid_base": base,
                "majority_correct": majority_vote_correct,
                "correct_chunks": correct_chunks,
                "total_chunks": total_chunks,
                "certainty_threshold": certainty_required
            })

        result_df = pd.DataFrame(results)

        if not result_df.empty:
            plt.figure(figsize=(8, 5))
            max_val = result_df["certainty_threshold"].max()
            bins = range(1, max_val + 2)
            plt.hist(result_df["certainty_threshold"], bins=bins, edgecolor='black')
            label_name = "Known (1)" if label_value == 1 else "Unknown (-1)"
            plt.title(f"Correct Chunks Needed for Certainty - {label_name}")
            plt.xlabel("Number of Correct Chunks Needed")
            plt.ylabel("Number of Bases")
            plt.tight_layout()
            filename = f"certainty_threshold_label{label_value}.png"
            plt.savefig(output_path / filename)
            plt.close()
