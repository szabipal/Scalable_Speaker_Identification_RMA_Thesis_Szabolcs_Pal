import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from pathlib import Path


def generate_confusion_matrices_by_group(df, output_path):
    """
    Generate confusion matrices:
    - Global (entire dataset)
    - Per config group
    - Per unique speaker count
    - Per unique instance count

    Parameters:
    - df (pd.DataFrame): Must include ['label', 'pred', 'config']
    - output_path (str or Path): Folder to save plots
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    label_order = [-1, 0, 1]  # -1 = unknown, 0 = abstain, 1 = known
    print(df.head())


    def plot_conf_matrix(y_true, y_pred, title, filename):
        cm = confusion_matrix(y_true, y_pred, labels=label_order)
        cm_df = pd.DataFrame(cm, index=label_order, columns=label_order)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Purples", cbar=False)
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True Label")
        plt.xticks(ticks=np.arange(3)+0.5, labels=["Unknown (-1)", "Abstain (0)", "Known (1)"])
        plt.yticks(ticks=np.arange(3)+0.5, labels=["Unknown (-1)", "Abstain (0)", "Known (1)"], rotation=0)
        plt.tight_layout()
        plt.savefig(output_path / filename)
        plt.close()

    # === Global confusion matrix ===
    plot_conf_matrix(df["label"], df["pred"], "Confusion Matrix: Global", "confusion_matrix_global.png")

    # Parse speaker and instance counts
    def parse_config(config_str):
        try:
            speakers, instances = config_str.split('_')
            return int(speakers.replace('s', '')), int(instances.replace('i', ''))
        except:
            return None, None

    # df["num_speakers"], df["num_instances"] = zip(*df["config"].apply(parse_config))
    print(df['config'][:10])
    df[["num_speakers", "num_instances"]] = df["config"].apply(parse_config).apply(pd.Series)
    # === By config ===
    for config_name, group_df in df.groupby("config"):
        plot_conf_matrix(group_df["label"], group_df["pred"],
                         f"Confusion Matrix: {config_name}",
                         f"confusion_matrix_{config_name}.png")

    # === By speaker count ===
    for speaker_count, group_df in df.groupby("num_speakers"):
        plot_conf_matrix(group_df["label"], group_df["pred"],
                         f"Confusion Matrix: {speaker_count}s (Speaker Count)",
                         f"confusion_matrix_{speaker_count}s.png")

    # === By instance count ===
    for instance_count, group_df in df.groupby("num_instances"):
        plot_conf_matrix(group_df["label"], group_df["pred"],
                         f"Confusion Matrix: {instance_count}i (Instance Count)",
                         f"confusion_matrix_{instance_count}i.png")

