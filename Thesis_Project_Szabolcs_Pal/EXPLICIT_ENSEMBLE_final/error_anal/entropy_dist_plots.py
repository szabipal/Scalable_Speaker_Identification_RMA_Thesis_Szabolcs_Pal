import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def generate_entropy_distribution_plots(df, output_path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path

    def parse_config(config_str):
        try:
            speakers, instances = config_str.split('_')
            return int(speakers.replace('s', '')), int(instances.replace('i', ''))
        except:
            return None, None

    df["num_speakers"], df["num_instances"] = zip(*df["config"].apply(parse_config))
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    def safe_histplot(data, title, filename, palette):
        plt.figure(figsize=(8, 5))
        if data["entropy"].nunique() > 1:
            sns.histplot(data=data, x="entropy", hue="label", kde=True, bins=20, palette=palette, element="step")
        else:
            sns.histplot(data=data, x="entropy", hue="label", kde=False, bins=20, palette=palette, element="step")
        plt.title(title)
        plt.xlabel("Entropy")
        plt.ylabel("Density")
        plt.legend(title="Label", labels=["Unknown (-1)", "Abstain (0)", "Known (1)"])
        plt.tight_layout()
        plt.savefig(output_path / filename)
        plt.close()

    # Histogram with KDE: Global plot
    safe_histplot(df, "Entropy Distribution by Label (Global)", "entropy_distribution_global.png", "Set1")

    # Histogram with KDE: By speaker count
    for s_count, group_df in df.groupby("num_speakers"):
        safe_histplot(group_df, f"Entropy Distribution by Label - {s_count} Speakers",
                      f"entropy_distribution_{s_count}s.png", "Set2")

    # Histogram with KDE: By instance count
    for i_count, group_df in df.groupby("num_instances"):
        safe_histplot(group_df, f"Entropy Distribution by Label - {i_count} Instances",
                      f"entropy_distribution_{i_count}i.png", "Set3")

    # === Global KDE-only plots for label 1 and -1 ===

    # 1. Speaker-wise KDE: Known (1) and Unknown (-1) only
    for target_label in [-1, 1, 0]:
        plt.figure(figsize=(10, 6))
        for s_count, group_df in df[df["label"] == target_label].groupby("num_speakers"):
            if group_df["entropy"].nunique() > 1:
                sns.kdeplot(group_df["entropy"], label=f"{s_count}s", linewidth=2)
        if target_label == 1:
            label_name = "Known (1)"
        elif target_label == -1:
            label_name = "Unknown (-1)"
        else:
            label_name = "Abstain (Incorrect known prediction)"
        
        plt.title(f"Entropy KDE by Number of Speakers - {label_name}")
        plt.xlabel("Entropy")
        plt.ylabel("Density")
        plt.legend(title="Speakers")
        plt.tight_layout()
        filename = f"global_kde_by_speakers_label{target_label}.png"
        plt.savefig(output_path / filename)
        plt.close()

    # 2. Instance-wise KDE: Known (1) and Unknown (-1) only
    for target_label in [-1, 1, 0]:
        plt.figure(figsize=(10, 6))
        for i_count, group_df in df[df["label"] == target_label].groupby("num_instances"):
            if group_df["entropy"].nunique() > 1:
                sns.kdeplot(group_df["entropy"], label=f"{i_count}i", linewidth=2)
        if target_label == 1:
            label_name = "Known (1)"
        elif target_label == -1:
            label_name = "Unknown (-1)"
        else:
            label_name = "Abstain (Incorrect known prediction)"
        plt.title(f"Entropy KDE by Number of Instances - {label_name}")
        plt.xlabel("Entropy")
        plt.ylabel("Density")
        plt.legend(title="Instances")
        plt.tight_layout()
        filename = f"global_kde_by_instances_label{target_label}.png"
        plt.savefig(output_path / filename)
        plt.close()