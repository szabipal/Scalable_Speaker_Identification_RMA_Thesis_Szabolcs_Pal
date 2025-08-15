import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import seaborn as sns


from pathlib import Path


project_root = Path(__file__).resolve().parents[1]  # Adjust [1] to [2], [3], etc. if deeper


# Dictionary to hold placement indices grouped by speaker and instance counts
placement_data_by_speakers = defaultdict(list)
placement_data_by_instances = defaultdict(list)

# Iterate through all CSV files
for file_path in base_dir.glob("*.csv"):
    filename = file_path.stem  # e.g., confidences_1_10s_10i
    parts = filename.split("_")
    if len(parts) < 3:
        continue

    # Parse speaker and instance counts
    speaker_part = parts[-1]
    instance_part = parts[-2]

    if not speaker_part.endswith("i") or not instance_part.endswith("s"):
        continue

    try:
        num_instances = int(speaker_part.replace("i", ""))
        num_speakers = int(instance_part.replace("s", ""))
    except ValueError:
        continue

    df = pd.read_csv(file_path)

    # Group by qid
    for qid, group in df.groupby("qid"):
        if 1 not in group["label"].values:
            continue

        # Sort by prediction score descending
        sorted_group = group.sort_values("pred", ascending=False).reset_index(drop=True)

        # Find the index where label==1
        correct_index = sorted_group[sorted_group["label"] == 1].index[0]

        # If it's already at the top (i.e., index 0), skip
        if correct_index == 0:
            continue

        # Store index in both dictionaries
        placement_data_by_speakers[num_speakers].append(correct_index)
        placement_data_by_instances[num_instances].append(correct_index)


print(f"Collected placements by speakers: {placement_data_by_speakers}")
print(f"Collected placements by instances: {placement_data_by_instances}")

# Create output directory if it doesn't exist
output_dir = project_root / "Spectral_error_anal/misplacement_average"
output_dir.mkdir(parents=True, exist_ok=True)

# Boxplot for speakers
plt.figure(figsize=(12, 6))
data_speakers = [
    placement_data_by_speakers[k] for k in sorted(placement_data_by_speakers.keys())
]
labels_speakers = [f"{k}s" for k in sorted(placement_data_by_speakers.keys())]

sns.boxplot(data=data_speakers)
plt.xticks(ticks=range(len(labels_speakers)), labels=labels_speakers)
plt.title("Misplacement Index by Number of Speakers (Spectral Model)")
plt.xlabel("Number of Speakers")
plt.ylabel("Placement of Correct Speaker (if not top)")
plt.tight_layout()
plt.savefig(output_dir / "misplacement_by_speakers_spectral.png")
plt.close()

# Boxplot for instances
plt.figure(figsize=(12, 6))
data_instances = [
    placement_data_by_instances[k] for k in sorted(placement_data_by_instances.keys())
]
labels_instances = [f"{k}i" for k in sorted(placement_data_by_instances.keys())]

sns.boxplot(data=data_instances)
plt.xticks(ticks=range(len(labels_instances)), labels=labels_instances)
plt.title("Misplacement Index by Number of Instances per Speaker (Spectral Model)")
plt.xlabel("Number of Instances per Speaker")
plt.ylabel("Placement of Correct Speaker (if not top)")
plt.tight_layout()
plt.savefig(output_dir / "misplacement_by_instances_spectral.png")
plt.close()
