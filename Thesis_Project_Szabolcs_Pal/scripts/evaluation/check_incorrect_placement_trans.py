import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict


def analyze_speaker_placement(base_paths, speaker_plot_path, instance_plot_path):
    """
    Analyzes speaker placement for incorrect top predictions and plots their distribution.
    
    Parameters:
    - base_paths: List of folder paths with result CSVs.
    - speaker_plot_path: Path to save boxplot by speaker count.
    - instance_plot_path: Path to save boxplot by instance count.
    """

    # Step 1: Collect files into two groupings
    grouped_by_speakers = defaultdict(list)
    grouped_by_instances = defaultdict(list)

    for base_path in base_paths:
        base = Path(base_path)
        grid_id = int(base.name.split('_')[1])  # e.g., grid_1_results → 1

        for csv_file in base.glob("*.csv"):
            df = pd.read_csv(csv_file)

            # Extract counts
            name_parts = csv_file.stem.split('_')  # ["40s", "10i"]
            speaker_count = name_parts[0]          # keep the '40s' format for grouping
            instance_count = name_parts[1]         # keep the '10i' format for grouping

            # Store numeric versions in DataFrame
            df["num_speakers"]  = int(str(name_parts[0]).rstrip("s"))
            df["num_instances"] = int(str(name_parts[1]).rstrip("i"))

            # Make query_id unique across grids
            df["query_id"] = df["query_id"].astype(str) + f"_g{grid_id}"

            # Group
            grouped_by_speakers[speaker_count].append(df)
            grouped_by_instances[instance_count].append(df)

    def compute_placement_stats(grouped_dfs, key_type="speakers"):
        """
        key_type:
        - "speakers": groups like "40s"      (x-axis = #speakers)
        - "instances": groups like "10i"     (x-axis = #instances per speaker)
        Returns: dict[group_key] -> list of placement indices (incorrect only)
        """
        placement_dict = {}

        for group_key, dfs in grouped_dfs.items():
            all_df = pd.concat(dfs, ignore_index=True)

            placement_list = []
            # one query_id = one ranked list over ALL enrolled speakers
            for _, group in all_df.groupby("query_id"):
                sorted_group = group.sort_values("match_rate", ascending=False).reset_index(drop=True)

                # Always use actual number of enrolled speakers for the slice
                n_speakers = int(group["num_speakers"].iloc[0])
                n_speakers = min(n_speakers, len(sorted_group))
                ranked = sorted_group.iloc[:n_speakers]

                idx = ranked.index[(ranked["query_speaker"] == ranked["enrolled_speaker"])]
                if len(idx) and idx[0] != 0:            # only incorrect-top cases
                    placement_list.append(int(idx[0]))

            placement_dict[group_key] = placement_list

        return placement_dict


    placements_by_speakers = compute_placement_stats(grouped_by_speakers, key_type="speakers")
    placements_by_instances = compute_placement_stats(grouped_by_instances, key_type="instances")

    # Step 3: Plotting Function
    def plot_placements(placement_dict, title, key_type, save_path):
        def parse_key(k):
            return int(str(k).rstrip("si"))

        plot_data = []
        labels = []
        for k, v in placement_dict.items():
            if not v:
                continue
            num_val = parse_key(k)
            labels.append(f"{num_val}{'s' if key_type == 'speakers' else 'i'}")

            if key_type == "speakers":
                # Convert raw placements to relative percentages
                rel_vals = [p / num_val for p in v]
                plot_data.append(rel_vals)
            else:
                # Keep raw placements for by-instances plot
                plot_data.append(v)

        # Sort both labels and data by the numeric value of the group key
        sorted_pairs = sorted(zip(labels, plot_data), key=lambda x: int(x[0][:-1]))
        labels, plot_data = zip(*sorted_pairs)

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=plot_data)
        plt.xticks(ticks=range(len(labels)), labels=labels)
        if key_type == "speakers":
            plt.ylabel("Relative Placement (fraction of total speakers)")
        else:
            plt.ylabel("Placement of Correct Speaker (if not top)")
        plt.xlabel("Number of Speakers" if key_type == "speakers" else "Number of Instances per Speaker")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # Step 4: Create both plots
    plot_placements(
        placements_by_speakers,
        title="Incorrect Top Predictions: Placement by Speaker Count",
        key_type="speakers",
        save_path=speaker_plot_path
    )

    plot_placements(
        placements_by_instances,
        title="Incorrect Top Predictions: Placement by Instance Count",
        key_type="instances",
        save_path=instance_plot_path
    )

base_paths = [
    "hubert_results_test_k_new/grid_1_results",
    "hubert_results_test_k_new/grid_2_results",
    "hubert_results_test_k_new/grid_3_results"
]

speaker_plot_path = "TRANSFORMERS_BASED_MODEL/final_eval_TEST_new3/Transformers_error_anal/speaker_placement_by_speakers_new.png"
instance_plot_path = "TRANSFORMERS_BASED_MODEL/final_eval_TEST_new3/Transformers_error_anal/speaker_placement_by_instances_new.png"

analyze_speaker_placement(base_paths, speaker_plot_path, instance_plot_path)
