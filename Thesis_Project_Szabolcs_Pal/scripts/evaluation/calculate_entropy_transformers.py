"""
entropy extraction for the transformers-based model.

what it does
- reads per-query match score csvs, takes the top-5 matches per query,
  turns their scores into proportions, and computes normalized shannon entropy.
- records the top match id, the entropy value, and a binary correctness label
  (for known/unknown thresholding), plus the source filename and grid number.
- can batch over multiple result folders and save one consolidated csv.

notes
- entropy is computed over the top-5 proportions and normalized by log2(5),
  so values lie roughly in [0, 1] (0 = confident/peaky, 1 = uncertain/flat).
- expects columns: 'query_id', 'sum' (match score), 'enrolled_speaker',
  and 'query_speaker' in the input files.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import entropy
import math
import re

def process_match_file(file_path, thresholded_label, grid_number):
    """
    compute normalized entropy for one match csv.

    parameters
    ----------
    file_path : pathlib.Path
        csv with per-query match scores.
    thresholded_label : str
        'known' or 'unknown'; controls how 'label' is assigned.
    grid_number : str | int
        grid id to log alongside results (parsed upstream).

    returns
    -------
    pandas.DataFrame
        rows = queries, columns = ['query_id','top_match_id','normalized_entropy',
        'source_file','grid_number','label'].
    """
    df = pd.read_csv(file_path)
    file_name = file_path.stem  # filename without .csv

    if df.empty or "query_id" not in df.columns or "sum" not in df.columns:
        return pd.DataFrame()

    results = []

    # group by query; use only the strongest 5 matches per query
    for query_id, group in df.groupby("query_id"):
        top5 = group.sort_values("sum", ascending=False).head(5)

        match_sum = top5["sum"].sum()
        if match_sum == 0:
            proportions = np.ones(len(top5)) / len(top5)
        else:
            proportions = top5["sum"] / match_sum

        # normalized shannon entropy over top-5 scores (base 2)
        norm_entropy = entropy(proportions, base=2) / math.log(5, 2) if len(proportions) > 1 else 0.0

        # pick the top match and set label depending on known/unknown mode
        top_match_row = top5.iloc[proportions.argmax()]
        top_match_id = top_match_row["enrolled_speaker"]
        query_sp=top_match_row['query_speaker']

        if thresholded_label != 'unknown':
            label = 1 if top_match_id == query_sp else 0
        else:
            label = 0

        results.append({
            "query_id": query_id,
            "top_match_id": top_match_id,
            "normalized_entropy": norm_entropy,
            "source_file": file_name,
            "grid_number": grid_number,      # ✅ Added column
            "label": label
        })

    return pd.DataFrame(results)


def process_folders(folder_paths, output_csv="entropy_transformers_known.csv", thresholded_label='known'):
    """
    batch process multiple folders of match csvs and write a single output file.

    parameters
    ----------
    folder_paths : list[str | pathlib.Path]
        folders containing *.csv match files (e.g., per grid).
    output_csv : str
        destination path for the consolidated results.
    thresholded_label : str
        'known' sets label via top match correctness; 'unknown' forces 0.

    side effects
    ------------
    saves `output_csv` with the concatenated results.
    """
    all_results = []

    for folder in folder_paths:
        print(f"[INFO] Processing folder: {folder}")
        folder_path = Path(folder)

        # try to extract grid id from folder name (e.g., "grid_1_results")
        match = re.search(r'grid_(\d+)', str(folder_path))
        grid_number = match.group(1) if match else "unknown"

        for file_path in tqdm(folder_path.glob("*.csv"), desc=f"Processing {folder}"):
            result_df = process_match_file(file_path, thresholded_label, grid_number)
            if not result_df.empty:
                all_results.append(result_df)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved all results to {output_csv}")

# === USAGE ===
if __name__ == "__main__":
    # folder_list = [
    #     "hubert_results_dev_k_new/grid_1_results",
    #     "hubert_results_dev_k_new/grid_2_results",
    #     "hubert_results_dev_k_new/grid_3_results",
    # ]
    # process_folders(folder_list, output_csv="TRANSFORMERS_BASED_MODEL/entropy_transformers_known_dev_new.csv", thresholded_label='known')

    # folder_list = [
    #     "hubert_results_dev_uk_new/grid_1_results",
    #     "hubert_results_dev_uk_new/grid_2_results",
    #     "hubert_results_dev_uk_new/grid_3_results",
    # ]
    # process_folders(folder_list, output_csv="TRANSFORMERS_BASED_MODEL/entropy_transformers_unknown_dev_new.csv", thresholded_label='unknown')


    # folder_list = [
    #     "hubert_results_test_k_new/grid_1_results",
    #     "hubert_results_test_k_new/grid_2_results",
    #     "hubert_results_test_k_new/grid_3_results",
    # ]
    # process_folders(folder_list, output_csv="TRANSFORMERS_BASED_MODEL/entropy_transformers_known_test_new2.csv", thresholded_label='known')

    # folder_list = [
    #     "hubert_results_test_uk_new/grid_1_results",
    #     "hubert_results_test_uk_new/grid_2_results",
    #     "hubert_results_test_uk_new/grid_3_results",
    # ]
    # process_folders(folder_list, output_csv="TRANSFORMERS_BASED_MODEL/entropy_transformers_unknown_test_new2.csv", thresholded_label='unknown')
