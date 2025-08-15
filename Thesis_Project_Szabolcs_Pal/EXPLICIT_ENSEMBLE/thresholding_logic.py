"""
entropy-based threshold search for transformers / lambdamart confidences.

what this does
- reads per-query confidence csvs (known & unknown) produced earlier.
- for each query (qid), picks the top candidate by `pred`, assigns a correctness label,
  and computes a normalized entropy over the top scores.
- builds matched datasets (known↔unknown) so we can search a threshold that best separates
  the target class (known = low entropy, unknown = high entropy).
- sweeps thresholds, reports the best one per config (e.g., "10s_10i") and saves results.

expects
- confidence csvs with at least: qid, pred, candidate_speaker (and typically label).
- qid encodes a gold speaker id as the 3rd underscore-separated token (see `get_labels`).
- filenames like: confidences_<prefix>_<config>.csv where <config> looks like "10s_10i".
"""

import pandas as pd
import numpy as np
from scipy.stats import differential_entropy, entropy
from pathlib import Path
import os
import sys

import warnings

# keep logs clean; these are common with some scipy/pandas paths
warnings.filterwarnings('ignore', category=DeprecationWarning)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


def get_labels(df, thresh_label='known'):
    """
    choose the top candidate per qid and mark the result (1/0) based on whether
    it matches the gold speaker id (inferred from qid). flips meaning for unknown.

    contract
    - df must contain: 'qid', 'pred', 'candidate_speaker' (and 'label' is ignored here).
    - the gold speaker id is parsed from qid: "prefix1_prefix2_<gold>_..." (3rd token).
    - thresh_label='known' → correct if top_cand == gold, else 0
      thresh_label='unknown' → correct if top_cand != gold, else 0

    returns
    - df with a new 'result' column per row (group-wise repeated).
    """

    # inner: operate per qid group
    def process_group(group_df, thresh_label='known'):
        group_df = group_df.copy()

        # find index of the maximal prediction within the group
        top_speaker = (None, float("-inf"))
        top_index = None
        for index, row in group_df.iterrows():
            if row['pred'] > top_speaker[1]:
                top_speaker = (index, row['pred'])
                top_index = index

        # parse chosen candidate and "gold" speaker id
        top_cand = int(group_df.loc[top_index, 'candidate_speaker'])
        correct_speaker = int(group_df['qid'].iloc[0].split('_')[2])

        # assign label per mode
        if thresh_label == 'known':
            group_df['result'] = 1 if top_cand == correct_speaker else 0
        elif thresh_label == 'unknown':
            group_df['result'] = 1 if top_cand != correct_speaker else 0

        return group_df

    # apply per qid
    df_with_results = df.groupby('qid', group_keys=False).apply(process_group, thresh_label=thresh_label)
    return df_with_results


from scipy.stats import entropy
import numpy as np

def calc_entropy(group_df):
    """
    compute a normalized entropy for a qid group using the top-N scores.
    the normalization and N vary slightly based on how many rows the group has.

    policy
    - len==2   → use both; normalize by log2(2)
    - len==5   → scale top-5 to [0,1], use top-4 as probs; normalize by log2(4)
    - len==10  → scale top-6 to [0,1], use top-5 as probs; normalize by log2(5)

    note: this mirrors earlier experimental settings; keep consistent with upstream.
    """
    sorted_preds = group_df.sort_values("pred", ascending=False)

    if len(sorted_preds) == 2:
        total = sorted_preds['pred'].sum()
        scaled_preds = sorted_preds['pred'] / total
        ent = entropy(scaled_preds)
        return ent / np.log2(2)

    elif len(sorted_preds) == 5:
        top_preds = sorted_preds["pred"].values[:5]
        scale_min = np.min(top_preds)
        scale_max = np.max(top_preds)
        scale_range = scale_max - scale_min
        top_4 = top_preds[:4]
        scaled = (top_4 - scale_min) / scale_range
        probs = scaled / np.sum(scaled)
        ent = entropy(probs)
        return ent / np.log2(4)

    elif len(sorted_preds) == 10:
        top_preds = sorted_preds["pred"].values[:6]
        scale_min = np.min(top_preds)
        scale_max = np.max(top_preds)
        scale_range = scale_max - scale_min
        top_5 = top_preds[:5]
        scaled = (top_5 - scale_min) / scale_range
        probs = scaled / np.sum(scaled)
        ent = entropy(probs)
        return ent / np.log2(5)


import re

def extract_config_name(filename):
    """
    pull the "<speakers>s_<instances>i" pattern (e.g., '10s_10i') from a filename.
    returns None if it can't be found.
    """
    match = re.search(r'\d+s_\d+i', filename)
    return match.group() if match else None


def build_uk_data(data_k, data_uk, mode='unknown'):
    """
    assemble balanced per-config datasets for threshold finding.

    idea
    - if mode='unknown': keep unknowns as-is, sample the same amount from knowns and set their label=0.
    - if mode='known'  : keep knowns as-is, sample the same amount from unknowns and set their label=0.

    inputs
    - data_k: dict[config_name -> df] for known
    - data_uk: dict[config_name -> df] for unknown
    - mode: 'unknown' | 'known'

    returns
    - dict[config_name -> df] where the primary set keeps its original labels,
      and the sampled set has label overwritten to 0.
    """
    assert mode in ['unknown', 'known'], "Mode must be 'unknown' or 'known'."
    
    combined_datasets = {}
    primary_data = data_uk if mode == 'unknown' else data_k
    sampled_data = data_k if mode == 'unknown' else data_uk

    for config_name, df_primary in primary_data.items():
        if config_name in sampled_data:
            df_other = sampled_data[config_name]

            if df_primary.empty:
                print(f"  Warning: Primary data for config '{config_name}' is empty. Skipping.")
                continue
            if df_other.empty:
                print(f"  Warning: Sample data for config '{config_name}' is empty. Skipping.")
                continue

            num_samples = len(df_primary)
            num_samples_to_take = min(num_samples, len(df_other))

            df_sampled = df_other.sample(n=num_samples_to_take, random_state=42).copy()
            df_sampled['label'] = 0  # only sampled part is forced to 0

            combined_df = pd.concat([df_primary.copy(), df_sampled], ignore_index=True)
            combined_datasets[config_name] = combined_df
        else:
            print(f"  Warning: Config '{config_name}' not found in sampled data source. Skipping.")

    return combined_datasets

def thresholding(conf_k_dir, conf_uk_dir, cut_off, threshold, output_path):
    """
    end-to-end threshold search.

    steps
    1) read all known/unknown confidence files for overlapping configs.
    2) per qid: derive 'result' (correctness) and an entropy score.
    3) build matched datasets (known-mode and unknown-mode).
    4) sweep thresholds and keep the best per config under a small TP floor.
    5) save best thresholds (known to `output_path`, unknown to a fixed file).

    notes
    - stems are expected like 'confidences_<prefix>_<config>.csv'. we prepend the prefix
      onto qid to keep them unique across sources.
    - `cut_off` is kept for historical compatibility but not used directly here.
    """

    def extract_config_name(stem):
        # pull last two tokens (e.g., confidences_1_10s_10i → "10s_10i")
        return "_".join(stem.split("_")[-2:])

    def extract_prefix_number(stem):
        # number between underscores, used to namespace qids
        match = re.search(r"_(\d+)_", stem)
        return match.group(1) if match else "0"

    configs_k = {}
    configs_uk = {}
    config_names = set()

    # --- load known configs ---
    for file in Path(conf_k_dir).glob("*.csv"):
        stem = file.stem
        name = extract_config_name(stem)      # e.g., 10s_10i
        prefix = extract_prefix_number(stem)  # e.g., 1
        if prefix == '5':                     # optional skip rule from earlier experiments
            continue

        config_names.add(name)

        df = pd.read_csv(file, on_bad_lines='skip')
        if "qid" in df.columns:
            # namespace qids so overlaps across prefixes don't clash
            df["qid"] = df["qid"].astype(str).apply(lambda q: f"{prefix}_{q}")

        configs_k.setdefault(name, []).append(df)

    # --- load unknown configs (only when config exists in known set) ---
    for file in Path(conf_uk_dir).glob("*.csv"):
        stem = file.stem
        name = extract_config_name(stem)
        prefix = extract_prefix_number(stem)
        if prefix == '5':
            continue

        if name in config_names:
            df = pd.read_csv(file, on_bad_lines='skip')
            if "qid" in df.columns:
                df["qid"] = df["qid"].astype(str).apply(lambda q: f"{prefix}_{q}")
            configs_uk.setdefault(name, []).append(df)

    # merge per-config lists
    configs_k = {k: pd.concat(v, ignore_index=True) for k, v in configs_k.items()}
    configs_uk = {k: pd.concat(v, ignore_index=True) for k, v in configs_uk.items()}

    # --- compute per-qid entropy tables ---
    def process_entropy_data(configs, thresh_label='known'):
        """
        for each config, group by qid → label via get_labels() and entropy via calc_entropy().
        returns dict[config -> df(qid, entropy, label)].
        """
        threshold_data = {}
        for name, df in configs.items():
            if df.empty:
                print(f" Skipping empty config: {name}")
                continue
            if 'qid' not in df.columns:
                print(f" 'qid' column missing in: {name}")
                continue

            entropy_rows = []
            for _, group in df.groupby("qid"):
                labeled_df = get_labels(group.copy(), thresh_label=thresh_label)
                e = calc_entropy(labeled_df)
                entropy_rows.append({
                    'qid': group['qid'].iloc[0],
                    'entropy': e,
                    'label': labeled_df['result'].iloc[0]
                })
            threshold_data[name] = pd.DataFrame(entropy_rows)
        return threshold_data

    print('started getting entropy data')
    threshold_data_k = process_entropy_data(configs_k, thresh_label='known')
    threshold_data_uk = process_entropy_data(configs_uk, thresh_label='unknown')
    
    # track unsuccessful thresholds for later inspection
    failed_threshold_values = pd.DataFrame(
        columns=['config', 'label', 'threshold', 'precision', 'recall', 'TP', 'FP', 'FN', 'TN', 'Num_T', 'Num_F']
    )

    def find_thresholds(threshold_data, threshold, thresholded_label='known'):
        """
        brute-force sweep for best threshold per config.

        policy
        - if thresholded_label='unknown' → predict unknown when entropy > t (search t in [0,1]).
        - if thresholded_label='known'   → predict known   when entropy < t (search t in (1..0]).
        - pick the t that maximizes precision, subject to a minimal TP fraction (~20% of positives).
        """
        found_thresholds = []
        target_label_value = 0 if thresholded_label == 'known' else 1  # not directly used; kept for clarity

        for config, df in threshold_data.items():
            if df.empty:
                print(f"⚠️ Skipping empty threshold data for config: {config} in find_thresholds for {label_type_to_predict}.")
                continue

            # search direction depends on mode
            threshold_range = np.arange(1, 0, -0.01) if thresholded_label == 'known' else np.arange(0, 1, 0.01)

            best_precision = 0
            best_recall = 0
            best_threshold = None
            tp_best = fp_best = fn_best = tn_best = 0
            num_true = (df['label'] == 1).sum()
            num_false = (df['label'] == 0).sum()

            for i in threshold_range:
                if thresholded_label == 'unknown':
                    df['pred'] = (df['entropy'] > i).astype(int)   # unknowns = high entropy
                else:
                    df['pred'] = (df['entropy'] < i).astype(int)   # knowns = low entropy

                tp = ((df['pred'] == 1) & (df['label'] == 1)).sum()
                fp = ((df['pred'] == 1) & (df['label'] == 0)).sum()
                fn = ((df['pred'] == 0) & (df['label'] == 1)).sum()
                tn = ((df['pred'] == 0) & (df['label'] == 0)).sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                # keep the best precision, with a tiny floor on tp so we don't select degenerate thresholds
                if precision > best_precision and tp > (num_true * 0.2) and i not in (0, 1):
                    best_precision = precision
                    best_recall = recall
                    best_threshold = i
                    tp_best, fp_best, fn_best, tn_best = tp, fp, fn, tn
                else:
                    failed_thr_data = {'config': config, 'label': thresholded_label, 'threshold': i, 'precision': precision}
                    failed_threshold_values.loc[len(failed_threshold_values)] = failed_thr_data

            if best_threshold is not None:
                found_thresholds.append({
                    'config': config,
                    'threshold': round(best_threshold, 3),
                    'precision': round(best_precision, 3),
                    'recall': round(best_recall, 3),
                    'TP': tp_best,
                    'FP': fp_best,
                    'FN': fn_best,
                    'TN': tn_best,
                    'Num_T': num_true,
                    'Num_F': num_false
                })
                
        return pd.DataFrame(found_thresholds)

    print('starting_finding_entropy_data')

    # balance datasets for each mode
    uk_data_combined = build_uk_data(threshold_data_k, threshold_data_uk, mode='unknown')
    k_data_combined  = build_uk_data(threshold_data_k, threshold_data_uk, mode='known')

    # calculate thresholds per mode
    found_uk_thresholds = find_thresholds(uk_data_combined, threshold, thresholded_label='unknown')
    found_k_thresholds  = find_thresholds(k_data_combined,  threshold, thresholded_label='known')

    # save results
    found_k_thresholds.to_csv(output_path)
    print('saved known')
    found_uk_thresholds.to_csv('EXPLICIT_ENSEMBLE_final/threshold_results_phonetic_uk_new0_25.csv')
    print('saved unknown')
    failed_threshold_values.to_csv('EXPLICIT_ENSEMBLE_final/failed_thresholds_phonetic_new0_25.csv')




# example call (phonetic); adjust paths/filenames to your layout
thresholding(
    conf_k_dir="EXPLICIT_ENSEMBLE_final/confidences_phonetic_known/",
    conf_uk_dir="EXPLICIT_ENSEMBLE_final/confidences_phonetic_unknown/",
    cut_off=0,
    threshold=0.7,
    output_path="EXPLICIT_ENSEMBLE_final/threshold_results_phonetic_k_new0_2.csv"
)
