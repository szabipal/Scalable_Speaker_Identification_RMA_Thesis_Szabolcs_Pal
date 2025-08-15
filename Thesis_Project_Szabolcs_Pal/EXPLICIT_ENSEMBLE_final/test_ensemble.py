
import os
import pandas as pd
import numpy as np
from scipy.stats import entropy
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys

import warnings

# Suppress DeprecationWarnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



def split_by_prefix(results_df):
    # Extract prefix from the first number in the qid (before first '_')
    results_df['prefix'] = results_df['qid'].apply(lambda x: str(x).split('_')[0] if pd.notna(x) and '_' in str(x) else 'global')

    return results_df


def plot_conf_matrix(y_true, y_pred, labels, display_labels, title, outpath):
    y_true = pd.Series(y_true).astype("Int64").dropna()
    y_pred = pd.Series(y_pred).astype("Int64").dropna()

    # Now cast to numpy int (required for sklearn)
    y_true = y_true.astype(int).values
    y_pred = y_pred.astype(int).values
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    # Convert to integers and drop NaNs

    disp.plot(ax=ax, cmap="Blues", values_format='d')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()





def get_labels(df, thresholded_label='known'):
    def process_group(group_df):
        group_df = group_df.copy()  # avoid modifying the original df

        top_speaker = (None, float("-inf"))  # index and max pred score
        top_index = None

        for index, row in group_df.iterrows():
            if row['pred'] > top_speaker[1]:
                top_speaker = (index, row['pred'])
                top_index = index
        top_cand = int(group_df.loc[top_index, 'candidate_speaker'])
        


        correct_speaker = int(group_df['qid'].iloc[0].split('_')[2])

        if thresholded_label == 'known':
            if top_cand == correct_speaker:
                group_df['result'] = 1
            else:
                group_df['result'] = 0
        elif thresholded_label == 'unknown':
            if top_cand != correct_speaker:
                group_df['result'] = -1
            else:
                group_df['result'] = 0

        return group_df

    return df.groupby('qid', group_keys=False).apply(process_group)

def calc_entropy(group_df):
    # Sort by 'pred' descending
    sorted_preds = group_df.sort_values("pred", ascending=False)

    # Take top 6 scores
    top_preds = sorted_preds["pred"].values[:6]

    if len(top_preds) < 5:
        # Not enough candidates; return maximum entropy (most uncertain)
        return 1.0

    
    # Min-max scaling on top 5 using min/max of top 6
    scale_min = np.min(top_preds)     # min of first 6
    scale_max = np.max(top_preds)     # max of first 6
    scale_range = scale_max - scale_min

    if scale_range == 0:
        return 0.0  # or 1.0 if identical values should imply max uncertainty

    # Take only top 5
    top_5 = top_preds[:5]

    # Min-max normalize to [0, 1]
    scaled = (top_5 - scale_min) / scale_range

    # Normalize to sum to 1 (turn into distribution)
    if np.sum(scaled) == 0:
        return 1.0  # uniform probability assumption
    probs = scaled / np.sum(scaled)

    ent = entropy(probs)

    return ent / np.log2(5)

def process_folder(folder_path, prefixes, thresholded_label):
    configs = {}
    for prefix in prefixes:
        # Extract number from prefix like "confidences_1_"
        prefix_number = prefix.split('_')[1]

        for file in Path(folder_path).glob(f"{prefix}*.csv"):
            config_name = file.stem.split(prefix)[-1]
            df = pd.read_csv(file)

            # Prefix the qid values (as strings or integers)
            df["qid"] = df["qid"].apply(lambda x: f"{prefix_number}_{x}")

            entropy_rows = []
            for _, group in df.groupby("qid"):
                labeled_df = get_labels(group.copy(), thresholded_label=thresholded_label)
                e = calc_entropy(labeled_df)
                entropy_rows.append({
                    'qid': group['qid'].iloc[0],
                    'entropy': e,
                    'label': labeled_df['result'].iloc[0]
                })

            entropy_df = pd.DataFrame(entropy_rows)
            if config_name in configs.keys():
                existing_df = configs[config_name]
                expanded_df=pd.concat([existing_df, entropy_df])
                configs[config_name] = expanded_df
            else:
                configs[config_name] = entropy_df
    return configs



def evaluate_thresholds_with_abstention(known_thresh, unknown_thresh, known_data, unknown_data):
    results = []
    all_results= []
    all_configs = sorted(set(known_data.keys()) & set(unknown_data.keys()))

    for config in all_configs:
        known_df = known_data[config].copy()
        # print(known_df.columns)

        known_df["true"] = known_df["label"]  # ✅ this is your gold label now
        # print(known_df["true"][0])
        unknown_df = unknown_data[config].copy() # unknowns are still treated as negative class
        unknown_df["true"] = unknown_df["label"]
        # print(unknown_df["true"][0])
        combined_df = pd.concat([known_df, unknown_df]).copy()
        # print(combined_df.head())

        known_val = known_thresh.loc[config]['threshold'] if config in known_thresh.index else None
        # print(f'known val {known_val}')
        unknown_val = unknown_thresh.loc[config]['threshold'] if config in unknown_thresh.index else None
        # print(f'unknown val {unknown_val}')
        base_prec = known_thresh.loc[config]['precision'] if config in known_thresh.index else None


        def predict(ent):
            # if unknown_val <= known_val:
            #     [print('chose 0 due to difference')]
            #     return 0
            if known_val != None and ent <= known_val:
                # print(f'chose 1 for this ent {ent} and for this threshold{known_val}')
                return 1
            elif unknown_val != None and ent >= unknown_val:
                # print(f'chose -1 for this ent {ent} and for this threshold{unknown_val}')
                return -1
            elif unknown_val != None and known_val != None and ent <= unknown_val and ent >= known_val:
                # print((f'chose inner for this ent {ent} and for this known threshold{known_val} and unkown {unknown_val}'))
                return 0
            else:
                return 0
        count_pos=0
        count_neg=0
        neutral=0
        for item in combined_df["entropy"]:
            if known_val != None and item < known_val:
                count_pos+=1
            elif unknown_val != None and item > unknown_val:
                count_neg+=1
            else:
                neutral+=1

        combined_df["pred"] = combined_df["entropy"].apply(predict)

        # known
        tp_known = ((combined_df["pred"] == 1) & ((combined_df["true"].astype(int)) == 1)).sum()

        fp_known = ((combined_df["pred"] == 1) & ((combined_df["true"].astype(int)) != 1)).sum()

        fn_known = ((combined_df["pred"] != 1) & ((combined_df["true"].astype(int)) == 1)).sum()


        precision_known = tp_known / (tp_known + fp_known) if (tp_known + fp_known) > 0 else 0
        recall_known = tp_known / (tp_known + fn_known) if (tp_known + fn_known) > 0 else 0
        precision_delta = precision_known - base_prec if base_prec is not None else np.nan

        # unknown
        tp_unk = ((combined_df["pred"] == -1) & ((combined_df["true"].astype(int)) == -1)).sum()
        fp_unk = ((combined_df["pred"] == -1) & ((combined_df["true"].astype(int)) != -1)).sum()
        fn_unk = ((combined_df["pred"] != -1) & ((combined_df["true"].astype(int)) == -1)).sum()

        precision_unk = tp_unk / (tp_unk + fp_unk) if (tp_unk + fp_unk) > 0 else 0
        recall_unk = tp_unk / (tp_unk + fn_unk) if (tp_unk + fn_unk) > 0 else 0

        # null_pred
        abstention_ratio = (combined_df["pred"] == 0).sum() / len(combined_df)

        results.append({
            "config": config,
            "precision_known": precision_known,
            "recall_known": recall_known,
            "precision_delta": precision_delta,
            "precision_unknown": precision_unk,
            "recall_unknown": recall_unk,
            "abstention_ratio": abstention_ratio
        })

        config_input=[]
        for i in range(len(combined_df)):
            config_input.append(config)

        combined_df['config']=config_input
        # print(combined_df.columns)
        all_results.append({config: combined_df})




    return pd.DataFrame(results), all_results



def plot_metrics_heatmap(df, metric, outdir, prefix_name=None):

    df[['speakers', 'instances']] = df['config'].str.extract(r'(\d+)s_(\d+)i').astype(int)
    heatmap_data = df.pivot_table(index="speakers", columns="instances", values=metric)

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="mako", linewidths=0.5)
    plt.title(f"{metric.replace('_', ' ').title()} Heatmap" + (f" (Prefix {prefix_name})" if prefix_name else ""))
    plt.xlabel("Instances per Speaker")
    plt.ylabel("Number of Speakers")
    plt.tight_layout()

    filename = f"{metric}_heatmap{'_' + prefix_name if prefix_name else ''}.png"
    plt.savefig(Path(outdir) / filename)
    plt.close()

def main(known_dir, unknown_dir, known_thresh_path, unknown_thresh_path, outdir, prefixes):
    os.makedirs(outdir, exist_ok=True)

    known_thresholds = pd.read_csv(known_thresh_path)
    if "config" in known_thresholds.columns:
        known_thresholds.set_index("config", inplace=True)

    unknown_thresholds = pd.read_csv(unknown_thresh_path)
    if "config" in unknown_thresholds.columns:
        unknown_thresholds.set_index("config", inplace=True)

    final_results_df=pd.DataFrame(columns=['qid', 'config', 'label', 'entropy', 'gold'])

    # Process all files with given prefixes
    known_data = process_folder(known_dir, prefixes, thresholded_label='known')


    unknown_data = process_folder(unknown_dir, prefixes, thresholded_label='unknown')



    # Evaluate
    results_df, all_results = evaluate_thresholds_with_abstention(
        known_thresholds, unknown_thresholds, known_data, unknown_data
    )
    # print("All results:", all_results)
    final_results = pd.DataFrame()

    # Loop through all results
    for item in all_results:


        # item is a dictionary, where the value is a DataFrame
        df = item[list(item.keys())[0]]  # Extract the DataFrame from the dictionary



        # Concatenate it to final_results
        if final_results.empty:
            final_results = df
        else:
            final_results = pd.concat([final_results, df], ignore_index=True)
    if results_df.empty:
        print("⚠️ No valid results to save or plot.")
        return

    # Save results
    results_df.to_csv(Path(outdir) / "final_threshold_results.csv", index=False)
    final_results.to_csv(Path(outdir) / 'final_results.csv', index=False)


    
    metrics = [
        "precision_known", "recall_known", "precision_delta",
        "precision_unknown", "recall_unknown", "abstention_ratio"
    ]
        # === Additional: Create per-prefix heatmaps ===
    df_split = split_by_prefix(results_df.copy())

    for prefix, sub_df in df_split.groupby("prefix"):
        for metric in metrics:
            plot_metrics_heatmap(sub_df.copy(), metric, outdir, prefix_name=prefix)

    for metric in metrics:
        plot_metrics_heatmap(df_split.copy(), metric, outdir, prefix_name=None)
    # Plot all metrics

    for metric in metrics:
        plot_metrics_heatmap(results_df, metric, outdir)


    print(f" All results saved to: {outdir}")


# main(
#     known_dir="EXPLICIT_ENSEMBLE_final/confidences_phonetic_known_test/",
#     unknown_dir="EXPLICIT_ENSEMBLE_final/confidences_phonetic_unknown_test/",
#     known_thresh_path="EXPLICIT_ENSEMBLE_final/threshold_results_phonetic_k_new.csv",
#     unknown_thresh_path="EXPLICIT_ENSEMBLE_final/threshold_results_phonetic_uk_new.csv",
#     outdir="EXPLICIT_ENSEMBLE_final/metric_results_phonetic_test_new_updated",
#     prefixes=["confidences_1_", "confidences_2_", "confidences_3_"]  # example
# )

# # main(
# #     known_dir="EXPLICIT_ENSEMBLE_final/confidences_spectral_known_test/",
# #     unknown_dir="EXPLICIT_ENSEMBLE_final/confidences_spectral_unknown_test/",
# #     known_thresh_path="EXPLICIT_ENSEMBLE_final/threshold_results_spectral_k_new.csv",
# #     unknown_thresh_path="EXPLICIT_ENSEMBLE_final/threshold_results_spectral_uk_new.csv",
# #     outdir="EXPLICIT_ENSEMBLE_final/metric_results_spectral_test_new",
# #     prefixes=["confidences_1_", "confidences_2_", "confidences_3_"]  # example
# # )

# main(
#     known_dir="EXPLICIT_ENSEMBLE_final/confidences_spectral_known_test/",
#     unknown_dir="EXPLICIT_ENSEMBLE_final/confidences_spectral_unknown_test/",
#     known_thresh_path="EXPLICIT_ENSEMBLE_final/threshold_results_spectral_k_new.csv",
#     unknown_thresh_path="EXPLICIT_ENSEMBLE_final/threshold_results_spectral_uk_new.csv",
#     outdir="EXPLICIT_ENSEMBLE_final/metric_results_spectral_test_new_updated",
#     prefixes=["confidences_1_", "confidences_2_", "confidences_3_"]  # example
