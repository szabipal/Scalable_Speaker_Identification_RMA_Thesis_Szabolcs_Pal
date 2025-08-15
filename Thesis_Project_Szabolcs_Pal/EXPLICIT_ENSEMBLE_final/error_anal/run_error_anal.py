from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from collections import Counter
import re

from certainty_whole_segm import analyze_certainty_threshold
from chunk_level_accuracy import analyze_chunkwise_correct_predictions
from entropy_dist_plots import generate_entropy_distribution_plots
from conf_matrix_whole import generate_confusion_matrices_by_group

# === MAIN PIPELINE ===
def run_full_error_analysis_pipeline(df, base_output_dir):
    base_output_dir = Path(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    generate_confusion_matrices_by_group(df.copy(), base_output_dir / "confusion_matrices")
    generate_entropy_distribution_plots(df.copy(), base_output_dir / "entropy_distributions")
    analyze_chunkwise_correct_predictions(df.copy(), base_output_dir / "chunkwise_accuracy")
    analyze_certainty_threshold(df.copy(), base_output_dir / "certainty_thresholds")
    analyze_top_chunks_per_base(df.copy(), base_output_dir / "most_informative_chunks")


df = pd.read_csv("EXPLICIT_ENSEMBLE_final/metric_results_phonetic_test_new/final_results.csv")
final_output_dir = Path("EXPLICIT_ENSEMBLE_final/Phonetic_error_anal")
run_full_error_analysis_pipeline(df, final_output_dir)

df = pd.read_csv("EXPLICIT_ENSEMBLE_final/metric_results_spectral_test_new/final_results.csv")
final_output_dir = Path("EXPLICIT_ENSEMBLE_final/Spectral_error_anal")
run_full_error_analysis_pipeline(df, final_output_dir)

# df = pd.read_csv("TRANSFORMERS_BASED_MODEL/final_eval_TEST_new3/thresholded_predictions.csv")
# final_output_dir = Path("TRANSFORMERS_BASED_MODEL/final_eval_TEST_new3/Transformers_error_anal")
# df = df.rename(columns={
#     "normalized_entropy": "entropy",
#     "source_file": "config"
# })
# df = df.loc[:, ~df.columns.duplicated()]
# # Optional: create a fake 'pred' column if needed for confusion matrices
# # df["pred"] = df["label"]  # or use your own prediction logic
# run_full_error_analysis_pipeline(df, final_output_dir)
