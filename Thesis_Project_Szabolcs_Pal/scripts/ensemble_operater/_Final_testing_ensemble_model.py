"""
driver for the final ensemble evaluation + error analysis.

what it does
- runs the ensemble on the new confidence/threshold folder structure and writes
  combined metrics to EXPLICIT_ENSEMBLE_final/metric_results_<model>_<split>_final/.
- then runs a set of error-analysis utilities (confusion matrices, entropy plots,
  chunkwise accuracy, certainty thresholds), optionally “most informative chunks”
  if available.

notes
- expects confidence csvs under:
    confidences/<model_type>/<dataset_type>/{queries,unknown_queries}/confidences_*_*.csv
  and threshold csvs:
    confidences/<model_type>/<dataset_type>/threshold_results_{k,uk}.csv
- model_type ∈ {"phonetic","spectral"}, dataset_type ∈ {"train","test"}.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] 
import pandas as pd

from EXPLICIT_ENSEMBLE_final.test_ensemble import main
from EXPLICIT_ENSEMBLE_final.error_anal.certainty_whole_segm import analyze_certainty_threshold
from EXPLICIT_ENSEMBLE_final.error_anal.chunk_level_accuracy import analyze_chunkwise_correct_predictions
from EXPLICIT_ENSEMBLE_final.error_anal.entropy_dist_plots import generate_entropy_distribution_plots
from EXPLICIT_ENSEMBLE_final.error_anal.conf_matrix_whole import generate_confusion_matrices_by_group

def run_final_ensemble_test(model_type: str, dataset_type: str = "test"):
    """
    run the ensemble step using the new directory layout and write final metrics.

    folder expectations
    - confidences/<model_type>/<dataset_type>/queries/
    - confidences/<model_type>/<dataset_type>/unknown_queries/
    - confidences/<model_type>/<dataset_type>/threshold_results_k.csv
    - confidences/<model_type>/<dataset_type>/threshold_results_uk.csv

    outputs
    - EXPLICIT_ENSEMBLE_final/metric_results_<model_type>_<dataset_type>_final/

    parameters
    ----------
    model_type : 'phonetic' | 'spectral'
    dataset_type : 'train' | 'test'
    """
    assert model_type in {"phonetic", "spectral"}
    assert dataset_type in {"train", "test"}

    base_path = Path("EXPLICIT_ENSEMBLE_final")
    conf_base = base_path / "confidences" / model_type / dataset_type

    known_dir = conf_base / "queries"
    unknown_dir = conf_base / "unknown_queries"
    known_thresh_path = conf_base / "threshold_results_k.csv"
    unknown_thresh_path = conf_base / "threshold_results_uk.csv"

    # infer dataset ids from filenames like confidences_<id>_<grid>.csv
    present_ids = sorted(
        {p.name.split("_", 2)[1]  # extract the <id> in confidences_<id>_<grid>.csv
         for p in known_dir.glob("confidences_*_*.csv")}
    )
    # fallback to 1..3 if no files found
    prefixes = [f"confidences_{i}_" for i in (present_ids if present_ids else ["1","2","3"])]

    outdir = base_path / f"metric_results_{model_type}_{dataset_type}_final"
    outdir.mkdir(parents=True, exist_ok=True)

    # delegate to the ensemble entrypoint
    main(
        known_dir=str(known_dir) + "/",            # keep trailing slash if main expects it
        unknown_dir=str(unknown_dir) + "/",
        known_thresh_path=str(known_thresh_path),
        unknown_thresh_path=str(unknown_thresh_path),
        outdir=str(outdir),
        prefixes=prefixes
    )

# === MAIN ERROR-ANALYSIS PIPELINE ===
def run_full_error_analysis_pipeline(df: pd.DataFrame, base_output_dir: Path | str):
    """
    run all error-analysis steps on the given results dataframe and write plots/tables.

    steps
    - confusion matrices by group
    - entropy distributions
    - chunkwise accuracy
    - certainty-threshold curves
    - (optional) most informative chunks if the helper is available

    parameters
    ----------
    df : pandas.DataFrame
        ensemble results (e.g., final_results.csv).
    base_output_dir : path-like
        folder where the analysis artifacts will be written.
    """
    base_output_dir = Path(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    generate_confusion_matrices_by_group(df.copy(), base_output_dir / "confusion_matrices")
    generate_entropy_distribution_plots(df.copy(), base_output_dir / "entropy_distributions")
    analyze_chunkwise_correct_predictions(df.copy(), base_output_dir / "chunkwise_accuracy")
    analyze_certainty_threshold(df.copy(), base_output_dir / "certainty_thresholds")
    # optional: only if this module exists
    try:
        from EXPLICIT_ENSEMBLE_final.error_anal.most_informative_chunks import analyze_top_chunks_per_base
        analyze_top_chunks_per_base(df.copy(), base_output_dir / "most_informative_chunks")
    except Exception:
        pass

def run_error_analysis_for_all_models(dataset_type: str = "test"):
    """
    run ensemble (if needed) and error analysis for both model families.

    parameters
    ----------
    dataset_type : 'train' | 'test'
        which split to process.
    """
    for model_type in ["phonetic", "spectral"]:
        # ensure ensemble results exist; if not, run the ensemble first
        result_path = Path(f"EXPLICIT_ENSEMBLE_final/metric_results_{model_type}_{dataset_type}_final/final_results.csv")
        if not result_path.exists():
            print(f"⚠️ Results not found for {model_type}: {result_path}. Running ensemble...")
            run_final_ensemble_test(model_type=model_type, dataset_type=dataset_type)

        if not result_path.exists():
            print(f" Still missing: {result_path}")
            continue

        output_path = Path(f"EXPLICIT_ENSEMBLE_final/{model_type.capitalize()}_error_anal_{dataset_type}")
        print(f" Running error analysis for: {model_type} ({dataset_type})")
        df = pd.read_csv(result_path)
        run_full_error_analysis_pipeline(df, output_path)
        print(f" Saved results to: {output_path}")

if __name__ == "__main__":
    # example: run full pipeline for TEST split using the new structure
    run_error_analysis_for_all_models(dataset_type="test")
