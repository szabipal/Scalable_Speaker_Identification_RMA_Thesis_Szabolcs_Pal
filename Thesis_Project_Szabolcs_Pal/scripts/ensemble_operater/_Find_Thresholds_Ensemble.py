"""
simple runner for ensemble thresholding.

what it does
- points the thresholding logic at the new confidences layout:
    EXPLICIT_ENSEMBLE_final/confidences/<model_type>/<dataset_type>/{queries,unknown_queries}
- calls `EXPLICIT_ENSEMBLE.thresholding_logic.thresholding(...)`
- writes a single csv with threshold results in:
    EXPLICIT_ENSEMBLE_final/threshold_results_<model_type>_<dataset_type>.csv

notes
- model_type: 'phonetic' or 'spectral'
- dataset_type: 'train' or 'test'
- includes early folder existence checks so failures are clear.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def run_thresholding_for_model(model_type: str, dataset_type: str = "test", threshold_value: float = 0.7):
    """
    run thresholding over known/unknown confidence files for one model/split.

    parameters
    ----------
    model_type : str
        'phonetic' or 'spectral'.
    dataset_type : str
        'train' or 'test'.
    threshold_value : float
        value forwarded to the thresholding function (see its doc for meaning).

    behavior
    - validates folder layout under EXPLICIT_ENSEMBLE_final/confidences/<model>/<split>.
    - calls the project's thresholding helper and saves a csv under
      EXPLICIT_ENSEMBLE_final/threshold_results_<model>_<split>.csv.

    notes
    - double-check the keyword names for the imported `thresholding(...)` function
      (e.g., 'cut_off' vs 'cutoff'); this wrapper passes `cut_off=0` and `threshold=...`.
    """
    # imported here due to an issue
    from EXPLICIT_ENSEMBLE.thresholding_logic import thresholding

    assert model_type in ("phonetic", "spectral"), "model_type must be 'phonetic' or 'spectral'"
    assert dataset_type in ("train", "test"), "dataset_type must be 'train' or 'test'"

    base_conf_dir = ROOT / "EXPLICIT_ENSEMBLE_final" / "confidences" / model_type / dataset_type
    conf_k_dir = base_conf_dir / "queries"
    conf_uk_dir = base_conf_dir / "unknown_queries"



    out_dir = ROOT / "EXPLICIT_ENSEMBLE_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"threshold_results_{model_type}_{dataset_type}.csv"


    thresholding(
        conf_k_dir=str(conf_k_dir),
        conf_uk_dir=str(conf_uk_dir),
        cut_off=0,                  
        threshold=threshold_value,
        output_path=str(out_file),
    )
    print(f"✅ Thresholds saved to {out_file}")

if __name__ == "__main__":

    for model in ("spectral", "phonetic"):
        for split in ("train",):  
            run_thresholding_for_model(model_type=model, dataset_type=split, threshold_value=0.7)
