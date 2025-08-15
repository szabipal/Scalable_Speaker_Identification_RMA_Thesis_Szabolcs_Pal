"""
driver to generate speaker embeddings for both spectral and phonetic pipelines.

what it does
- for a given split (train/test/dev), figures out where to read processed inputs
  and where to write embeddings.
- spectral: calls `generate_all_model_embeddings.main` per feature type, optionally
  filtered by grid/query/unknown_query jsons.
- phonetic: calls `explicit_feature_extractor.main_explicit` across feature types
  using the processors dict, with the same grid handling.
- includes a tiny test entrypoint that runs a dev phonetic pass.

notes
- expects the enrollment_sets/<split> structure for train/test (grid_*, query_*,
  unknown_query_*). dev runs over full data (no grids).
- feature roots are under data/processed_<split>/feature/<rep>/<feature_type>.
"""

import os
from pathlib import Path
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] 
# Spectral embedding script
from scripts.evaluation.generate_all_model_embeddings import main as main_spectral

# Phonetic embedding script
from scripts.evaluation.explicit_feature_extractor import main_explicit, processors


FEATURE_TYPES = ["mfcc", "low", "mid", "high"]


def _grid_roots_for(split: str) -> list[str] | None:
    """
    resolve the list of grid-like folders for a given split.

    dev → returns None (meaning: process full data, no grid filtering).
    train/test → returns a list of folders like:
      enrollment_sets/<split>/(grid_*|query_*|unknown_query_*)

    returns
    -------
    list of str paths, or None when there are no grids for this split.
    """
    if split == "dev":
        return None

    base = Path("enrollment_sets") / split
    # Expect numbered folders like grid_1.., query_1.., unknown_query_1..
    roots = []
    # how many sets? infer by scanning directories that match the naming
    for kind in ["grid", "query", "unknown_query"]:
        for p in sorted(base.glob(f"{kind}_*")):
            if p.is_dir():
                roots.append(str(p))
    return roots


def _processed_base_for(split: str) -> str:
    """
    map split name to its processed data base folder: data/processed_<split>.

    raises on unknown split to fail fast.
    """
    # data/processed_<split>
    if split not in {"train", "test", "dev"}:
        raise ValueError(f"Unknown split: {split}")
    return f"data/processed_{split}"


def generate_embeddings(model_type: str, dataset_type: str):
    """
    generate embeddings for a chosen model family and dataset split.

    parameters
    ----------
    model_type : 'spectral' | 'phonetic'
        which pipeline to run.
    dataset_type : 'train' | 'test' | 'dev'
        which split to read from and write to.

    behavior
    - sets input/output roots based on split and model type.
    - if grids exist (train/test), iterates each json inside grid/query/unknown_query folders.
    - forwards the correct arguments to the underlying spectral/phonetic runners.
    """
    assert model_type in ["spectral", "phonetic"], "❌ Invalid model_type"
    assert dataset_type in ["train", "test", "dev"], "❌ Invalid dataset_type"

    # === Where to read features from
    processed_base = _processed_base_for(dataset_type)  # e.g., data/processed_train

    # === Where to write embeddings
    output_root = (
        f"embeddings_spectral_final_{dataset_type}"
        if model_type == "spectral"
        else f"embeddings_phonetic_final_{dataset_type}"
    )

    # === Where to find grids/queries/unknowns (dev = full data, no grids)
    grid_roots = _grid_roots_for(dataset_type)

    # === SPECTRAL ===
    if model_type == "spectral":
        print(f"Generating spectral embeddings for {dataset_type} set")

        # For DEV: embed full dataset per feature type (no grid filtering)
        if grid_roots is None:
            for ft in FEATURE_TYPES:
                input_base_path = f"{processed_base}/feature/spectral/{ft}"
                print(f"[DEV] spectral/{ft} → {input_base_path}")
                main_spectral(
                    trained_models_dir="trained_models_spectral_large_data",
                    output_root=output_root,
                    input_base_path=input_base_path,   # leaf path: …/spectral/<ft>
                    grid_path=None
                )
        else:
            # For TRAIN/TEST: iterate grid, query, unknown_query numbered folders
            for root in grid_roots:
                json_files = [f for f in os.listdir(root) if f.endswith(".json")]
                if not json_files:
                    print(f"⚠️ No JSON files found in {root}")
                for jf in json_files:
                    full_grid_path = os.path.join(root, jf)
                    # Run once per feature type, pointing input_base_path at the leaf dir
                    for ft in FEATURE_TYPES:
                        input_base_path = f"{processed_base}/feature/spectral/{ft}"
                        print(f"🔄 {root} :: spectral/{ft} :: {full_grid_path}")
                        main_spectral(
                            trained_models_dir="trained_models_spectral_large_data",
                            output_root=output_root,
                            grid_path=full_grid_path,
                            input_base_path=input_base_path
                        )

    # === PHONETIC ===
    elif model_type == "phonetic":
        print(f"🔍 Generating phonetic embeddings for {dataset_type} set")

        # Build feature-type → input path dict using the required structure
        # data/processed_<split>/feature/phonetic/<feature_type>
        processors_dict = processors  # dict mapping feature_type → extractor/processor
        input_dirs_dict = {
            ft: f"{processed_base}/feature/phonetic/{ft}"
            for ft in processors_dict.keys()
        }

        if grid_roots is None:
            # DEV: run once over full dataset
            print(f"🔄 [DEV] phonetic over: {input_dirs_dict}")
            main_explicit(
                processors_dict=processors_dict,
                input_dirs_dict=input_dirs_dict,
                output_root=output_root,
                grid_path=None
            )
        else:
            # TRAIN/TEST: iterate each numbered grid folder and each JSON within
            for root in grid_roots:
                json_files = [f for f in os.listdir(root) if f.endswith(".json")]
                if not json_files:
                    print(f"⚠️ No JSON files found in {root}")
                for jf in json_files:
                    full_grid_path = os.path.join(root, jf)
                    print(f"🔄 {root} :: {full_grid_path}")
                    main_explicit(
                        processors_dict=processors_dict,
                        input_dirs_dict=input_dirs_dict,
                        output_root=output_root,
                        grid_path=full_grid_path
                    )


def test_generate_embeddings():
    """
    quick check: run a dev phonetic pass.

    useful during refactors to confirm wiring and paths without relying on
    train/test grid folders.
    """
    # Example: run dev phonetic
    print("\n=== Running test for Phonetic Model (dev set) ===")
    generate_embeddings(model_type="phonetic", dataset_type="dev")


if __name__ == "__main__":
    test_generate_embeddings()




