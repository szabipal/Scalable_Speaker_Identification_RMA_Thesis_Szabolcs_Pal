# import argparse
# import subprocess
# import sys
# from pathlib import Path

# ROOT = Path(__file__).resolve().parents[2]  # repo root (../../ from this file)

# RUNNERS = {
#     "embeddings": ROOT / "scripts/transformers_operater/111_Generate_hubert_embeddings.py",
#     "train":      ROOT / "scripts/transformers_operater/111_Run_siamese_training.py",
#     "evaluate":   ROOT / "scripts/transformers_operater/111_Evaluate_transformers_model.py",
#     "entropy":    ROOT / "scripts/transformers_operater/111_Run_entropy_calculation_transformers.py",
#     "thresholds": ROOT / "scripts/transformers_operater/111_Thresholding_transformers.py",
#     "final":      ROOT / "scripts/transformers_operater/111_FINAL_TESTING_TRANSFORMERS.py",
# }

# def run_step(name: str, path: Path, extra_args=None):
#     if not path.exists():
#         raise FileNotFoundError(f"[{name}] Runner not found: {path}")

#     # make sure it's executable; call with current Python if it's a .py, else run directly
#     if path.suffix == ".py":
#         cmd = [sys.executable, str(path)]
#     else:
#         cmd = [str(path)]
#     if extra_args:
#         cmd.extend(map(str, extra_args))

#     print(f"\n=== ▶ {name}\n    {' '.join(cmd)}")
#     subprocess.run(cmd, check=True)
#     print(f"=== ✔ {name} — done")

# def parse_args():
#     p = argparse.ArgumentParser(description="Run full transformers pipeline.")
#     p.add_argument("--skip-embeddings", action="store_true", help="Skip Stage 1")
#     p.add_argument("--skip-train",      action="store_true", help="Skip Stage 2")
#     p.add_argument("--skip-evaluate",   action="store_true", help="Skip Stage 3")
#     p.add_argument("--skip-entropy",    action="store_true", help="Skip Stage 4")
#     p.add_argument("--skip-thresholds", action="store_true", help="Skip Stage 5")
#     p.add_argument("--skip-final",      action="store_true", help="Skip Stage 6")
#     p.add_argument("--dry-run",         action="store_true", help="List steps without executing")
#     return p.parse_args()

# def main():
#     args = parse_args()

#     plan = []
#     if not args.skip_embeddings: plan.append(("Stage 1: Generate HuBERT embeddings", RUNNERS["embeddings"]))
#     if not args.skip_train:      plan.append(("Stage 2: Train Siamese model", RUNNERS["train"]))
#     if not args.skip_evaluate:   plan.append(("Stage 3: Evaluate Siamese model", RUNNERS["evaluate"]))
#     if not args.skip_entropy:    plan.append(("Stage 4: Compute entropy files", RUNNERS["entropy"]))
#     if not args.skip_thresholds: plan.append(("Stage 5: Compute thresholds (use TRAIN)", RUNNERS["thresholds"]))
#     if not args.skip_final:      plan.append(("Stage 6: FINAL TEST using TRAIN thresholds", RUNNERS["final"]))

#     print("Pipeline plan:")
#     for name, path in plan:
#         print(f"  - {name}: {path}")

#     if args.dry_run:
#         print("\n(dry-run) Exiting without execution.")
#         return

#     for name, path in plan:
#         run_step(name, path)

#     print("\n Pipeline complete.")

# if __name__ == "__main__":
#     main()

from pathlib import Path
import subprocess, sys, os

ROOT = Path(__file__).resolve().parents[2]

def run_step(name, module_name):
    print(f"\n=== ▶ {name}\n    {sys.executable} -m {module_name}")
    subprocess.run(
        [sys.executable, "-m", module_name],
        check=True,
        cwd=str(ROOT),                       # run from repo root
        env=dict(os.environ, PYTHONPATH=str(ROOT)),  # extra safety
    )

STAGES = [
    ("Stage 1: Generate HuBERT embeddings",       "scripts.transformers_operater.111_Generate_hubert_embeddings"),
    ("Stage 2: Train Siamese model",              "scripts.transformers_operater.111_Run_siamese_training"),
    ("Stage 3: Evaluate Siamese model",           "scripts.transformers_operater.111_Evaluate_transformers_model"),
    ("Stage 4: Compute entropy files",            "scripts.transformers_operater.111_Run_entropy_calculation_transformers"),
    ("Stage 5: Compute thresholds (use TRAIN)",   "scripts.transformers_operater.111_Thresholding_transformers"),
    ("Stage 6: FINAL TEST using TRAIN thresholds","scripts.transformers_operater.111_FINAL_TESTING_TRANSFORMERS"),
]

def main():
    print("Pipeline plan:")
    for name, mod in STAGES:
        print(f"  - {name}: {mod}")
    for name, mod in STAGES:
        run_step(name, mod)

if __name__ == "__main__":
    main()