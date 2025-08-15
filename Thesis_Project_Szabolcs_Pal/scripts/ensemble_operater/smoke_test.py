# scripts/ensemble_operater/pipeline_smoke_test.py
from pathlib import Path
import sys, os, glob, json
ROOT = Path(__file__).resolve().parents[2]

from scripts.ensemble_operater.pipeline_config import (
    ROOT, all_configs, test_cases, train_cases, grid_map, FEATURE_FOLDER_MAP
)


def _fail(msg): 
    raise SystemExit(f"❌ {msg}")

def _warn(msg):
    print(f"⚠️  {msg}")

def _ok(msg):
    print(f"✅ {msg}")

def _expect_dir(p: Path, strict=True):
    if not p.exists() or not p.is_dir():
        if strict:
            _fail(f"missing directory: {p}")
        else:
            _warn(f"missing directory: {p}")
            return False
    return True

def _expect_file(p: Path, strict=True):
    if not p.exists() or not p.is_file():
        if strict:
            _fail(f"missing file: {p}")
        else:
            _warn(f"missing file: {p}")
            return False
    return True

def check_dev_quantiles(model: str):
    cfg = all_configs[model]
    quant_dir = Path(cfg["quantile_dir"])
    features = [FEATURE_FOLDER_MAP.get(f, f) for f in cfg["feature_types"]]
    _expect_dir(quant_dir, strict=False)
    missing = []
    for feat in features:
        f = quant_dir / f"quantiles_{feat}.json"
        if not _expect_file(f, strict=False):
            missing.append(str(f))
        else:
            # basic schema sniff
            try:
                j = json.loads(Path(f).read_text())
                assert "intra" in j and "inter" in j
            except Exception as e:
                _fail(f"bad quantiles json schema: {f} ({e})")
    if missing:
        _warn(f"quantiles missing for: {', '.join(missing)}")
    else:
        _ok(f"quantiles present for all features in {quant_dir}")

def check_embeddings_tree(model: str, split: str, cases: list[int]):
    assert split in ("train", "test")
    cfg = all_configs[model]
    base = Path(cfg["train_embedding_root"] if split == "train" else cfg["embedding_root"])
    feats = [FEATURE_FOLDER_MAP.get(f, f) for f in cfg["feature_types"]]
    _expect_dir(base)

    for c in cases:
        grid_root = base / f"grid_{c}"
        query_root = base / f"query_{c}"
        unk_root = base / f"unknown_query_{c}"

        for root in (grid_root, query_root, unk_root):
            if not _expect_dir(root, strict=False):
                _warn(f"{split} case {c}: missing {root}; downstream builds will skip/fail.")

        # check per-feature subfolders exist (don’t force strict, but warn loudly)
        for root in (grid_root, query_root, unk_root):
            if not root.exists():
                continue
            for feat in feats:
                p = root / feat
                if not p.exists():
                    _warn(f"{split} case {c}: missing feature folder: {p}")
                else:
                    # check folder isn’t totally empty
                    has_npy = any(p.glob("**/*.npy"))
                    if not has_npy:
                        _warn(f"{split} case {c}: no npy files under {p}")

    _ok(f"{split} embeddings tree check completed for {model}")

def check_lambdamart_outputs(model: str):
    cfg = all_configs[model]
    feats = [FEATURE_FOLDER_MAP.get(f, f) for f in cfg["feature_types"]]

    # TEST OUTPUTS
    for c in test_cases:
        # queries
        out_dir = Path(cfg["output_root"]) / f"lambdamart_cosine_dataset_queries_normed{c}"
        _expect_dir(out_dir, strict=False)
        for gid, grid_name in grid_map.items():
            path = out_dir / grid_name
            if not _expect_dir(path, strict=False):
                continue
            csvs = list(path.glob("*.csv"))
            if not csvs:
                _warn(f"no csv produced for test queries case {c} grid {grid_name} under {path}")
        # unknown queries
        out_dir = Path(cfg["output_root"]) / f"lambdamart_cosine_dataset_unknown_queries_normed{c}"
        _expect_dir(out_dir, strict=False)
        for gid, grid_name in grid_map.items():
            path = out_dir / grid_name
            if not _expect_dir(path, strict=False):
                continue
            csvs = list(path.glob("*.csv"))
            if not csvs:
                _warn(f"no csv produced for test unknown case {c} grid {grid_name} under {path}")

    # TRAIN OUTPUTS
    for c in train_cases:
        out_dir = Path(cfg["train_output_root"]) / f"lambdamart_cosine_dataset_queries_train_normed{c}"
        _expect_dir(out_dir, strict=False)
        for gid, grid_name in grid_map.items():
            path = out_dir / grid_name
            if not _expect_dir(path, strict=False):
                continue
            csvs = list(path.glob("*.csv"))
            if not csvs:
                _warn(f"no csv produced for train queries case {c} grid {grid_name} under {path}")

    _ok(f"lambdamart outputs check completed for {model}")

def main():
    models = list(all_configs.keys())  # ["spectral", "phonetic"]
    # 1) quantiles (dev)
    for m in models:
        print(f"\n=== checking DEV quantiles for {m} ===")
        check_dev_quantiles(m)

    # 2) embeddings trees for train and test
    for m in models:
        print(f"\n=== checking TEST embeddings for {m} ===")
        check_embeddings_tree(m, "test", cases=[1,2,3])  # align to your test_cases
        print(f"\n=== checking TRAIN embeddings for {m} ===")
        check_embeddings_tree(m, "train", cases=[1,2,3,4,5])  # align to your train_cases

    # 3) lambdamart outputs
    for m in models:
        print(f"\n=== checking lambdamart outputs for {m} ===")
        check_lambdamart_outputs(m)

    _ok("\nsmoke test finished")

if __name__ == "__main__":
    main()
