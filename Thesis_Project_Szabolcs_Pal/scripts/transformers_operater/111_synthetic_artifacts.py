#!/usr/bin/env python3
"""
Create a minimal synthetic tree so the transformers pipeline can be validated
without running embeddings, training, eval, or entropy in reality.

It writes:
- enrollment_sets/train|test/... with 1 tiny JSON per type
- data/processed_* with 1 dummy .npy waveform
- data/speaker_pairs/train_pairs.csv + val_pairs.csv (tiny)
- hubert_embeddings_*/*/grid_1/*.csv (tiny, with correct columns)
- siamese_eval_results/*/*/grid_1/*.csv (tiny summaries)
- entropy_results_transformers/*/entropy_*.csv (tiny, with correct columns)
- thresholds_transformers/transformers_thresholds_train.csv (fake thresholds)

After this, your validators and “final testing using train thresholds” can run
without error, proving wiring and plotting work.
"""
from pathlib import Path
import os, json, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]

def ensure(p: Path): p.mkdir(parents=True, exist_ok=True); return p

def write_json(p: Path, obj): ensure(p.parent); p.write_text(json.dumps(obj, indent=2))

def write_csv(p: Path, df: pd.DataFrame): ensure(p.parent); df.to_csv(p, index=False)

def main():
    # 1) minimal enrollment_sets
    for split in ["train", "test"]:
        base = ROOT / "enrollment_sets" / split
        for kind in ["grid_1", "query_1", "unknown_query_1"]:
            write_json(base / kind / "10s_10i.json", {
                "speakers": [
                    {"speaker_id": "19", "files": ["19-198-0000_chunk0.npy"]},
                    {"speaker_id": "24", "files": ["24-100-0001_chunk0.npy"]},
                ]
            })

    # 2) minimal wave dirs with one npy each
    for d in [ROOT / "data/processed_train/wave_chunks_2s",
              ROOT / "data/processed_test/wave_chunks_2s"]:
        ensure(d)
        for fname in ["19-198-0000_chunk0.npy", "24-100-0001_chunk0.npy"]:
            np.save(d / fname, np.zeros(16000, dtype=np.float32))  # 1s of zeros

    # 3) siamese train/val pairs (tiny)
    pairs_dir = ROOT / "data/speaker_pairs"
    write_csv(pairs_dir / "train_pairs.csv", pd.DataFrame({
        "x1_path": ["emb_a.npy"], "x2_path": ["emb_b.npy"], "label": [1]
    }))
    write_csv(pairs_dir / "val_pairs.csv", pd.DataFrame({
        "x1_path": ["emb_c.npy"], "x2_path": ["emb_d.npy"], "label": [0]
    }))

    # 4) fake embeddings CSVs with correct columns
    def emb_rows(spk):
        return [{
            "speaker_id": spk, "session_id": "198", "instance_id": "0000", "chunk_id": "0",
            "embedding": np.random.randn(768).astype(float).tolist()
        }]
    for split in ["train", "test"]:
        base = ROOT / f"hubert_embeddings_{split}"
        for sub in ["grids", "queries", "unknown_queries"]:
            df = pd.DataFrame(emb_rows("19") + emb_rows("24"))
            write_csv(base / sub / "grid_1" / "10s_10i.csv", df)

    # 5) fake siamese eval summaries (what your entropy step expects)
    def eval_rows():
        # ensure columns: query_id, query_speaker, enrolled_speaker, sum, count, match_rate
        rows = []
        for qspk in ["19","24"]:
            for espk in ["19","24"]:
                rows.append({
                    "query_id": f"{qspk}_198_0000_chunk0",
                    "query_speaker": qspk,
                    "enrolled_speaker": espk,
                    "sum": 10 if qspk==espk else 4,
                    "count": 10,
                    "match_rate": 1.0 if qspk==espk else 0.4
                })
        return rows

    for split in ["train", "test"]:
        for qtype in ["queries", "unknown_queries"]:
            df = pd.DataFrame(eval_rows())
            write_csv(ROOT / "siamese_eval_results" / split / qtype / "grid_1" / "10s_10i.csv", df)

    # 6) fake entropy results (known+unknown) with expected columns
    def entropy_rows(label_known=True):
        rows = []
        for qspk in ["19","24"]:
            # pretend top match equals true for one, not the other
            top = qspk if label_known else ("19" if qspk=="24" else "24")
            rows.append({
                "query_id": f"{qspk}_198_0000_chunk0",
                "top_match_id": top,
                "normalized_entropy": float(np.clip(np.random.rand(), 0.05, 0.95)),
                "source_file": "10s_10i",
                "grid_number": "1",
                "label": 1 if label_known else 0
            })
        return rows

    for split in ["train", "test"]:
        write_csv(ROOT / "entropy_results_transformers" / split / f"entropy_transformers_{split}_known.csv",
                  pd.DataFrame(entropy_rows(label_known=True)))
        write_csv(ROOT / "entropy_results_transformers" / split / f"entropy_transformers_{split}_unknown.csv",
                  pd.DataFrame(entropy_rows(label_known=False)))

    # 7) fake train thresholds per config (matches source_file)
    thr = pd.DataFrame([{
        "config": "10s_10i",
        "known_threshold": 0.30,
        "known_train_precision": 0.9,
        "known_train_recall": 0.9,
        "unknown_threshold": 0.70,
        "unknown_train_precision": 0.9,
        "unknown_train_recall": 0.9
    }])
    write_csv(ROOT / "thresholds_transformers" / "transformers_thresholds_train.csv", thr)

    print("✅ Synthetic artifacts created. You can now run:")
    print("   - validator")
    print("   - final testing (using train thresholds on test)")
    print("   - plotting/heatmaps")
    print("…without running any heavy compute.")

if __name__ == "__main__":
    main()
