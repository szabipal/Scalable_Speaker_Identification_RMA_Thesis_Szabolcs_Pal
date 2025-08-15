# scripts/transformers_operater/hubert_grid_extractor.py
import os, json, re, random, ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel


def _merge_per_grid_csvs_to_single_csv(output_base: Path, merged_csv_path: Path) -> Path:
    """
    scan output_base (which contains grid_1, grid_2, ...) and concatenate all
    per-config csvs into a single hubert_embeddings.csv with columns:
      [speaker_id, session_id, instance_id, chunk_id, embedding]
    """
    frames = []
    for grid_dir in sorted(output_base.glob("grid_*")):
        for csv_file in sorted(grid_dir.glob("*.csv")):
            df = pd.read_csv(csv_file)
            # ensure expected columns exist; skip bad artifacts
            req = {"speaker_id", "embedding"}
            if not req.issubset(df.columns):
                continue
            frames.append(df[["speaker_id", "session_id", "instance_id", "chunk_id", "embedding"]])

    if not frames:
        raise FileNotFoundError(f"no per-grid csvs found under {output_base}")

    merged = pd.concat(frames, ignore_index=True)
    # normalize embedding to a json-serializable string (list of floats)
    # if already a list-like, keep; if string of list, keep; if numpy, tolist()
    def _normalize(e):
        if isinstance(e, str):
            return e  # assumed json-like list string from your extractor
        if isinstance(e, (list, tuple, np.ndarray)):
            return np.asarray(e).tolist()
        return e
    merged["embedding"] = merged["embedding"].apply(_normalize)
    merged_csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_csv_path, index=False)
    print(f"✅ merged embeddings -> {merged_csv_path} ({len(merged)} rows)")
    return merged_csv_path


def create_speaker_split_datasets_from_embeddings_csv(
    csv_path: Path,
    output_dir: Path,
    val_ratio: float = 0.2,
    num_pairs: int = 10_000,
    seed: int = 42,
):
    """
    build {train_pairs.csv,val_pairs.csv} for siamese training from hubert_embeddings.csv.

    expects columns:
      - speaker_id (str/int)
      - embedding (json string list OR python-literal list)  → will be parsed to np.array
    output columns:
      - embedding_1 (json list)
      - embedding_2 (json list)
      - label (1 same speaker, 0 different)
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    df = pd.read_csv(csv_path)
    if "speaker_id" not in df.columns or "embedding" not in df.columns:
        raise ValueError(f"{csv_path} must contain 'speaker_id' and 'embedding' columns")

    # parse embeddings safely (prefer json; fall back to ast.literal_eval)
    def _parse(cell):
        if isinstance(cell, str):
            try:
                return np.array(json.loads(cell))
            except Exception:
                return np.array(ast.literal_eval(cell))  # e.g., "[0.1, 0.2, ...]"
        if isinstance(cell, (list, tuple, np.ndarray)):
            return np.array(cell)
        raise TypeError(f"cannot parse embedding cell: {cell!r}")

    df["embedding"] = df["embedding"].apply(_parse)

    # split by speakers
    speakers = df["speaker_id"].dropna().unique().tolist()
    if len(speakers) < 2:
        raise ValueError("need at least 2 speakers to create positive/negative pairs")

    from sklearn.model_selection import train_test_split
    train_speakers, val_speakers = train_test_split(speakers, test_size=val_ratio, random_state=seed)

    # group embeddings by speaker
    speaker_embeddings = {}
    for spk, rows in df.groupby("speaker_id"):
        speaker_embeddings[spk] = [e for e in rows["embedding"].tolist() if isinstance(e, np.ndarray)]

    def _gen_pairs(speaker_pool, label, target_count):
        pairs, attempts = [], 0
        pool = list(speaker_pool)
        if not pool:
            return pairs
        while len(pairs) < target_count and attempts < target_count * 50:
            attempts += 1
            if label == 1:
                spk = rng.choice(pool)
                embs = speaker_embeddings.get(spk, [])
                if len(embs) < 2:
                    continue
                a, b = rng.sample(embs, 2)
            else:
                if len(pool) < 2:
                    break
                spk1, spk2 = rng.sample(pool, 2)
                embs1 = speaker_embeddings.get(spk1, [])
                embs2 = speaker_embeddings.get(spk2, [])
                if not embs1 or not embs2:
                    continue
                a, b = rng.choice(embs1), rng.choice(embs2)
            pairs.append((a, b, label))
        return pairs

    def _save_pairs(pairs, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # serialize embeddings as json-friendly lists
        recs = [{"embedding_1": a.tolist(), "embedding_2": b.tolist(), "label": int(lbl)} for a, b, lbl in pairs]
        pd.DataFrame(recs).to_csv(path, index=False)
        print(f"✅ saved {len(recs)} pairs -> {path}")

    half = num_pairs // 2
    train_pairs = _gen_pairs(train_speakers, 1, half) + _gen_pairs(train_speakers, 0, half)
    val_pairs   = _gen_pairs(val_speakers,   1, half) + _gen_pairs(val_speakers,   0, half)
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_pairs(train_pairs, output_dir / "train_pairs.csv")
    _save_pairs(val_pairs,   output_dir / "val_pairs.csv")


# ------------------------------------------------------------
# existing extractor (now with optional pair generation)
# ------------------------------------------------------------

def run_extraction(
    query_config_base: str,
    wave_dir: str,
    output_base: str,
    num_grids: int,
    sampling_rate: int = 16000,
    hubert_model_name: str = "facebook/hubert-base-ls960",
    *,
    # new knobs:
    make_pairs: bool = True,
    pairs_output_dir: str | Path = "data/speaker_pairs",
    pairs_val_ratio: float = 0.2,
    pairs_num_pairs: int = 10_000,
):
    """
    extract hubert embeddings per grid json, write per-grid csvs to output_base/grid_i,
    then (optionally) build siamese training split from the concatenated embeddings.
    """
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # --- model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = AutoFeatureExtractor.from_pretrained(hubert_model_name)
    model = AutoModel.from_pretrained(
        hubert_model_name,
        output_hidden_states=True,
        trust_remote_code=True,
        use_safetensors=True,
    )
    model.eval().to(device)

    filename_re = re.compile(r"(\d+)-(\d+)-(\d+)_chunk(\d+)\.npy")

    # --- per-grid extraction (unchanged core) ---
    for grid_idx in range(1, num_grids + 1):
        config_folder = Path(f"{query_config_base}{grid_idx}")
        out_dir = output_base / f"grid_{grid_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not config_folder.exists():
            print(f"[!] skipping missing config folder: {config_folder}")
            continue

        for json_file in config_folder.glob("*.json"):
            with open(json_file, "r") as f:
                config = json.load(f)

            rows = []
            for speaker in tqdm(config.get("speakers", []), desc=f"grid {grid_idx} - {json_file.name}"):
                speaker_id = speaker["speaker_id"]
                for rel_file in speaker.get("files", []):
                    full_path = os.path.join(wave_dir, rel_file)
                    try:
                        waveform = np.load(full_path)
                        if waveform.ndim > 1:
                            waveform = waveform[0]
                        inputs = extractor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        with torch.no_grad():
                            outputs = model(**inputs)
                            last_hidden = outputs.hidden_states[-1]
                            embedding = last_hidden.mean(dim=1).squeeze().cpu().numpy()

                        filename = os.path.basename(full_path)
                        match = filename_re.match(filename)
                        if match:
                            session_id, instance_id, chunk_id = match.group(2), match.group(3), match.group(4)
                        else:
                            session_id = instance_id = chunk_id = "unknown"

                        rows.append({
                            "speaker_id": speaker_id,
                            "session_id": session_id,
                            "instance_id": instance_id,
                            "chunk_id": chunk_id,
                            # store embedding as json-friendly list string
                            "embedding": json.dumps(embedding.tolist()),
                        })
                    except Exception as e:
                        print(f"⚠️ failed to process {full_path}: {e}")

            out_csv = out_dir / json_file.with_suffix(".csv").name
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"✅ saved {len(rows)} embeddings -> {out_csv}")

    # --- merge and build pairs (new) ---
    if make_pairs:
        merged_csv = output_base.parent / "hubert_embeddings.csv"
        merged_csv = _merge_per_grid_csvs_to_single_csv(output_base, merged_csv)
        create_speaker_split_datasets_from_embeddings_csv(
            csv_path=merged_csv,
            output_dir=Path(pairs_output_dir),
            val_ratio=pairs_val_ratio,
            num_pairs=pairs_num_pairs,
        )
