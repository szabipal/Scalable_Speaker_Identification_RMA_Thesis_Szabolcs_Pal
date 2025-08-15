import torch
import pandas as pd
import numpy as np
import ast
from torch.utils.data import Dataset

# my_datasets/siamese_separated_dataset.py
import pandas as pd, numpy as np, json, os
from pathlib import Path

class SiamesePairCSV:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # accept common alias pairs
        alias_pairs = [
            ("embedding_1", "embedding_2"),
            ("path_a", "path_b"),
            ("emb1", "emb2"),
            ("embedding_a", "embedding_b"),
            ("x1_path", "x2_path"), 
        ]
        for a, b in alias_pairs:
            if a in df.columns and b in df.columns:
                c1, c2 = a, b
                break
        else:
            raise KeyError(f"expected one of {alias_pairs} in CSV; got {df.columns.tolist()}")

        if "label" not in df.columns:
            raise KeyError("CSV must contain a 'label' column (0/1 for different/same)")

        # convert cells to numpy vectors
        def _to_vec(x):
            # numpy file path
            if isinstance(x, str) and x.endswith(".npy") and os.path.exists(x):
                return np.load(x)
            # json list in string
            if isinstance(x, str):
                try:
                    return np.array(json.loads(x))
                except Exception:
                    pass
            # already array-like
            if isinstance(x, (list, tuple, np.ndarray)):
                return np.array(x)
            # fallback: raise so we see bad rows early
            raise TypeError(f"cannot parse embedding cell: {x!r}")

        self.pairs = []
        for a, b, y in zip(df[c1], df[c2], df["label"]):
            v1 = _to_vec(a)
            v2 = _to_vec(b)
            self.pairs.append((v1, v2, int(y)))
