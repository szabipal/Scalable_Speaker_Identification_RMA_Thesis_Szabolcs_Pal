# scripts/ensemble_operater/pipeline_config.py

from pathlib import Path

# repo root if you need it elsewhere
ROOT = Path(__file__).resolve().parents[2]

all_configs = {
    "spectral": {
        "feature_types": ["mfcc", "mel_band_low", "mel_band_high", "mel_band_mid"],
        "quantile_dir": "spectral_cosine_model_quantiles",
        "embedding_root": "embeddings_spectral_final_test",
        "train_embedding_root": "embeddings_spectral_final_train",
        "dev_embedding_root": "embeddings_spectral_final_dev",
        "output_root": "output_spectral_final_test",
        "train_output_root": "output_spectral_final_train",
    },
    "phonetic": {
        "feature_types": ["mfcc", "mid", "low", "high"],
        "quantile_dir": "phonetic_cosine_model_quantiles",
        "embedding_root": "embeddings_phonetic_final_test",
        "train_embedding_root": "embeddings_phonetic_final_train",
        "dev_embedding_root": "embeddings_phonetic_final_dev",
        "output_root": "output_phonetic_final_test",
        "train_output_root": "output_phonetic_final_train",
    },
}

test_cases  = [1, 2, 3]
train_cases = [1, 2, 3, 4, 5]

grid_map = {
     1: "10s_10i",  2: "20s_10i",  3: "2s_10i",   4: "40s_10i",  5: "5s_10i",  6: "60s_10i",  7: "80s_10i",
     8: "10s_1i",   9: "20s_1i",  10: "2s_1i",  11: "40s_1i", 12: "5s_1i", 13: "60s_1i", 14: "80s_1i",
    15: "10s_20i", 16: "20s_20i", 17: "2s_20i", 18: "40s_20i", 19: "5s_20i", 20: "60s_20i", 21: "80s_20i",
    22: "10s_40i", 23: "20s_40i", 24: "2s_40i", 25: "40s_40i", 26: "5s_40i", 27: "60s_40i", 28: "80s_40i",
    29: "10s_5i",  30: "20s_5i",  31: "2s_5i",  32: "40s_5i",  33: "5s_5i",  34: "60s_5i",  35: "80s_5i",
    36: "10s_60i", 37: "20s_60i", 38: "2s_60i", 39: "40s_60i", 40: "5s_60i", 41: "60s_60i", 42: "80s_60i",
    43: "10s_80i", 44: "20s_80i", 45: "2s_80i", 46: "40s_80i", 47: "5s_80i", 48: "60s_80i", 49: "80s_80i",
}

FEATURE_FOLDER_MAP = {
    "mfcc": "mfcc",
    "mel_band_low": "low",
    "mel_band_mid": "mid",
    "mel_band_high": "high",
    # phonetic feature names already match folder names
}

__all__ = [
    "ROOT", "all_configs", "test_cases", "train_cases",
    "grid_map", "FEATURE_FOLDER_MAP",
]
