import numpy as np
import librosa
from scipy.stats import kurtosis, skew

def extract_high_melband_features(y, sr=16000):
    features = {}

    # --- Audio loading for spectral analysis ---
    # y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=128,
                                       n_mels=80, fmin=0, fmax=8000)

    # Align with model's upper third (bins 54–79 if n_mels=80)
    high_band_indices = np.arange(2 * 80 // 3, 80)
    high_band_spec = S[high_band_indices, :]

    # --- Spectral Flatness (fricative cue) ---
    flatness = librosa.feature.spectral_flatness(S=high_band_spec)
    features["high_flatness"] = np.mean(flatness)

    # --- Spectral Centroid (fricative cue) ---
    centroid = librosa.feature.spectral_centroid(S=high_band_spec, sr=sr)
    features["high_centroid"] = np.mean(centroid)

    # --- Spectral Roll-off (95%) (fricative cue) ---
    rolloff = librosa.feature.spectral_rolloff(S=high_band_spec, sr=sr, roll_percent=0.95)
    features["high_rolloff"] = np.mean(rolloff)

    # --- Spectral Bandwidth (fricative cue) ---
    bandwidth = librosa.feature.spectral_bandwidth(S=high_band_spec, sr=sr)
    features["high_bandwidth"] = np.mean(bandwidth)

    # # --- Spectral Contrast (fricative cue) ---
    # contrast = librosa.feature.spectral_contrast(S=high_band_spec, sr=sr)
    # features["high_contrast"] = np.mean(contrast)

    # --- High Band Energy ---
    high_band_energy = np.mean(high_band_spec)
    features["high_energy"] = high_band_energy

    # --- High-Band Energy Std Dev (temporal variation) ---
    frame_energy = np.sum(high_band_spec, axis=0)
    features["high_energy_std"] = np.std(frame_energy)

    # --- High-Band Delta Energy (frame-to-frame change) ---
    delta_energy = librosa.feature.delta(np.mean(high_band_spec, axis=0))
    features["high_energy_delta"] = np.mean(np.abs(delta_energy))

    # --- Spectral Shape: Skewness and Kurtosis ---
    frame_kurtosis = kurtosis(high_band_spec, axis=0)
    frame_skewness = skew(high_band_spec, axis=0)
    features["high_kurtosis"] = np.mean(frame_kurtosis)
    features["high_skewness"] = np.mean(frame_skewness)

    # --- Onset Strength in High Band (fricative transitions) ---
    onset_env = librosa.onset.onset_strength(S=high_band_spec, sr=sr)
    features["high_onset_strength"] = np.mean(onset_env)

    # # --- Zero Crossing Rate (whole signal) ---
    # zcr = librosa.feature.zero_crossing_rate(y)
    # features["zcr"] = np.mean(zcr)

    # --- Derived Ratio: Centroid / Bandwidth (sharpness) ---
    features["sharpness_ratio"] = features["high_centroid"] / (features["high_bandwidth"] + 1e-5)

    return np.array(list(features.values()), dtype=np.float32)
