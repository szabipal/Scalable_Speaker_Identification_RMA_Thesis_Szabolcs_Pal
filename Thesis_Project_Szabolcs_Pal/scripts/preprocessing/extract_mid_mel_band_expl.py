# mid_melband_feature.py

import parselmouth
import numpy as np
import librosa
import scipy.stats

def extract_mid_melband_features(waveform: np.ndarray, sr: int = 16000):
    # snd = parselmouth.Sound(audio_path)
    features = {}

    try:
        # Normalize waveform
        if np.max(np.abs(waveform)) == 0:
            raise ValueError("Silent waveform")
        waveform = waveform / np.max(np.abs(waveform))
        waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32)
        snd = parselmouth.Sound(values=waveform, sampling_frequency=sr)
        # --- F2 Slope and F2–F3 Distance ---
        formant = snd.to_formant_burg()
        f2s, f3s = [], []

        for t in np.arange(0, snd.duration, 0.01):
            f2 = formant.get_value_at_time(2, t)
            f3 = formant.get_value_at_time(3, t)
            if f2 is not None and not np.isnan(f2):
                f2s.append(f2)
            if f3 is not None and not np.isnan(f3):
                f3s.append(f3)

        f2s = np.array(f2s)
        f3s = np.array(f3s)
        times = np.arange(len(f2s))

        if len(f2s) > 2:
            slope, _, _, _, _ = scipy.stats.linregress(times, f2s)
            features["f2_slope"] = slope
            features["f2_var"] = np.var(f2s)
        else:
            features["f2_slope"] = 0
            features["f2_var"] = 0

        if len(f3s) > 2:
            features["f3_var"] = np.var(f3s)
        else:
            features["f3_var"] = 0

        if len(f3s) == len(f2s) and len(f2s) > 0:
            diff = f3s - f2s
            features["f2f3_dist"] = np.mean(diff[diff > 0]) if np.any(diff > 0) else 0
        else:
            features["f2f3_dist"] = 0


            # --- Load audio for mid-band spectral analysis ---
            # y, sr = librosa.load(audio_path, sr=None)
        S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=512, hop_length=128,
                                        n_mels=80, fmin=0, fmax=8000)
        mid_band_indices = np.arange(80 // 3, 2 * 80 // 3)  # 27–53 if n_mels=80
        mid_band_spec = S[mid_band_indices, :]

        # --- Spectral Flatness (Mid Band) ---
        flatness = librosa.feature.spectral_flatness(S=mid_band_spec)
        features["mid_flatness"] = np.mean(flatness)

        # --- Spectral Centroid (Mid Band) ---
        centroid = librosa.feature.spectral_centroid(S=mid_band_spec, sr=sr)
        features["mid_centroid"] = np.mean(centroid)

            # # --- Zero-Crossing Rate (Whole Signal) ---
            # zcr = librosa.feature.zero_crossing_rate(y)
            # features["zcr"] = np.mean(zcr)

    except Exception as e:
        print(f"[mid-band] Error: {e}")

    return np.array(list(features.values()), dtype=np.float32)

