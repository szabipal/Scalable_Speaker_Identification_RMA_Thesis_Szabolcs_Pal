# mfcc_complement_feature.py

import parselmouth
import numpy as np
import librosa

def extract_mfcc_complement_features(waveform: np.ndarray, sr: int = 16000):
    snd = parselmouth.Sound(values=waveform, sampling_frequency=sr)
    features = {}

    # 1. F0 (Pitch)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 50]
    features["f0_mean"] = np.mean(pitch_values) if len(pitch_values) else 0
    features["f0_std"] = np.std(pitch_values) if len(pitch_values) else 0

    # 2. Formants
    formant = snd.to_formant_burg()
    f1s, f2s, f3s = [], [], []
    for t in np.arange(0, snd.duration, 0.01):
        f1 = formant.get_value_at_time(1, t)
        f2 = formant.get_value_at_time(2, t)
        f3 = formant.get_value_at_time(3, t)
        f1s.append(f1 if f1 is not None and not np.isnan(f1) else np.nan)
        f2s.append(f2 if f2 is not None and not np.isnan(f2) else np.nan)
        f3s.append(f3 if f3 is not None and not np.isnan(f3) else np.nan)

    features["f1_mean"] = np.nanmean(f1s) if len(f1s) > 0 else 0
    features["f2_mean"] = np.nanmean(f2s) if len(f2s) > 0 else 0
    features["f3_mean"] = np.nanmean(f3s) if len(f3s) > 0 else 0

    try:
        if np.max(np.abs(waveform)) == 0:
            raise ValueError("Silent waveform")

        waveform = waveform / np.max(np.abs(waveform))
        waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32)

        sound = parselmouth.Sound(values=waveform, sampling_frequency=sr)

        # ✅ Use Parselmouth method that returns a Python object, not internal Praat object
        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)

        num_points = parselmouth.praat.call(point_process, "Get number of points")
        if num_points < 2:
            raise ValueError("Too few glottal pulses")

        # ✅ This now works because both are Parselmouth objects
        # features["jitter_local"] = parselmouth.praat.call(
        #     [sound, point_process], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        # )
        features["jitter_local"] = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features["jitter_ppq5"] = parselmouth.praat.call(
             point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3
        )
        features["shimmer_local"] = parselmouth.praat.call(
            [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        features["shimmer_apq5"] = parselmouth.praat.call(
            [sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )

    except Exception as e:
        print(f"⚠️ Jitter/Shimmer extraction failed: {e}")
        features["jitter_local"] = 0
        features["jitter_ppq5"] = 0
        features["shimmer_local"] = 0
        features["shimmer_apq5"] = 0


    # 4. Harmonics-to-Noise Ratio (HNR)
    harmonicity = snd.to_harmonicity_cc()
    hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
    features["hnr"] = hnr if not np.isnan(hnr) else 0

    # 5. Energy
    y = waveform
    energy = np.mean(librosa.feature.rms(y=y))
    features["energy"] = energy

    return np.array(list(features.values()), dtype=np.float32)
