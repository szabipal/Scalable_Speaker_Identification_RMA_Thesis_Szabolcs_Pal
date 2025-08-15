# # low_melband_feature.py

# import parselmouth
# import numpy as np
# import librosa
# import librosa.display
# import scipy.stats

# def extract_low_melband_features(audio_path):
#     snd = parselmouth.Sound(audio_path)
#     features = {}

#     # --- Pitch Stats ---
#     pitch = snd.to_pitch()
#     pitch_values = pitch.selected_array['frequency']
#     voiced = pitch_values > 50
#     voiced_values = pitch_values[voiced]
    
#     features["f0_mean"] = np.mean(voiced_values) if len(voiced_values) else 0
#     features["f0_std"] = np.std(voiced_values) if len(voiced_values) else 0

#     # Pitch slope (via linear regression)
#     if len(voiced_values) > 2:
#         times = np.arange(len(voiced_values))
#         slope, _, _, _, _ = scipy.stats.linregress(times, voiced_values)
#         features["f0_slope"] = slope
#     else:
#         features["f0_slope"] = 0

#     # --- F1 (low formant) behavior ---
#     formant = snd.to_formant_burg()
#     f1s = [formant.get_value_at_time(1, t) or 0 for t in np.arange(0, snd.duration, 0.01)]
#     features["f1_mean"] = np.mean(f1s)
#     features["f1_var"] = np.var(f1s)

#     # --- Jitter (pitch instability) ---
#     try:
#         if np.max(np.abs(waveform)) == 0:
#             raise ValueError("Silent waveform")

#         waveform = waveform / np.max(np.abs(waveform))
#         waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32)

#         sound = parselmouth.Sound(values=waveform, sampling_frequency=sr)

#         # ✅ Use Parselmouth method that returns a Python object, not internal Praat object
#         point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)

#         num_points = parselmouth.praat.call(point_process, "Get number of points")
#         if num_points < 2:
#             raise ValueError("Too few glottal pulses")

#         # ✅ This now works because both are Parselmouth objects
#         # features["jitter_local"] = parselmouth.praat.call(
#         #     [sound, point_process], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
#         # )
#         features["jitter_local"] = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
#         features["jitter_ppq5"] = parselmouth.praat.call(
#              point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3
#         )
#         # features["shimmer_local"] = parselmouth.praat.call(
#         #     [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
#         # )
#         # features["shimmer_apq5"] = parselmouth.praat.call(
#         #     [sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6
#         # )
#     # --- Voiced frame ratio ---
#         features["voiced_ratio"] = np.sum(voiced) / len(pitch_values) if len(pitch_values) else 0

#     except Exception as e:
#         print(f" Jitter extraction failed: {e}")
#         features["jitter_local"] = 0
#         features["jitter_ppq5"] = 0
#         features["voiced_ratio"] = 0




# # --- Low-band energy (first 1/3 of mel bins = 0–~1000 Hz) ---
#     y, sr = librosa.load(audio_path, sr=None)
#     S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=128,
#                                     n_mels=80, fmin=0, fmax=8000)
#     low_band_indices = np.arange(0, 80 // 3)  # bins 0–26
#     low_band_energy = np.mean(S[low_band_indices, :])
#     features["low_energy"] = low_band_energy


#     return np.array(list(features.values()), dtype=np.float32)

import numpy as np
import parselmouth
import librosa
import scipy.stats
from parselmouth.praat import call

def extract_low_melband_features(waveform: np.ndarray, sr: int = 16000):
    features = {}

    try:
        # Normalize waveform
        if np.max(np.abs(waveform)) == 0:
            raise ValueError("Silent waveform")
        waveform = waveform / np.max(np.abs(waveform))
        waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32)

        # Parselmouth sound
        snd = parselmouth.Sound(values=waveform, sampling_frequency=sr)

        # --- Pitch Stats ---
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        voiced = pitch_values > 50
        voiced_values = pitch_values[voiced]

        features["f0_mean"] = np.mean(voiced_values) if len(voiced_values) else 0
        features["f0_std"] = np.std(voiced_values) if len(voiced_values) else 0

        if len(voiced_values) > 2:
            times = np.arange(len(voiced_values))
            slope, _, _, _, _ = scipy.stats.linregress(times, voiced_values)
            features["f0_slope"] = slope
        else:
            features["f0_slope"] = 0

        # --- F1 Formant behavior ---
        formant = snd.to_formant_burg()
        f1s = []
        for t in np.arange(0, snd.duration, 0.01):
            val = formant.get_value_at_time(1, t)
            if val is not None and not np.isnan(val):
                f1s.append(val)

        features["f1_mean"] = np.mean(f1s) if f1s else 0
        features["f1_var"] = np.var(f1s) if f1s else 0

        # --- Jitter ---
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
        num_points = call(point_process, "Get number of points")
        if num_points < 2:
            raise ValueError("Too few glottal pulses")

        features["jitter_local"] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features["jitter_ppq5"] = call( point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)

        # --- Voiced Ratio ---
        features["voiced_ratio"] = np.sum(voiced) / len(pitch_values) if len(pitch_values) else 0

    except Exception as e:
        print(f"[Low-band] Error in pitch/formant/jitter: {e}")
        features["f0_mean"] = 0
        features["f0_std"] = 0
        features["f0_slope"] = 0
        features["f1_mean"] = 0
        features["f1_var"] = 0
        features["jitter_local"] = 0
        features["jitter_ppq5"] = 0
        features["voiced_ratio"] = 0

    try:
        # --- Low-band Energy (first 1/3 of mel bins = 0–~1000 Hz) ---
        S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=512, hop_length=128,
                                           n_mels=80, fmin=0, fmax=8000)
        low_band_indices = np.arange(0, 80 // 3)  # bins 0–26
        low_band_energy = np.mean(S[low_band_indices, :])
        features["low_energy"] = low_band_energy

    except Exception as e:
        print(f"[Low-band] Error in mel feature: {e}")
        features["low_energy"] = 0

    return np.array(list(features.values()), dtype=np.float32)
