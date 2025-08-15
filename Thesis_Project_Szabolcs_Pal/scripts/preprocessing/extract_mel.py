# === mel_extractor.py ===
import numpy as np
import librosa
import torch
def extract_mel_spectrogram(waveform_path, sr=16000, n_mels=40, hop_length=160, n_fft=400):
    """
    Load a waveform from .npy and compute the mel spectrogram.

    Args:
        waveform_path (str or Path): Path to the .npy file containing raw waveform.
        sr (int): Sample rate.
        n_mels (int): Number of mel bands.
        hop_length (int): Hop length for STFT.
        n_fft (int): FFT window size.

    Returns:
        np.ndarray: Mel spectrogram (n_mels x time)
    """
    waveform = np.load(waveform_path, allow_pickle=True)
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec


