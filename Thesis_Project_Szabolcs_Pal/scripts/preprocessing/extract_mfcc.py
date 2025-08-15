# === extract_mfcc.py ===
import numpy as np
import torchaudio
import torchaudio.transforms as T
import torch

def extract_mfcc_from_npy(waveform_path, sr=16000, n_mfcc=13, win_length=400, hop_length=160):
    """
    Extract MFCCs from a .npy waveform using torchaudio.

    Args:
        waveform_path (str or Path): Path to the .npy file containing raw waveform.
        sr (int): Sample rate.
        n_mfcc (int): Number of MFCC coefficients to extract.
        win_length (int): Window length for FFT.
        hop_length (int): Hop length for STFT.

    Returns:
        np.ndarray: MFCC features [time, n_mfcc]
    """
    y = np.load(waveform_path, allow_pickle=True)  # shape: (samples,)
    waveform = torch.tensor(y).unsqueeze(0)  # shape: (1, samples)

    mfcc_transform = T.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={
            'n_fft': 512,
            'win_length': win_length,
            'hop_length': hop_length,
            'n_mels': 40,
            'center': True,
            'power': 2.0
        }
    )

    mfcc = mfcc_transform(waveform)  # shape: (1, n_mfcc, time)
    return mfcc.squeeze(0).transpose(0, 1).numpy()  # shape: (time, n_mfcc)

