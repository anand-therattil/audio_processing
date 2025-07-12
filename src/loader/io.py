# src/utils/io.py

import torchaudio
import torchaudio.transforms as T
import os

from .config import SAMPLE_RATE


def load_audio(path: str, sample_rate: int = SAMPLE_RATE, normalize: bool = True):
    """
    Loads an audio file and resamples it to the target sample rate.

    Args:
        path (str): Path to audio file.
        sample_rate (int): Target sampling rate.
        normalize (bool): Whether to normalize audio amplitude to [-1, 1].

    Returns:
        waveform (Tensor): Shape [1, samples]
        sr (int): Actual sampling rate of the waveform
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sr = torchaudio.load(path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
        sr = sample_rate

    # Normalize
    if normalize:
        waveform = waveform / waveform.abs().max()

    return waveform, sr
