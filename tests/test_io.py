# tests/test_io.py

import pytest
from src.loader.io import load_audio

def test_load_audio():
    waveform, sr = load_audio(r"C:\Users\Anand\Desktop\repository\audio_processing\data\wav\LJ001-0012.wav")
    assert sr == 16000
    assert waveform.shape[0] == 1
    assert waveform.ndim == 2
