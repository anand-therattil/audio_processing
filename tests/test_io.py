# tests/test_io.py

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from loader.io import load_audio

def test_load_audio():
    waveform, sr = load_audio(r"/Users/cmi_10128/Desktop/documents/projects/audio_processing/data/wav/LJ001-0012.wav")
    assert sr == 16000
    assert waveform.shape[0] == 1
    assert waveform.ndim == 2
