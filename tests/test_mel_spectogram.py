import pytest
import torch
import sys
import os
import matplotlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from loader.io import load_audio
from features.mel_spectogram import mel_spectrogram_transform, plot_spectrogram

matplotlib.use('Agg')  # Use non-GUI backend for matplotlib in tests

@pytest.fixture
def dummy_waveform():
    waveform, sample_rate = load_audio(r"/Users/cmi_10128/Desktop/documents/projects/audio_processing/data/wav/LJ001-0012.wav")
    return waveform, sample_rate    

def test_mel_spectrogram_transform_output_shape(dummy_waveform):
    waveform, sample_rate = dummy_waveform
     
    mel_spec = mel_spectrogram_transform(waveform, sample_rate=sample_rate)
    
    # Check shape: (channels, n_mels, time_frames)
    assert mel_spec.ndim == 3
    assert mel_spec.shape[0] == 1  # One channel
    assert mel_spec.shape[1] == 64  # Default n_mels

def test_plot_spectrogram_runs_without_error(dummy_waveform):
    waveform, sample_rate = dummy_waveform
    
    mel_spec = mel_spectrogram_transform(waveform, sample_rate=sample_rate)
    log_mel_spec = mel_spec.clamp(min=1e-9).log2()
    
    # Should run without error and produce a plot object
    try:
        plot_spectrogram(log_mel_spec[0].numpy(), title="Mel Spectrogram Test", ylabel="Mel bins")
    except Exception as e:
        pytest.fail(f"plot_spectrogram raised an exception: {e}")
