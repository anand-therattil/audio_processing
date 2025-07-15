import torchaudio.transforms as T
import matplotlib.pyplot as plt

def mel_spectrogram_transform(audio_array=None, sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64):
    """
    Create a MelSpectrogram transform with the specified sample rate.
    
    Args:
        sample_rate (int): The sample rate of the audio signal.
        
    Returns:
        T.MelSpectrogram: A MelSpectrogram transform configured with the given sample rate.
    """
    if audio_array is None:
        raise ValueError("audio_array must be provided")
    
    transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    return transform(audio_array) 

def plot_spectrogram(spec, title, ylabel, aspect='auto', xmax=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Time')
    im = ax.imshow(spec, origin='lower', aspect=aspect)
    if xmax:
        ax.set_xlim((0, xmax))
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


