import os
from pathlib import Path
import numpy as np
import FreesoundDataset
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_mel_spectrogram(path):
    # Load using file's default sample rate
    sig, sr = librosa.load(path, sr=None, mono=True)
    # Normalize signal
    max_val = np.max((np.abs(np.min(sig)), np.max(sig)))
    sig = np.divide(sig, max_val)
    # Trim silence at beginning and end
    sig = librosa.effects.trim(sig)[0]
    # Fix to specified length
    sig = librosa.util.fix_length(sig, FreesoundDataset.SIG_LENGTH * FreesoundDataset.GLOBAL_SR)
    # Calculate Mel spectrogram
    # Use 87 mel bins to make the resulting spec quadratic 87x87
    mel_spec = librosa.feature.melspectrogram(sig, FreesoundDataset.GLOBAL_SR, n_mels=87)

    # Normalise mel spectrogram
    mellog = np.log(mel_spec + 1e-9)
    mel_spec = librosa.util.normalize(mellog)

    # Add "color" channel dimension
    # mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    plt.figure(figsize=(5, 4))
    librosa.display.specshow(mel_spec, sr=FreesoundDataset.GLOBAL_SR, x_axis='time', y_axis='mel', fmax=FreesoundDataset.GLOBAL_SR / 2)
    plt.show()

wavs = Path("data/test1").rglob("*.wav")
for wav in wavs:
    plot_mel_spectrogram(wav)