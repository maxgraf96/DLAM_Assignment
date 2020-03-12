import os
from pathlib import Path
import numpy as np
import FreesoundDataset
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_mel_spectrogram(path):
    spec = FreesoundDataset.get_spectrogram(path)

    librosa.display.specshow(spec, sr=FreesoundDataset.GLOBAL_SR, hop_length=FreesoundDataset.HOP_SIZE,
                             x_axis='time', y_axis='log', fmax=FreesoundDataset.GLOBAL_SR / 2)
    plt.title("Spectrogram for " + str(path))
    plt.tight_layout()
    plt.show()

wavs = Path("data/Arturia MicroBrute Saw Oscillator").rglob("*.wav")
for wav in wavs:
    plot_mel_spectrogram(wav)