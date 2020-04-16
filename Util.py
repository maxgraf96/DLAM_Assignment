import librosa
import librosa.display
import matplotlib.pyplot as plt

from Hyperparameters import sample_rate, hop_size


def plot_final(data):
    """
    Take an amplitude spectrogram expressed in dB and plot it
    :param data:
    :return:
    """
    plt.figure(figsize=(14, 8))
    librosa.display.specshow(data, y_axis='log', x_axis='time', sr=sample_rate, hop_length=hop_size)
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def plot_final_mel(data):
    """
    Take a
    :param data:
    :return:
    """
    plt.figure(figsize=(14, 8))
    librosa.display.specshow(data, y_axis='mel', x_axis='time', sr=sample_rate, hop_length=hop_size)
    plt.colorbar(format='%+2.0f dB')
    plt.show()