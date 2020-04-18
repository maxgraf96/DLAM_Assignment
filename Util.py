import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from Hyperparameters import sample_rate, hop_size

def map_to_zero_one(spec, min_from, max_from):
    return map_to_range(spec, min_from, max_from, 0, 1)

def map_to_range(spec, input_start, input_end, output_start, output_end):
    copy = np.copy(spec)
    slope = 1.0 * (output_end - output_start) / (input_end - input_start)
    copy = output_start + slope * (copy - input_start)
    return copy

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