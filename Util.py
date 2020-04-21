import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from Hyperparameters import sample_rate, hop_size

mpl.rcParams['figure.dpi'] = 300 # Set high dpi for high-quality plots

def get_spectrogram(path):
    """
    Helper function to load a spectrogram from a *.npy file
    :param path: The path to the *.npy file
    :return: The spectrogram in numpy's standard ndarray format
    """
    if not os.path.exists(path):
        print("Error: Spectrogram for file '" + path + "' does not exist! Aborting...")
        return None

    spec = np.load(path)
    return spec


def map_to_zero_one(spec, min_from, max_from):
    """
    Helper function to map a given numpy ndarray into a given range
    :param spec: The input array
    :param min_from: The input's minimum value
    :param max_from: The input's maximum value
    :return: A copy of the input, mapped to [0...1]
    """
    return map_to_range(spec, min_from, max_from, 0, 1)

def map_to_range(spec, input_start, input_end, output_start, output_end):
    """
    Helper function to map a given ndarray to a given range
    :param spec: The input ndarray
    :param input_start: The input's minimum value
    :param input_end: The input's maximum value
    :param output_start: The desired minimum value
    :param output_end: The desired maximum value
    :return: A copy of the input, mapped to the desired range
    """
    # Copy the input
    copy = np.copy(spec)
    # Calculate the slope
    slope = 1.0 * (output_end - output_start) / (input_end - input_start)
    # Map the input
    copy = output_start + slope * (copy - input_start)
    # Return the copy of the input
    return copy

def plot_mel(data):
    """
    Take an amplitude spectrogram expressed in dB and plot it
    :param data:
    :return:
    """
    plt.figure(figsize=(14, 8))
    librosa.display.specshow(data, y_axis='mel', x_axis='time', sr=sample_rate, hop_length=hop_size)
    plt.colorbar(format='%+2.0f dB')
    plt.show()