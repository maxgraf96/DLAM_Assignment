import os
from pathlib import Path

import librosa
import numpy as np

from Hyperparameters import sep, gen_dir, n_fft, hop_size, sample_rate

def initialise_dataset():
    """
    This method assumes that all the *.wav files are already generated from MIDI and present in the data/* folders.
    It's purpose is to create and store all the necessary spectrogram data
    :return:
    """

    # Create the necessary folder if it doesn't exist yet
    if not os.path.exists(gen_dir):
        Path(gen_dir).mkdir(parents=True, exist_ok=True)

    # Create spectrograms for piano if they don't exist yet
    piano_dir = 'data' + sep + 'piano'
    counter = 0

    # Mean and standard deviation
    means = []
    stds = []

    number_of_wavs = len([name for name in os.listdir(piano_dir)])
    piano_wavs = Path(piano_dir).rglob("*.wav")
    for wav in piano_wavs:
        counter += 1
        wav_str = str(wav)
        # Get file name
        filename = wav_str.split(sep)[-1][:-4]
        folder = gen_dir + sep + filename
        # Check if there is already an existing spectrogram
        if os.path.exists(folder):
            # print("Spectrogram for " + filename + " already exists. Skipping...")
            continue

        # Print progress information
        print('Generating spectrogram for file ' + filename + ': ' + str(counter) + ' / ' + str(number_of_wavs))

        # Create folder
        Path(folder).mkdir(parents=True, exist_ok=True)

        # Load signal data, sample rate should always be 22050
        sig, sr = librosa.load(wav_str)

        # Limit to 30s for now
        # TODO change this later
        sig = librosa.util.fix_length(sig, 30 * sample_rate)

        # Calculate STFT
        spec = librosa.stft(sig, n_fft, hop_length=hop_size, window='hann')

        # Create square data points for the CNN
        width = int(n_fft / 2) + 1
        frame_counter = 0
        for frame in range(0, spec.shape[1], width):
            if frame + width > spec.shape[1]:
                break
            current = spec[:, frame : frame + width]
            np.save(folder + sep + filename + "_" + str(frame_counter) + "_piano", current)
            frame_counter += 1

            # Get mean and std and append to list of all
            # means.append(np.mean(current))
            # stds.append(np.std(current))

    # Save mean and std
    # if not os.path.exists(gen_dir + sep + "mean.npy"):
    #     np.save(gen_dir + sep + "mean", np.mean(means))
    # if not os.path.exists(gen_dir + sep + "std.npy"):
    #     np.save(gen_dir + sep + "std", np.mean(stds))