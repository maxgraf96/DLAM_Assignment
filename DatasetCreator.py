import os
from pathlib import Path

import librosa
import numpy as np

from Util import map_to_zero_one
from Hyperparameters import sep, gen_dir, n_fft, hop_size, sample_rate, spec_width, limit_s, n_mels, top_db


def create_spectrogram(path):
    """
    Create a Mel spectrogram from a specified *.wav file, which is trimmed to "limit_s" seconds before conversion.
    :param path: The path to the *.wav file
    :return: The Mel spectrogram, mapped to a range of [0...1]
    """
    # Load signal data, sample rate should always be 22050
    sig, sr = librosa.load(path)

    # Limit to "limit_s" seconds to guarantee the same length for all data points
    sig = librosa.util.fix_length(sig, limit_s * sample_rate)

    # Calculate the Mel spectrogram
    mel = librosa.feature.melspectrogram(sig, sr=sample_rate, n_fft=n_fft, hop_length=hop_size, n_mels=n_mels)
    # Convert to dB
    mel = librosa.power_to_db(mel, ref=np.max, top_db=top_db)
    # Map to range [0...1]
    min_db = np.min(mel)
    max_db = np.max(mel)
    mel = map_to_zero_one(mel, min_db, max_db)

    return mel

def create_datapoints(wav_path, mode, number_of_wavs, counter):
    """
    This function is called by the "initialise_dataset()" method below.
    Helper function to create and store a Mel spectrogram in a *.npy file, using a given *.wav file
    :param wav_path: The path to the *.wav file
    :param mode: The mode of the file, currently either "piano" or "synth", used to specify the output folder
    :param number_of_wavs: The total number of *.wav files that are generated
    :param counter: Counter indicating the overall progress
    :return: None
    """
    wav_str = str(wav_path)
    # Extract file name
    filename = wav_str.split(sep)[-1][:-4]
    folder = gen_dir + sep + filename + sep + mode
    # Check if there is already an existing spectrogram
    if os.path.exists(folder):
        print("Spectrogram for " + filename + " already exists. Skipping...")
        return

    # Print progress information
    print('Generating ' + mode + ' spectrogram for file ' + filename + ': ' + str(counter) + ' / ' + str(number_of_wavs))

    # Create a separate folder for each *.wav file
    Path(folder).mkdir(parents=True, exist_ok=True)

    # Create Mel spectrogram
    mel = create_spectrogram(wav_str)

    # Create data points for the CNN
    # Currently one data point corresponds to the whole clip (30s long)
    # In the future this could be altered to accomodate longer clips
    width = spec_width
    frame_counter = 0
    for frame in range(0, mel.shape[1], width):
        if frame + width > mel.shape[1]:
            break
        current_mel = mel[:, frame : frame + width]
        # Save to a *.npy file for access by the Dataset subclasses
        np.save(folder + sep + filename + "_" + str(frame_counter) + "_" + mode + "_mel", current_mel)
        frame_counter += 1

def initialise_dataset():
    """
    This method assumes that all the *.wav files are already generated from MIDI and present in the data/* folders.
    It's purpose is to create and store all the necessary spectrogram data points
    :return: None
    """
    # Create the necessary folder if it doesn't exist yet
    if not os.path.exists(gen_dir):
        Path(gen_dir).mkdir(parents=True, exist_ok=True)

    # Create spectrograms
    piano_dir = 'data' + sep + 'piano'
    synth_dir = 'data' + sep + 'synth'
    counter = 0

    number_of_wavs = len([name for name in os.listdir(piano_dir)])
    piano_wavs = Path(piano_dir).rglob("*.wav")
    for wav in piano_wavs:
        counter += 1
        create_datapoints(wav, 'piano', number_of_wavs, counter)

    # Synth
    counter = 0
    synth_wavs = Path(synth_dir).rglob("*.wav")
    for wav in synth_wavs:
        counter += 1
        create_datapoints(wav, 'synth', number_of_wavs, counter)