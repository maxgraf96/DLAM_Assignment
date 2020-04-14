import os
from pathlib import Path

import librosa
import numpy as np

from Dataset import map_to_zero_one
from Hyperparameters import sep, gen_dir, n_fft, hop_size, sample_rate, spec_width, limit_s


def create_spectrogram(path):
    # Load signal data, sample rate should always be 22050
    sig, sr = librosa.load(path)

    # Limit to 30s for now
    # TODO change this later
    sig = librosa.util.fix_length(sig, limit_s * sample_rate)

    # Calculate STFT
    spec = librosa.stft(sig, n_fft, hop_length=hop_size, window='hann')
    # spec = librosa.feature.melspectrogram(sig, sr=sample_rate, n_fft=n_fft, hop_length=hop_size, n_mels=n_mels)

    spec = np.abs(spec)
    # Convert power to dB
    spec = librosa.amplitude_to_db(spec, ref=np.max, top_db=120)
    min_db = np.min(spec)
    max_db = np.max(spec)
    spec = map_to_zero_one(spec, min_db, max_db)

    return spec

def create_datapoints(wav_path, mode, number_of_wavs, counter):
    wav_str = str(wav_path)
    # Get file name
    filename = wav_str.split(sep)[-1][:-4]
    folder = gen_dir + sep + filename + sep + mode
    # Check if there is already an existing spectrogram
    if os.path.exists(folder):
        # print("Spectrogram for " + filename + " already exists. Skipping...")
        return

    # Print progress information
    print('Generating ' + mode + ' spectrogram for file ' + filename + ': ' + str(counter) + ' / ' + str(number_of_wavs))

    # Create folder
    Path(folder).mkdir(parents=True, exist_ok=True)

    spec = create_spectrogram(wav_str)

    # Create data points for the CNN
    width = spec_width
    frame_counter = 0
    for frame in range(0, spec.shape[1], width):
        if frame + width > spec.shape[1]:
            break
        current = spec[:, frame: frame + width]
        np.save(folder + sep + filename + "_" + str(frame_counter) + "_" + mode, current)
        frame_counter += 1

def initialise_dataset():
    """
    This method assumes that all the *.wav files are already generated from MIDI and present in the data/* folders.
    It's purpose is to create and store all the necessary spectrogram data
    :return:
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
        # if 'chpn_op7_1' not in str(wav):
        #     continue
        counter += 1
        create_datapoints(wav, 'piano', number_of_wavs, counter)

    # Synth
    counter = 0
    synth_wavs = Path(synth_dir).rglob("*.wav")
    for wav in synth_wavs:
        # if 'chpn_op7_1' not in str(wav):
        #     continue
        counter += 1
        create_datapoints(wav, 'synth', number_of_wavs, counter)