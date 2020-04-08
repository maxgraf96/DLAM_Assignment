import os

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Desired signal length in seconds
SIG_LENGTH = 0.5
# Sample rate
GLOBAL_SR = 44100
# Hop size for STFT
HOP_SIZE = 512
SPEC_DIMS_W = 257
SPEC_DIMS_H = 259
SPEC_DIMS = SPEC_DIMS_W * SPEC_DIMS_H

class AssignmentDataset(Dataset):
    """
    Dataset created from auto generated audio tracks from MIDI.
    """
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.freesound_frame = pd.read_csv(csv_file)
        # Get precalculated means and stds
        self.mean_std_df = pd.read_csv(csv_file[:-4] + "_mean_std.csv")
        self.mean = self.mean_std_df.iloc[0, 0]
        self.std = self.mean_std_df.iloc[0, 1]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.freesound_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get spectrogram
        file_name = os.path.join(self.root_dir, self.freesound_frame.iloc[idx, 0])
        spec = get_spectrogram(file_name, self.mean, self.std)
        # Add "color" channel dimension for pytorch
        spec = np.expand_dims(spec, axis=0)

        # Get descriptors: Currently this is the Class Number
        descriptors = self.freesound_frame.iloc[idx, 2]
        descriptors = np.array([descriptors])
        sample = {'sound': spec, 'descriptors': descriptors}

        if self.transform:
            sample = self.transform(sample)

        return sample['sound'], sample['descriptors']

def get_spectrogram(path, mean, std):
    # Load using file's default sample rate
    sig, sr = librosa.load(path, sr=None, mono=True)
    # Trim silence at beginning and end
    sig = librosa.effects.trim(sig)[0]
    # Fix to specified length
    sig = librosa.util.fix_length(sig, SIG_LENGTH * GLOBAL_SR)

    # Calculate Spectrogram
    spec = np.abs(librosa.core.stft(sig, n_fft=512, hop_length=HOP_SIZE))

    # Normalise by precalculated mean and standard deviation of dataset
    spec = (spec - mean) / std

    return spec

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        spec, descriptors = sample['sound'], sample['descriptors']
        sound = torch.from_numpy(spec).float()
        # TODO implement number mapping for descriptors
        return {'sound': sound, 'descriptors': torch.from_numpy(descriptors)}