import glob
import os
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import stats

from Hyperparameters import sep, gen_dir


def map_to_zero_one(spec, min_from, max_from):
    for i in range(spec.shape[0]):
        spec[i] = (1 - 0) * ((spec[i] - min_from) / (max_from - min_from))
    return spec


class AssignmentDataset(Dataset):
    """
    Dataset created from auto generated audio tracks from MIDI.
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory containing the spectrogram data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # Get precalculated means and stds
        self.root_dir = root_dir
        # self.mean = np.load(root_dir + sep + "mean.npy")
        # self.std = np.load(root_dir + sep + "std.npy")
        self.transform = transform
        piano_npys = Path(root_dir).rglob("*_piano.npy")
        self.filenames = [str(npy) for npy in piano_npys]
        # TODO add synths here as well
        self.length = len(self.filenames)

        # Load one spectrogram to get width and height
        spec = np.load(self.filenames[0])
        # self.spec_width = spec.shape[1]
        # self.spec_height = spec.shape[0]
        self.spec_dim = spec.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get spectrogram
        # Convert idx to filename
        path = self.filenames[idx]

        mag, phase = self.get_spectrogram(path)

        # Add "color" channel dimension for pytorch
        # spec = np.expand_dims(spec, axis=0)

        spec = np.stack((mag, phase))
        sample = {'sound': spec, 'filename': path}

        if self.transform:
            sample = self.transform(sample)

        return sample['sound']

    def get_spectrogram(self, path):
        if not os.path.exists(path):
            print("Error: Spectrogram for file '" + path + "' does not exist! Aborting...")
            return None

        spec = np.load(path)

        # Split into magnitude and phase
        # TODO: Use this to split magnitude and phase right
        # Or skip it and use griffin-lim
        # S, P = librosa.core.magphase(spec)
        mag = np.abs(spec)
        phase = np.angle(spec)

        # Standardise by precalculated mean and standard deviation of dataset
        # spec = (spec - self.mean) / self.std

        # Standardise using boxcox transformation and 0...1 mapping
        # spec = standardise(spec)

        return mag, phase

    def get_spec_dims(self):
        # return self.spec_width, self.spec_height
        return self.spec_dim

    # def get_mean_std(self):
    #     return self.mean, self.std


def standardise(spec):
    l2 = 10 ** -7
    # Get 0 and max value after transform
    min_idx = np.argmin(spec)
    max_idx = np.argmax(spec)
    spec = stats.boxcox(spec, lmbda=l2)
    zero_transformed = spec[min_idx]
    max_transformed = spec[max_idx]
    spec = map_to_zero_one(spec, zero_transformed, max_transformed)

    return spec


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        spec = sample['sound']
        sound = torch.from_numpy(spec).float()
        return {'sound': sound, 'filename': sample['filename']}
