import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from Hyperparameters import sep, gen_dir

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
        self.mean = np.load(root_dir + sep + "mean.npy")
        self.std = np.load(root_dir + sep + "std.npy")
        self.transform = transform
        # - 2 because mean and std are also stored in that folder
        self.filenames = [f for f in glob.glob(root_dir + sep + "*_piano.npy")]
        # TODO add synths here as well
        self.length = len(self.filenames)

        # Load one spectrogram to get width and height
        spec = np.load(self.filenames[0])
        self.spec_width = spec.shape[1]
        self.spec_height = spec.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get spectrogram
        # Convert idx to filename
        path = self.filenames[idx]

        spec = self.get_spectrogram(path)

        # Add "color" channel dimension for pytorch
        spec = np.expand_dims(spec, axis=0)

        sample = {'sound': spec, 'filename': path}

        if self.transform:
            sample = self.transform(sample)

        return sample['sound']

    def get_spectrogram(self, path):
        if not os.path.exists(path):
            print("Error: Spectrogram for file '" + path + "' does not exist! Aborting...")
            return None

        spec = np.load(path)

        # Normalise by precalculated mean and standard deviation of dataset
        spec = (spec - self.mean) / self.std

        return spec

    def get_spec_dims(self):
        return self.spec_width, self.spec_height

    def get_mean_std(self):
        return self.mean, self.std

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        spec = sample['sound']
        sound = torch.from_numpy(spec).float()
        return {'sound': sound, 'filename': sample['filename']}