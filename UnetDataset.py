import os
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.utils.data import Dataset

from Hyperparameters import sep, spec_width


class UnetDataset(Dataset):
    def __init__(self, root_dir, gt_dir, transform=None):
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
        # Input for this dataset is output from autoencoder
        input_mel_npys = Path(root_dir).rglob("*_output.npy")
        gt_mel_npys = Path(gt_dir).rglob("*_synth_mel.npy")
        self.input_mel_filenames = [str(npy) for npy in input_mel_npys]
        self.gt_mel_filenames = [str(npy) for npy in gt_mel_npys]

        # Create mapping between input and ground truth names (so that the order is correct)
        self.input_to_gt = {}
        len_suffix = len("_output.npy")
        for input_path in self.input_mel_filenames:
            input_filename = input_path.split(sep)[-1][:-len_suffix]
            for gt_path in self.gt_mel_filenames:
                if input_filename in gt_path:
                    self.input_to_gt[input_path] = gt_path

        self.length = len(self.input_mel_filenames)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get spectrogram
        # Convert idx to filename
        input_mel_path = self.input_mel_filenames[idx]
        gt_mel_path = self.input_to_gt[self.input_mel_filenames[idx]]

        input_mel = self.get_spectrogram(input_mel_path)[:, :2576]
        gt_mel = self.get_spectrogram(gt_mel_path)[:, :2576]

        # If only the magnitude is used another channel dimension is needed for pytorch
        input_mel = np.expand_dims(input_mel, axis=0)
        gt_mel = np.expand_dims(gt_mel, axis=0)
        sample = {'input_mel': input_mel, 'gt_mel': gt_mel}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_spectrogram(self, path):
        if not os.path.exists(path):
            print("Error: Spectrogram for file '" + path + "' does not exist! Aborting...")
            return None

        spec = np.load(path)
        return spec

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        input_mel = sample['input_mel']
        gt_mel = sample['gt_mel']
        input_mel = torch.from_numpy(input_mel).float()
        gt_mel = torch.from_numpy(gt_mel).float()
        return {'input_mel': input_mel, 'gt_mel': gt_mel}
