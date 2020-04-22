from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from Hyperparameters import sep, unet_width
from Util import get_spectrogram


class UNetDataset(Dataset):
    """
    Dataset for accessing data opints of the autoencoder output.
    """
    def __init__(self, root_dir, gt_dir, transform=None):
        """
        Initialise the dataset.
        :param root_dir: The path to the data points
        :param gt_dir: The path to the ground truth versions of the data points
        :param transform: Transformation to apply to the data points
        """

        self.root_dir = root_dir
        self.transform = transform
        # The input for this dataset is the output from the autoencoder
        input_mel_npys = Path(root_dir).rglob("*_output.npy")
        # The U-Net is trained to minimise the error between the autoencoder output
        # and the clean ("ground truth") versions of the synthesised files
        gt_mel_npys = Path(gt_dir).rglob("*_synth_mel.npy")
        self.input_mel_filenames = [str(npy) for npy in input_mel_npys]
        self.gt_mel_filenames = [str(npy) for npy in gt_mel_npys]

        # Create mappings between input and ground truth names (so that the order is correct)
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
        # This is included for completeness. Future versions could handle the retrieval of multiple data points
        # simultaneously
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get spectrogram
        # Convert idx to filename
        input_mel_path = self.input_mel_filenames[idx]
        gt_mel_path = self.input_to_gt[self.input_mel_filenames[idx]]

        # Get the spectrograms and trim to the length required by the U-Net
        input_mel = get_spectrogram(input_mel_path)[:, :unet_width]
        gt_mel = get_spectrogram(gt_mel_path)[:, :unet_width]

        # Add extra channel dimension for pytorch
        input_mel = np.expand_dims(input_mel, axis=0)
        gt_mel = np.expand_dims(gt_mel, axis=0)
        sample = {'input_mel': input_mel, 'gt_mel': gt_mel, 'filename': idx}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """
    Transformation used to convert ndarrays in sample to PyTorch tensors.
    """
    def __call__(self, sample):
        # Get ndarrays
        input_mel = sample['input_mel']
        gt_mel = sample['gt_mel']
        # Convert to float tensors
        input_mel = torch.from_numpy(input_mel).float()
        gt_mel = torch.from_numpy(gt_mel).float()
        return {'input_mel': input_mel, 'gt_mel': gt_mel, 'filename': sample['filename']}
