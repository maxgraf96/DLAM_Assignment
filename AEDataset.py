from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from Util import get_spectrogram


class AssignmentDataset(Dataset):
    """
    Dataset for accessing data points of auto generated audio tracks from MIDI.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory containing the stored spectrogram data points (*.npy files).
            transform (callable, optional): Transformation to apply to samples. This is used to convert the tensors
            to GPU-compatible tensors
        """

        self.root_dir = root_dir
        self.transform = transform
        # Get all piano and synth *.npy files
        mel_piano_npys = Path(root_dir).rglob("*_piano_mel.npy")
        mel_synth_npys = Path(root_dir).rglob("*_synth_mel.npy")
        self.piano_mel_filenames = [str(npy) for npy in mel_piano_npys]
        self.synth_mel_filenames = [str(npy) for npy in mel_synth_npys]
        self.length = len(self.piano_mel_filenames)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # This is included for completeness. Future versions could handle the retrieval of multiple data points
        # simultaneously
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get Mel spectrogram from *.npy file
        # Convert idx to filename
        mel_piano_path = self.piano_mel_filenames[idx]
        mel_synth_path = self.synth_mel_filenames[idx]

        mel_piano = get_spectrogram(mel_piano_path)
        mel_synth = get_spectrogram(mel_synth_path)

        # Because only the magnitude is used, an additional channel dimension is needed for PyTorch
        mel_piano = np.expand_dims(mel_piano, axis=0)
        mel_synth = np.expand_dims(mel_synth, axis=0)

        # Pack sample
        sample = {'piano_mel': mel_piano, 'synth_mel': mel_synth}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """
    Transformation used to convert ndarrays in sample to PyTorch tensors.
    """
    def __call__(self, sample):
        # Get ndarrays
        piano_mel = sample['piano_mel']
        synth_mel = sample['synth_mel']
        # Convert to float tensors
        piano_mel = torch.from_numpy(piano_mel).float()
        synth_mel = torch.from_numpy(synth_mel).float()
        return {'piano_mel': piano_mel, 'synth_mel': synth_mel}
