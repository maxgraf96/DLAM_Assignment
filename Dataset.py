import os
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.utils.data import Dataset


def map_to_zero_one(spec, min_from, max_from):
    return map_to_range(spec, min_from, max_from, 0, 1)

def map_to_range(spec, input_start, input_end, output_start, output_end):
    copy = np.copy(spec)
    slope = 1.0 * (output_end - output_start) / (input_end - input_start)
    copy = output_start + slope * (copy - input_start)
    return copy

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
        mel_piano_npys = Path(root_dir).rglob("*_piano_mel.npy")
        mel_synth_npys = Path(root_dir).rglob("*_synth_mel.npy")
        self.piano_mel_filenames = [str(npy) for npy in mel_piano_npys]
        self.synth_mel_filenames = [str(npy) for npy in mel_synth_npys]
        self.length = len(self.piano_mel_filenames)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get spectrogram
        # Convert idx to filename
        mel_piano_path = self.piano_mel_filenames[idx]
        mel_synth_path = self.synth_mel_filenames[idx]

        mel_piano = self.get_spectrogram(mel_piano_path)
        mel_synth = self.get_spectrogram(mel_synth_path)

        # If only the magnitude is used another channel dimension is needed for pytorch
        mel_piano = np.expand_dims(mel_piano, axis=0)
        mel_synth = np.expand_dims(mel_synth, axis=0)
        sample = {'piano_mel': mel_piano, 'synth_mel': mel_synth}

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
        piano_mel = sample['piano_mel']
        synth_mel = sample['synth_mel']
        piano_mel = torch.from_numpy(piano_mel).float()
        synth_mel = torch.from_numpy(synth_mel).float()
        return {'piano_mel': piano_mel, 'synth_mel': synth_mel}
