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
        piano_npys = Path(root_dir).rglob("*_piano.npy")
        self.filenames = [str(npy) for npy in piano_npys]
        synth_npys = Path(root_dir).rglob("*_synth.npy")
        self.synth_filenames = [str(npy) for npy in synth_npys]
        self.length = len(self.filenames)

        # Load one spectrogram to get width and height
        mapped = self.get_spectrogram(self.filenames[0])
        self.spec_height = mapped.shape[0]
        self.spec_width = mapped.shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get spectrogram
        # Convert idx to filename
        path = self.filenames[idx]
        path_synth = self.synth_filenames[idx]

        spec = self.get_spectrogram(path)
        spec_synth = self.get_spectrogram(path_synth)

        # flip axes
        spec = np.flip(np.swapaxes(spec, 0, 1).copy(), axis=0).copy()
        spec_synth = np.flip(np.swapaxes(spec_synth, 0, 1).copy(), axis=0).copy()

        # If only the magnitude is used another channel dimension is needed for pytorch
        # spec = np.expand_dims(spec, axis=0)
        # spec_synth = np.expand_dims(spec_synth, axis=0)
        sample = {'sound': spec, 'sound_synth': spec_synth
            # , 'filename': path
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_spectrogram(self, path):
        if not os.path.exists(path):
            print("Error: Spectrogram for file '" + path + "' does not exist! Aborting...")
            return None

        stft = np.load(path)
        return stft

    def get_spec_dims(self):
        return self.spec_width, self.spec_height

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
        spec_synth = sample['sound_synth']
        sound = torch.from_numpy(spec).float()
        sound_synth = torch.from_numpy(spec_synth).float()
        return {'sound': sound, 'sound_synth': sound_synth
            # , 'filename': sample['filename']
                }
