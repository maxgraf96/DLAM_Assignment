import os

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FreesoundDataset(Dataset):
    """
    Dataset containing files from freesound
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
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.freesound_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = os.path.join(self.root_dir, self.freesound_frame.iloc[idx, 0])
        # Load using file's default sample rate
        sig, sr = librosa.load(file_name, sr=None)
        spec = librosa.stft(sig)

        # Get descriptors
        descriptors = self.freesound_frame.iloc[idx, 1:]
        descriptors = np.array([descriptors])
        # TODO check what effect this has
        descriptors = descriptors.astype('string').reshape(-1, 2)
        sample = {'sound': spec, 'descriptors': descriptors}

        if self.transform:
            sample = self.transform(sample)

        return sample