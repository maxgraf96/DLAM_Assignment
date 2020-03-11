import os

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Desired signal length in seconds
SIG_LENGTH = 1
# Sample rate
GLOBAL_SR = 44100

class FreesoundDataset(Dataset):
    """
    Dataset created from previously fetched files from freesound.
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
        sig, sr = librosa.load(file_name, sr=None, mono=True)
        # Normalize signal
        max_val = np.max((np.abs(np.min(sig)), np.max(sig)))
        sig = np.divide(sig, max_val)
        # Trim silence at beginning and end
        sig = librosa.effects.trim(sig)[0]
        # Fix to specified length
        sig = librosa.util.fix_length(sig, SIG_LENGTH * GLOBAL_SR)
        # Calculate Mel spectrogram
        # Use 87 mel bins to make the resulting spec quadratic 87x87
        mel_spec = librosa.feature.melspectrogram(sig, GLOBAL_SR, n_mels=87)

        # Normalise mel spectrogram
        mellog = np.log(mel_spec + 1e-9)
        mel_spec = librosa.util.normalize(mellog)
        # Add "color" channel dimension
        mel_spec = np.expand_dims(mel_spec, axis=0)

        # Get descriptors: Currently this is the Class Number
        descriptors = self.freesound_frame.iloc[idx, 2]
        descriptors = np.array([descriptors])
        sample = {'sound': mel_spec, 'descriptors': descriptors}

        if self.transform:
            sample = self.transform(sample)

        return sample['sound'], sample['descriptors']


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        spec, descriptors = sample['sound'], sample['descriptors']
        sound = torch.from_numpy(spec)
        # TODO implement number mapping for descriptors
        return {'sound': sound, 'descriptors': torch.from_numpy(descriptors)}