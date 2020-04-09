import os

import librosa
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

import Dataset
import DatasetCreator
from Hyperparameters import cuda, batch_size_ann, epochs, seed, log_interval, sep, sample_rate, n_fft, hop_size
# Initialise dataset
from Model import Model
from SpecVAE import SpecVAEANN

model_path = "model.torch"

# Pytorch init
torch.manual_seed(seed)
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# device = "cpu"
# cuda = False

# Initialise dataset (create spectrograms if not exist)
DatasetCreator.initialise_dataset()
root_dir = "data" + sep + "generated"

global main
global dataset

def generate(path):
    # Load signal data, sample rate should always be 22050
    sig, sr = librosa.load(path)

    sig = librosa.util.fix_length(sig, 30 * sample_rate)

    # Create power spectrogram
    spec = np.abs(librosa.stft(sig, n_fft, hop_length=hop_size, window='hann')) ** 2

    # Normalize
    mean, std = dataset.get_mean_std()
    spec = (spec - mean) / std

    result = main.generate(spec)

    spec = result.cpu().numpy()

    sig = librosa.istft(spec, hop_length=hop_size, win_length=n_fft, window='hann')
    return sig

if __name__ == '__main__':
    # Create dataset
    transform = Dataset.ToTensor()
    dataset = Dataset.AssignmentDataset(root_dir=root_dir, transform=transform)

    # Create and initialise VAE
    spec_width, spec_height = dataset.get_spec_dims()
    # Set batch size
    batch_size = batch_size_ann

    if os.path.exists(model_path):
        model = torch.load(model_path)
        main = Model(model, device, batch_size, log_interval)

    else:
        model = SpecVAEANN(spec_width, spec_height, epochs, is_plot=False).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Split into training and test sets
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        main = Model(model, device, batch_size, log_interval)

        for epoch in range(1, epochs + 1):
            main.train(epoch, train_loader, optimizer)
            main.test(test_loader, epoch)

        # Save model so we don't have to train every time
        torch.save(model, model_path)

    # Generate something
    sig = generate("data/piano/chp_op18.wav")
    librosa.output.write_wav("output.wav", sig, sample_rate)