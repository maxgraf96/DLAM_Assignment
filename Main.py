import os

import librosa
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

import Dataset
from Dataset import standardise
import DatasetCreator
from Hyperparameters import cuda, batch_size_ann, batch_size_cnn, epochs, seed, log_interval, sep, sample_rate, n_fft, hop_size
# Initialise dataset
from Model import Model
from SpecVAE import SpecVAEANN, SpecVAECNN

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
global batch_size
global spec_dim

def generate(path):
    # Load signal data, sample rate should always be 22050
    sig, sr = librosa.load(path)

    sig = librosa.util.fix_length(sig, 30 * sample_rate)

    # Create power spectrogram
    # spec_orig = librosa.stft(sig, n_fft, hop_length=hop_size, window='hann')
    spec = librosa.stft(sig, n_fft, hop_length=hop_size, window='hann')
    # Split into mag and phase
    mag = np.abs(spec)
    phase = np.angle(spec)
    spec = np.stack((mag, phase))

    result = np.zeros(spec.shape)
    step = spec_dim * batch_size
    for frame in range(0, spec.shape[2], step):
        if frame + step > spec.shape[2]:
            break

        current = spec[:, :, frame : frame + step]
        current = current.reshape(batch_size, 2, spec_dim, spec_dim)
        result_frames = main.generate(current).reshape(batch_size, 2, spec_dim, spec_dim)
        result_frames = result_frames.cpu().numpy()

        # Convert from batches to sequential
        for batch in range(result_frames.shape[0]):
            current_square = result_frames[batch]
            start = frame + batch * spec_dim
            end = frame + (batch + 1) * spec_dim
            result[:, :, start : end] = current_square

    final = result[0] + (result[1] * 1j)
    sig = librosa.istft(final, hop_length=hop_size, win_length=n_fft, window='hann')

    # result = result[0][0].cpu().numpy()
    # sig = librosa.istft(result, win_length=n_fft, hop_length=hop_size, window='hann')

    return sig

if __name__ == '__main__':
    # Create dataset
    transform = Dataset.ToTensor()
    dataset = Dataset.AssignmentDataset(root_dir=root_dir, transform=transform)

    # Set batch size
    # batch_size = batch_size_ann
    batch_size = batch_size_cnn
    # Create and initialise VAE
    spec_dim = dataset.get_spec_dims()

    if os.path.exists(model_path):
        model = torch.load(model_path).to(device)
        main = Model(model, device, batch_size, log_interval)

    else:
        # model = SpecVAEANN(spec_dim, epochs, is_plot=False).to(device)
        model = SpecVAECNN(spec_dim, epochs).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Split into training and test sets
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_dataset = torch.utils.data.Subset(dataset, np.arange(train_size))
        test_dataset = torch.utils.data.Subset(dataset, np.arange(train_size, dataset.length))

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

        main = Model(model, device, batch_size, log_interval)

        for epoch in range(1, epochs + 1):
            main.train(epoch, train_loader, optimizer)
            main.test(test_loader, epoch)

        # Save model so we don't have to train every time
        torch.save(model, model_path)

    # Generate something
    gen = generate("data/piano/chpn_op7_1.wav")

    # Display (only works on IPython notebooks)
    librosa.output.write_wav("output.wav", gen, sample_rate)