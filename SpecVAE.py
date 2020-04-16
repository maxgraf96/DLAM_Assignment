import librosa.display
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from Hyperparameters import batch_size_cnn, spec_height, input_channels, hop_size, sample_rate, spec_width, top_db

# from Freesound import Plot

# Parameters for CNN
# kernel_size = 0

padding = 0
stride = 0

# Parameters for ANN
ZDIMS_ANN = 100
unflatten_dims : torch.Tensor = None

class SpecVAE(nn.Module):
    """
    Superclass for spectrogram related VAE
    CNN and ANN implementations are child classes of this.
    """
    def __init__(self, epochs, dataset_length, is_plot=False):
        nn.Module.__init__(self)
        # Flag for plotting
        self.is_plot = is_plot
        # Counter for epoch (used for naming in plotting)
        self.total_epochs = epochs
        self.current_epoch = 1
        self.dataset_length = dataset_length

    def set_epoch(self, epoch):
        self.current_epoch = epoch

class Flatten(nn.Module):
    def forward(self, input):
        global unflatten_dims
        if unflatten_dims is None:
            unflatten_dims = input.shape
        return input.view(-1)

class UnFlatten(nn.Module):
    def forward(self, input: torch.Tensor):
        unflattened = input.view(unflatten_dims)
        return unflattened

class SpecVAECNN(SpecVAE):
    def __init__(self, epochs, dataset_length, is_plot=False):
        SpecVAE.__init__(self, epochs, dataset_length, is_plot)

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(spec_height, 1), padding=padding),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64, 128, kernel_size=(1, 2), stride=(1, 2), padding=padding),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(128, 256, kernel_size=(1, 2), stride=(1, 2), padding=padding),
            nn.BatchNorm2d(256),
            # Flatten()
        )

        self.decoder = nn.Sequential(
            # UnFlatten(),
            nn.ConvTranspose2d(256, 128, kernel_size=(1, 2), stride=(1, 2), padding=padding),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.ConvTranspose2d(128, 64, kernel_size=(1, 2), stride=(1, 2), padding=padding),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.ConvTranspose2d(64, input_channels, kernel_size=(spec_height, 4), padding=padding),
            nn.Sigmoid()
        )

        test_encode = self.encoder(torch.randn(batch_size_cnn, input_channels, spec_height, spec_width))
        test_decode = self.decoder(test_encode)
        decoder_shape = test_decode.shape
        H_DIMS_CNN = test_encode.shape[0]
        ZDIMS_CNN = 128

        self.spec_width = spec_width
        self.spec_height = spec_height

        # self.mel2lin = Mel2LinCBHG(epochs, self.dataset_length)

        # self.fc1 = nn.Linear(H_DIMS_CNN, ZDIMS_CNN)
        # self.fc2 = nn.Linear(H_DIMS_CNN, ZDIMS_CNN)
        # self.fc3 = nn.Linear(ZDIMS_CNN, H_DIMS_CNN)

    def bottleneck(self, h):
        # mu, logvar = self.fc1(h), self.fc2(h)
        mu = h.clone()
        logvar = h.clone()
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        z, mu, logvar = self.bottleneck(x)
        # z = self.fc3(z)
        mel = self.decoder(z)
        return mel, mu, logvar

    def forward_sample(self, sample):
        """
        Forward a single element (with no accompanying synth data) => only used for testing
        :param sample:
        :return:
        """
        sample = self.encoder(sample)
        z, mu, logvar = self.bottleneck(sample)
        # z = self.fc3(z)
        mel = self.decoder(z)
        return mel, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]