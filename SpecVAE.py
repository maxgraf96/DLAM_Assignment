import numpy as np
import librosa.display
import torch
from torch import nn
from torch.nn import functional as F
from Hyperparameters import batch_size_cnn, spec_height, input_channels, hop_size, sample_rate, spec_width

# from Freesound import Plot

# Parameters for CNN
# kernel_size = 0
padding = 0
stride = 1

# Parameters for ANN
ZDIMS_ANN = 100
channels_last = 8
unflatten_dims = None

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
    def forward(self, input):
        unflattened = input.view(unflatten_dims)
        return unflattened

class SpecVAECNN(SpecVAE):
    def __init__(self, epochs, dataset_length, is_plot=False):
        SpecVAE.__init__(self, epochs, dataset_length, is_plot)

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(2, 1), stride=(1, 1), padding=padding),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64, 128, kernel_size=(2, 2), stride=(1, 2), padding=padding),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(128, 256, kernel_size=(2, 2), stride=(1, 2), padding=padding),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.Conv2d(256, 512, kernel_size=(2, 2), stride=1, padding=padding),
            nn.BatchNorm2d(512),
            # Flatten()
        )

        self.decoder = nn.Sequential(
            # UnFlatten(),
            nn.ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=1, padding=padding),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(1, 2), padding=padding),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(1, 2), padding=padding),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.ConvTranspose2d(64, input_channels, kernel_size=(2, 1), stride=(1, 1), padding=padding),
            nn.Sigmoid()
        )

        test_encode = self.encoder(torch.randn(batch_size_cnn, input_channels, spec_height, spec_width))
        test_decode = self.decoder(test_encode)
        decoder_shape = test_decode.shape
        H_DIMS_CNN = test_encode.shape[0]
        ZDIMS_CNN = 15

        self.spec_width = spec_width
        self.spec_height = spec_height

        # self.fc1 = nn.Linear(H_DIMS_CNN, ZDIMS_CNN)
        # self.fc2 = nn.Linear(H_DIMS_CNN, ZDIMS_CNN)
        # self.fc3 = nn.Linear(ZDIMS_CNN, H_DIMS_CNN)

    def bottleneck(self, h):
        # mu, logvar = self.fc1(h), self.fc2(h)
        # z = self.reparameterize(mu, logvar)
        mu = h
        logvar = h
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        # z = self.fc3(z)
        z = self.decoder(z)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

def plot_final(data):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 8))
    plt.subplot(1, 1, 1)
    data_db = librosa.amplitude_to_db(data, ref=np.max)
    librosa.display.specshow(data_db, y_axis='log', x_axis='time', sr=sample_rate, hop_length=hop_size)
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def plot_final_mel(data):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 8))
    plt.subplot(1, 1, 1)
    data_db = librosa.power_to_db(data, ref=np.max)
    librosa.display.specshow(data_db, y_axis='mel', x_axis='time', sr=sample_rate, hop_length=hop_size)
    plt.colorbar(format='%+2.0f dB')
    plt.show()

# class SpecVAEANN(SpecVAE):
#     def __init__(self, spec_dim, epochs, is_plot=False):
#         SpecVAE.__init__(self, epochs, is_plot)
#
#         # self.spec_width = spec_width
#         # self.spec_height = spec_height
#         self.spec_dim = spec_dim
#
#         self.fc1 = nn.Linear(self.spec_dim, 1024)
#         self.fc1_1 = nn.Linear(1024, 768)
#         self.fc1_2 = nn.Linear(768, 400)
#         self.fc1_3 = nn.Linear(400, 200)
#         self.fc21 = nn.Linear(200, ZDIMS_ANN)
#         self.fc22 = nn.Linear(200, ZDIMS_ANN)
#
#         self.fc3 = nn.Linear(ZDIMS_ANN, 200)
#         self.fc3_1 = nn.Linear(200, 400)
#         self.fc3_2 = nn.Linear(400, 768)
#         self.fc3_3 = nn.Linear(768, 1024)
#
#         self.fc4 = nn.Linear(1024, self.spec_dim)
#
#     def encode(self, x):
#         # Flatten input
#         x = x.flatten()
#         h1 = F.relu(self.fc1_3(F.relu(self.fc1_2(F.relu(self.fc1_1(F.relu(self.fc1(x))))))))
#         return self.fc21(h1), self.fc22(h1)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std
#
#     def decode(self, z):
#         h3 = F.relu(self.fc3_3(F.relu(self.fc3_2(F.relu(self.fc3_1(F.relu(self.fc3(z))))))))
#         return torch.sigmoid(self.fc4(h3))
#
#     def forward(self, x):
#         mu, logvar = self.encode(x)
#
#         # Plot latent representation at the beginning and end
#         # if self.is_plot and (self.current_epoch == 1 or self.current_epoch == self.total_epochs):
#         #     Plot.plot_latent_representations_ann(x, mu, is_plot_original=False)
#
#         z = self.reparameterize(mu, logvar)
#         decoded = self.decode(z)
#         # "Unflatten" => turn vector back into spectrogram
#         # decoded = decoded.view(self.spec_height, self.spec_width)
#         return decoded, mu, logvar