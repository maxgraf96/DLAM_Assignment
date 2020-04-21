import torch
from torch import nn

from Hyperparameters import batch_size_autoencoder, spec_height, input_channels, spec_width

# Parameters for CNN
padding = 0

class Autoencoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(spec_height, 1), padding=padding),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64, 128, kernel_size=(1, 2), stride=(1, 2), padding=padding),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(128, 256, kernel_size=(1, 2), stride=(1, 2), padding=padding),
            nn.BatchNorm2d(256),
        )

        # Decoder using transposed convolution layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(1, 2), stride=(1, 2), padding=padding),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.ConvTranspose2d(128, 64, kernel_size=(1, 2), stride=(1, 2), padding=padding),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.ConvTranspose2d(64, input_channels, kernel_size=(spec_height, 4), padding=padding),
            nn.Sigmoid()
        )

        # Included to query the dimensionality of the latent representations
        # test_encode = self.encoder(torch.randn(batch_size_cnn, input_channels, spec_height, spec_width))
        # test_decode = self.decoder(test_encode)

    def bottleneck(self, x):
        """
        Create the latent representation
        :param x: The input tensor
        :return: The latent representation
        """
        mu = x.clone()
        logvar = x.clone()
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Create latent representation by reparametrising the input
        :param mu: The mean
        :param logvar:
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass
        :param x: The input tensor
        :return: The output tensor (mel spectrogram), its mean and variance
        """
        x = self.encoder(x)
        z, mu, logvar = self.bottleneck(x)
        mel = self.decoder(z)
        return mel, mu, logvar

    def forward_sample(self, sample):
        """
        Forward a single element (with no accompanying synth data) => only used for testing
        :param sample: The sample to forward
        :return:
        """
        sample = self.encoder(sample)
        z, mu, logvar = self.bottleneck(sample)
        mel = self.decoder(z)
        return mel, mu, logvar