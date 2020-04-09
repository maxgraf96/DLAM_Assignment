import torch
from torch import nn
from torch.nn import functional as F

# from Freesound import Plot

# Parameters for CNN
kernel_size = 17
padding = int(kernel_size / 2)

# Parameters for ANN
ZDIMS_ANN = 100

class SpecVAE(nn.Module):
    """
    Superclass for spectrogram related VAE
    CNN and ANN implementations are child classes of this.
    """
    def __init__(self, epochs, is_plot=False):
        nn.Module.__init__(self)
        # Flag for plotting
        self.is_plot = is_plot
        # Counter for epoch (used for naming in plotting)
        self.total_epochs = epochs
        self.current_epoch = 1

    def set_epoch(self, epoch):
        self.current_epoch = epoch

class SpecVAECNN(SpecVAE):
    def __init__(self, epochs, is_plot=False):
        SpecVAE.__init__(self, epochs, is_plot)

        self.conv1 = nn.Conv2d(1, 2, kernel_size=kernel_size, padding=padding)
        self.conv21 = nn.Conv2d(2, 4, kernel_size=kernel_size, padding=padding)
        self.conv22 = nn.Conv2d(2, 4, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(4, 2, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)

    def encode(self, x):
        h1 = F.relu(self.conv1(x))
        mu = self.conv21(h1)
        logvar = self.conv22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.conv3(z))
        return F.relu(torch.sigmoid(self.conv4(h3)))

    def forward(self, x):
        mu, logvar = self.encode(x)

        # Plot latent representations if wanted
        # Currently plot it at the beginning, every epoch / 3 times, and at the end
        # if self.is_plot and (self.current_epoch == 1 or self.current_epoch % 5 == 0):
        #     Plot.plot_latent_representations_cnn(x, mu, is_plot_original=False)

        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class SpecVAEANN(SpecVAE):
    def __init__(self, spec_width, spec_height, epochs, is_plot=False):
        SpecVAE.__init__(self, epochs, is_plot)

        self.spec_width = spec_width
        self.spec_height = spec_height
        self.spec_dims = spec_width * spec_height

        self.fc1 = nn.Linear(self.spec_dims, 1024)
        self.fc1_1 = nn.Linear(1024, 768)
        self.fc1_2 = nn.Linear(768, 400)
        self.fc1_3 = nn.Linear(400, 200)
        self.fc21 = nn.Linear(200, ZDIMS_ANN)
        self.fc22 = nn.Linear(200, ZDIMS_ANN)

        self.fc3 = nn.Linear(ZDIMS_ANN, 200)
        self.fc3_1 = nn.Linear(200, 400)
        self.fc3_2 = nn.Linear(400, 768)
        self.fc3_3 = nn.Linear(768, 1024)

        self.fc4 = nn.Linear(1024, self.spec_dims)

    def encode(self, x):
        # Flatten input
        x = x.flatten()
        h1 = F.relu(self.fc1_3(F.relu(self.fc1_2(F.relu(self.fc1_1(F.relu(self.fc1(x))))))))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3_3(F.relu(self.fc3_2(F.relu(self.fc3_1(F.relu(self.fc3(z))))))))
        return F.relu(torch.sigmoid(self.fc4(h3)))

    def forward(self, x):
        mu, logvar = self.encode(x)

        # Plot latent representation at the beginning and end
        # if self.is_plot and (self.current_epoch == 1 or self.current_epoch == self.total_epochs):
        #     Plot.plot_latent_representations_ann(x, mu, is_plot_original=False)

        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        # "Unflatten" => turn vector back into spectrogram
        decoded = decoded.view(self.spec_width, self.spec_height)
        return decoded, mu, logvar