import torch
from torch import nn
from torch.nn import functional as F

class SpecVAE(nn.Module):
    def __init__(self):
        super(SpecVAE, self).__init__()

        self.conv1 = nn.Conv2d(1, 2, kernel_size=5, padding=2)
        self.conv21 = nn.Conv2d(2, 4, kernel_size=5, padding=2)
        self.conv22 = nn.Conv2d(2, 4, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(4, 2, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(2, 1, kernel_size=5, padding=2)

        self.fc1 = nn.Linear(1025 * 87, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 1025 * 87)

    def encode(self, x):
        h1 = F.relu(self.conv1(x))
        return self.conv21(h1), self.conv22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.conv3(z))
        return torch.sigmoid(self.conv4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
