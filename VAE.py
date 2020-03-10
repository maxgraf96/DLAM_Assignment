import torch
from torch import nn
from torch import functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.conv1 = nn.Conv2d(1, 2, kernel_size=5, padding=2)
        self.conv21 = nn.Conv2d(2, 4, kernel_size=5, padding=2)
        self.conv22 = nn.Conv2d(2, 4, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(4, 2, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(2, 1, kernel_size=5, padding=2)

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        # return self.fc21(h1), self.fc22(h1)
        h1 = F.relu(self.conv1(x))
        return self.conv21(h1), self.conv22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        # h3 = F.relu(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))
        h3 = F.relu(self.conv3(z))
        return torch.sigmoid(self.conv4(h3))

    def forward(self, x):
        # mu, logvar = self.encode(x.view(-1, 784))
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar