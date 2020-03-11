from __future__ import print_function
import argparse
import os

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from pathlib import Path
import pandas as pd

import DatasetCreator
from FreesoundDataset import FreesoundDataset, ToTensor
from FreesoundInterface import FreesoundInterface
from SpecVAE import SpecVAE
from VAE import VAE

# Hyperparameters
cuda = torch.cuda.is_available()
batch_size = 128
epochs = 5
seed = 1
log_interval = 10

torch.manual_seed(seed)
device = torch.device("cuda" if cuda else "cpu")

# deactivate cuda for now
device = torch.device("cpu")
cuda = False

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=batch_size, shuffle=True, **kwargs)


# Establish connection to freesound.org
# NB: The auth_code needs to be regenerated every day!
fs_interface = FreesoundInterface(auth_code="mp5UjlwibNuMLfgvResm00qRU1ovFL")

root_dir = "test1"
# Terms and instrument categories must always have the same length
terms = ["synth note", "piano note", "guitar note"]
instrument_categories = ["synth", "piano", "guitar"]
tags = []

# Path to dataset csv
path_csv = "data/" + root_dir + "/" + root_dir + ".csv"

# First check if any data for this dataset was queried yet
# If not, query all the data
if not os.path.exists(path_csv):
    for i in range(len(terms)):
        # Create part: This could be something like a set of files describing a bright synth
        DatasetCreator.create_part(fs_interface, root_dir, terms[i], tags, instrument_categories[i], class_nr=i)

else:
    # Some part of the data already exists => go over dataframe and check which parts already exist
    # And only add those parts that don't exist yet
    # Get instrument category column
    col_category = pd.read_csv(path_csv).values[:, 1]
    for i in range(len(terms)):
        if instrument_categories[i] not in col_category:
            # Create part: This could be something like a set of files describing a bright synth
            DatasetCreator.create_part(fs_interface, root_dir, terms[i], tags, instrument_categories[i], class_nr=i)

transform = ToTensor()
fs_dataset = FreesoundDataset(csv_file="data/" + root_dir + "/" + root_dir + ".csv", root_dir="data/" + root_dir,
                              transform=transform)

example = fs_dataset.__getitem__(0)

# TODO do cool stuff with dataset :D
# Create dataloader
train_loader = DataLoader(fs_dataset, batch_size=4,
                          shuffle=True, num_workers=4)

# Create and initialise VAE
model = SpecVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    test = enumerate(train_loader)
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(epoch)
        # test(epoch)
        with torch.no_grad():
            # 64 => 8x8 matrix, 4 => bottleneck dimension of VAE / Seems to be batch size???, spec height, spec width
            sample = torch.randn(3, 4, 1025, 87).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(3, 1, 1025, 87),
                       'results/sample_' + str(epoch) + '.png')

# def test(epoch):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, (data, _) in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#             if i == 0:
#                 n = min(data.size(0), 8)
#                 comparison = torch.cat([data[:n],
#                                       recon_batch.view(batch_size, 1, 28, 28)[:n]])
#                 save_image(comparison.cpu(),
#                          'results/reconstruction_' + str(epoch) + '.png', nrow=n)
#
#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))
#
