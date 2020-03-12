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
from SpecVAE import SpecVAECNN, SpecVAEANN
from VAE import VAE

# Hyperparameters
MODEL_PATH = "spec_vae.model"
cuda = torch.cuda.is_available()
batch_size = 4
epochs = 10
seed = 1
log_interval = 10

torch.manual_seed(seed)
device = torch.device("cuda" if cuda else "cpu")

# deactivate cuda for now
device = torch.device("cpu")
cuda = False

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# Establish connection to freesound.org
# NB: The auth_code needs to be regenerated every day!
fs_interface = FreesoundInterface(auth_code="iBGdXAJ6f3M9vdNJwrDqtD8cicLaIF")

root_dir = "test1"
# Terms and instrument categories must always have the same length
terms = ["synth", "piano", "guitar"]
instrument_categories = ["synth", "piano", "guitar"]
tags = []
# Only look for sounds with a single sonic event
ac_single_event = True
# Look for single note
ac_note_names = ["A2"]
# Path to dataset csv
path_csv = "data/" + root_dir + "/" + root_dir + ".csv"

# Downloads sounds using the specifications provided above
# DatasetCreator.download_sounds(fs_interface, root_dir, terms, tags, instrument_categories, ac_single_event, ac_note_names, path_csv)

# Import downloaded pack
root_dir = "Arturia MicroBrute Saw Oscillator"
path_pack_csv = "data" + os.path.sep + root_dir + os.path.sep + root_dir + ".csv"
DatasetCreator.import_offline_pack(root_dir, path_pack_csv, instrument_category="synth")

# Create dataset
transform = ToTensor()
fs_dataset = FreesoundDataset(csv_file="data" + os.path.sep + root_dir + os.path.sep + root_dir + ".csv",
                              root_dir="data" + os.path.sep + root_dir,
                              transform=transform)

# Create dataloader
train_loader = DataLoader(fs_dataset, batch_size=4,
                          shuffle=True, num_workers=8)

# Create and initialise VAE
model = SpecVAECNN().to(device)
# model = SpecVAEANN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
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
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        # Use if nans start showing up again
        # nans = (recon_batch != recon_batch).any()
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
    if not os.path.exists(MODEL_PATH):
        for epoch in range(1, epochs + 1):
            train(epoch)
            # test(epoch)
            with torch.no_grad():
                # 3 => 3x1 matrix, 4 => number of channels in the bottleneck dimension of VAE /, spec height, spec width
                sample = torch.randn(3, 4, 257, 259).to(device)
                sample = model.decode(sample).cpu()
                save_image(sample.view(3, 1, 257, 259),
                           'results/sample_' + str(epoch) + '.png')
        # Save model so we don't have to train every time
        torch.save(model, MODEL_PATH)
    else:
        model = torch.load(MODEL_PATH)
        model.eval()

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
