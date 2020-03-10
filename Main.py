from __future__ import print_function
import argparse
import os

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from pathlib import Path
import pandas as pd
from FreesoundDataset import FreesoundDataset
from Freesound_connector import Freesound_connector
from VAE import VAE

# Hyperparameters
cuda = torch.cuda.is_available()
batch_size = 128
epochs = 10
seed = 1
log_interval = 10

torch.manual_seed(seed)
device = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=batch_size, shuffle=True, **kwargs)

# Establish connection with freesound

# Establish connection to freesound.org
freesound_connector = Freesound_connector(auth_code="mp5UjlwibNuMLfgvResm00qRU1ovFL")

term = "synth bright"
tags = ["bright", "synth"]
tags_str = ",".join(tags)
synth_sounds = freesound_connector.search(term, tags=tags)

# Number of files to download
max_files = 20

# Folder to which the files should be saved
save_dir = "data/" + term + "/"
# Only load data if not exists yet
if not os.path.exists(save_dir) or len(os.listdir(save_dir)) == 0:
    print("Directory is empty")
    # Create list for annotation data
    data = []
    # Create folder (no effect if folder already exists)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    # Store the *.wav files

    counter = 0
    for sound in synth_sounds:
        # Get sound data
        id = str(sound["id"])
        name = str(sound["name"])
        # Download raw audio data
        wav = freesound_connector.download_sound(id)
        # Save raw audio data
        f = open(save_dir + name, 'wb')
        f.write(wav)
        f.close()
        # Store annotation data
        data.append([name, tags_str])

        print("Saved " + name + ".")
        counter = counter + 1
        if counter >= max_files:
            break

    # Convert annotation data to dataframe
    df = pd.DataFrame(data, columns=["Name", "tags"])
    # Save dataframe to CSV
    df.to_csv(save_dir + term + ".csv", index=False)
    print("Done saving data for term '" + term + "'.")


freesound_dataset = FreesoundDataset(csv_file=save_dir + term + ".csv", root_dir=save_dir)

a = 2
# TODO do cool stuff with dataset :D


# model = VAE().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# Reconstruction + KL divergence losses summed over all elements and batch
# def loss_function(recon_x, x, mu, logvar):
#     # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
#     BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
#
#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#
#     return BCE + KLD


# def train(epoch):
#     model.train()
#     train_loss = 0
#     for batch_idx, (data, _) in enumerate(train_loader):
#         data = data.to(device)
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(data)
#         loss = loss_function(recon_batch, data, mu, logvar)
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()
#         if batch_idx % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader),
#                 loss.item() / len(data)))
#
#     print('====> Epoch: {} Average loss: {:.4f}'.format(
#           epoch, train_loss / len(train_loader.dataset)))
#
#
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
# if __name__ == "__main__":
#     for epoch in range(1, epochs + 1):
#         train(epoch)
#         test(epoch)
#         with torch.no_grad():
#             sample = torch.randn(64, 4, 28, 28).to(device)
#             sample = model.decode(sample).cpu()
#             save_image(sample.view(64, 1, 28, 28),
#                        'results/sample_' + str(epoch) + '.png')
