from __future__ import print_function
import os

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from Freesound import FS_DatasetCreator
from Freesound.FS_FreesoundDataset import FreesoundDataset, ToTensor, SPEC_DIMS, SPEC_DIMS_W, SPEC_DIMS_H
from Freesound.FS_FreesoundInterface import FreesoundInterface
from SpecVAE import SpecVAEANN, ZDIMS_ANN

from Hyperparameters import cuda, batch_size_cnn, batch_size_ann, epochs, seed, log_interval

torch.manual_seed(seed)
device = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    if type(model).__name__ is "SpecVAEANN":
        BCE = F.binary_cross_entropy(recon_x, x[0][0], reduction='mean')
    else:
        BCE = F.binary_cross_entropy(recon_x, x, reduction='mean')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Normalise KLD by batch size and size of spectrogram
    KLD /= batch_size * SPEC_DIMS

    return BCE + KLD

def train(epoch):
    # Set current epoch in model (used for plotting)
    model.set_epoch(epoch)
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        print(label)
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

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(batch_size, 1, SPEC_DIMS_W, SPEC_DIMS_H)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def save_sample_CNN():
    with torch.no_grad():
        # 3 => 3x1 matrix, 4 => number of channels in the bottleneck dimension of VAE /, spec height, spec width
        sample = torch.randn(3, 4, 257, 259).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(3, 1, 257, 259),
                   'results/sample_' + str(epoch) + '.png')

def save_sample_ANN():
    with torch.no_grad():
        # 3 => 3x1 matrix, 4 => number of channels in the bottleneck dimension of VAE /, spec height, spec width
        sample = torch.randn(64, ZDIMS_ANN).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, SPEC_DIMS_W, SPEC_DIMS_H),
                   'results/sample_' + str(epoch) + '.png')

if __name__ == "__main__":
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
    FS_DatasetCreator.import_offline_pack(root_dir, path_pack_csv, instrument_category="synth")

    # Create dataset
    transform = ToTensor()
    fs_dataset = FreesoundDataset(csv_file="data" + os.path.sep + root_dir + os.path.sep + root_dir + ".csv",
                                  root_dir="data" + os.path.sep + root_dir,
                                  transform=transform)

    # Create and initialise VAE
    # model = SpecVAECNN(epochs, is_plot=True).to(device)
    model = SpecVAEANN(epochs, is_plot=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Split into training and test sets
    train_size = int(0.8 * len(fs_dataset))
    test_size = len(fs_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(fs_dataset, [train_size, test_size])

    # Create dataloaders
    batch_size = batch_size_cnn if type(model).__name__ is "SpecVAECNN" else batch_size_ann
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    #if not os.path.exists(model_path):
    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)
        if type(model).__name__ is "SpecVAECNN":
            save_sample_CNN()
        elif type(model).__name__ is "SpecVAEANN":
            save_sample_ANN()

    # Save model so we don't have to train every time
    #torch.save(model, model_path)
    #else:
        #model = torch.load(model_path)
        #model.eval()
#
