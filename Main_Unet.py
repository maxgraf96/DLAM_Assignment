import os
import sys

import librosa
import numpy as np
import torch
from torch import optim, nn
import torch.nn
from torch.utils.data import DataLoader

import UnetDataset
from Util import map_to_range
from DatasetCreator import create_spectrogram
from Hyperparameters import sep, model_path, device, log_interval, batch_size_autoencoder, top_db, input_channels, \
    spec_height, spec_width, sample_rate, n_fft, hop_size, epochs_unet, batch_size_unet
from AEModel import AEModel
from Unet_Denoise import UNet, generate
from torch.nn import functional as F

from Util import plot_mel

loss_fn = nn.MSELoss()
root_dir = "data" + sep + "generated"
output_dir = "data" + sep + "ae_output"
model_path = "unet.torch"

global model

def loss_function(input, target):
    l2 = loss_fn(input, target)
    return l2

if __name__ == '__main__':
    # U-Net is applied after the autoencoder was trained
    if not os.path.exists(output_dir):
        print("Need autoencoder output data for training U-Net. Aborting...")
        sys.exit()

    if os.path.exists(model_path):
        print("Unet model exists. Loading model...")
        model = UNet(n_classes=1, depth=4, padding=True).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Unet model loaded.")

    else:
        # Create dataset
        transform = UnetDataset.ToTensor()
        dataset = UnetDataset.UnetDataset(root_dir=output_dir, gt_dir=root_dir, transform=transform)

        # Split into training and validation sets
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size_unet, shuffle=True, num_workers=8, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_unet, shuffle=True, num_workers=8, drop_last=True)

        model = UNet(n_classes=1, depth=4, padding=True).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        # Placeholders for loss tracking
        train_losses = []
        val_losses = []

        for epoch in range(1, epochs_unet + 1):
            model.train()
            train_loss = 0
            for batch_idx, data in enumerate(train_loader):
                # Convert tensors to cuda
                input = data['input_mel'].to(device)
                ground_truth = data['gt_mel'].to(device)
                optimizer.zero_grad()
                out = model(input)
                loss = loss_function(out, ground_truth)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0 and epoch % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(input), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                               loss.item() / len(input)))

            # Get average train loss for epoch
            current_avg_loss = train_loss / len(train_loader.dataset)
            train_losses.append(current_avg_loss)

            if epoch % 5 == 0:
                generate(model, input, "data" + sep + "synth" + sep + "chpn_op7_1.wav", with_return=False)

            print('====> Epoch: {} Average loss: {:.10f}'.format(
                epoch, train_loss / len(train_loader.dataset)))

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # Convert tensors to cuda
                    input = data['input_mel'].to(device)
                    ground_truth = data['gt_mel'].to(device)
                    out = model(input)
                    loss = loss_function(out, ground_truth)
                    val_loss += loss.item()

            val_loss /= len(val_loader.dataset)
            print('====> Validation set loss: {:.4f}'.format(val_loss))

            val_losses.append(val_loss)

        # Save losses
        np.save("train_losses_unet", np.array(train_losses))
        np.save("val_losses_unet", np.array(val_losses))

        # Save model so we don't have to train every time
        torch.save(model.state_dict(), model_path)