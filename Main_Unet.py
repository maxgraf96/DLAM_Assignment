import os
import sys

import librosa
import numpy as np
import torch
from torch import optim, nn
import torch.nn
from torch.utils.data import DataLoader

import UnetDataset
from Dataset import map_to_range
from DatasetCreator import create_spectrogram
from Hyperparameters import sep, model_path, device, log_interval, batch_size_cnn, top_db, input_channels, \
    spec_height, spec_width, sample_rate, n_fft, hop_size, epochs_unet, batch_size_unet
from Model import Model
from Unet_Denoise import UNet, generate
from torch.nn import functional as F

from Util import plot_final_mel

loss_fn = nn.MSELoss()
root_dir = "data" + sep + "generated"
output_dir = "data" + sep + "ae_output"
model_path = "unet.torch"

global model

def loss_function(input, target):
    if input.shape != target.shape:
        # Get smaller one
        input_width = input.shape[3]
        target_width = target.shape[3]
        if input_width < target_width:
            target = target[:, :, :, :input_width]
        else:
            input = input[:, :, :, :target_width]

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

        train_dataset = dataset

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size_unet, shuffle=True, num_workers=8, drop_last=True)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size_cnn, shuffle=True, num_workers=8, drop_last=True)

        model = UNet(n_classes=1, depth=4, padding=True).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

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

            if epoch % 5 == 0:
                generate(model, "data" + sep + "synth" + sep + "chpn_op7_1.wav", with_return=False)

            print('====> Epoch: {} Average loss: {:.10f}'.format(
                epoch, train_loss / len(train_loader.dataset)))

        # Save model so we don't have to train every time
        torch.save(model.state_dict(), model_path)