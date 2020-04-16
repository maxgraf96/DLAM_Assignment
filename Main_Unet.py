import os
import sys

import torch
from torch import optim
from torch.utils.data import DataLoader

import UnetDataset
from Hyperparameters import sep, model_path, epochs, device, log_interval, batch_size_cnn
from Model import Model
from Unet_Denoise import UNet
from torch.nn import functional as F


root_dir = "data" + sep + "generated"
output_dir = "data" + sep + "ae_output"

if __name__ == '__main__':
    # U-Net is applied after the autoencoder was trained
    if not os.path.exists(output_dir):
        print("Need autoencoder output data for training U-Net. Aborting...")
        sys.exit()

    # Create dataset
    transform = UnetDataset.ToTensor()
    dataset = UnetDataset.UnetDataset(root_dir=output_dir, gt_dir=root_dir, transform=transform)

    train_dataset = dataset

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size_cnn, shuffle=True, num_workers=8, drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size_cnn, shuffle=True, num_workers=8, drop_last=True)

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_function = F.l1_loss

    for epoch in range(1, epochs + 1):
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            # Convert tensors to cuda
            input = data['input_mel'].to(device)
            ground_truth = data['gt_mel'].to(device)
            optimizer.zero_grad()
            out = model(input)
            # Main point here: Loss function takes the synth sound as target, so the network learns
            # to map the piano sound to the synth sound!
            loss = loss_function(input, ground_truth)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % log_interval == 0 and epoch % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(input), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item() / len(input)))

        print('====> Epoch: {} Average loss: {:.10f}'.format(
            epoch, train_loss / len(train_loader.dataset)))