import os
import sys

import numpy as np
import torch.nn
from torch import optim, nn
from torch.utils.data import DataLoader

import UNetDataset
from Hyperparameters import sep, device, log_interval, epochs_unet, batch_size_unet
from UNet import UNet, generate

# Define L2 loss function for U-Net
loss_fn = nn.MSELoss()

# Set operating directories
root_dir = "data" + sep + "generated"
output_dir = "data" + sep + "ae_output"

# Define path for state dict of trained model (for saving and restoring the model once it's trained)
model_path = "unet.torch"

global model

def loss_function(input, target):
    """
    Calculate MSE loss between input and target
    :param input: The input tensor
    :param target: The target (ground truth) tensor
    :return: The loss value
    """
    l2 = loss_fn(input, target)
    return l2

if __name__ == '__main__':
    # The U-Net is applied after the autoencoder is trained
    if not os.path.exists(output_dir):
        print("Need autoencoder output data in folder " + output_dir + " for training U-Net. Aborting...")
        sys.exit()

    # Load trained model if it exists
    if os.path.exists(model_path):
        print("Unet model exists. Loading model...")
        model = UNet(n_classes=1, depth=4, padding=True).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Unet model loaded.")

    else:
        # Create dataset
        transform = UNetDataset.ToTensor()
        dataset = UNetDataset.UNetDataset(root_dir=output_dir, gt_dir=root_dir, transform=transform)

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

        # Train for 50 epochs
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
                # Get item from dataset and plot spectrogram to show progress of the model
                sample = train_dataset.__getitem__(0)
                input = sample['input_mel'].cpu().numpy()[0]
                ground_truth = sample['gt_mel'].cpu().numpy()[0]
                generate(model, input, ground_truth, plot_original=False)

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

        # Save losses for evaluation
        np.save("train_losses_unet", np.array(train_losses))
        np.save("val_losses_unet", np.array(val_losses))

        # Save model so we don't have to train every time
        torch.save(model.state_dict(), model_path)