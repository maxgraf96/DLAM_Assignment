import os
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

import AEDataset
import DatasetCreator
from Autoencoder import Autoencoder
from Hyperparameters import batch_size_autoencoder, epochs, sep, device, model_path

from AEModel import AEModel

# Pytorch init
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

# Initialise dataset (create spectrograms if not exist)
DatasetCreator.initialise_dataset()
root_dir = "data" + sep + "generated"
output_dir = "data" + sep + "ae_output"

global main
global dataset
global spec_width, spec_height

def create_unet_dataset():
    """
    Create the data for the U-Net dataset once the autoencoder model is trained
    :return: None
    """
    # Iterate over piano songs and save generated spectrograms for use in U-Net
    print("Generating autoencoder outputs for use in U-Net...")
    piano_wavs = Path("data" + sep + "piano").rglob("*.wav")
    if not os.path.exists(output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    for wav in piano_wavs:
        path = str(wav)
        print("Generating autoencoder output for " + path)
        gen = main.generate(path, plot_original=False)
        filename = path.split(sep)[-1][:-4]
        # Save to *.npy file
        np.save(output_dir + sep + filename + "_output", gen)

if __name__ == '__main__':
    # Create dataset
    transform = AEDataset.ToTensor()
    dataset = AEDataset.AssignmentDataset(root_dir=root_dir, transform=transform)

    if os.path.exists(model_path):
        print("Model exists. Loading model...")
        model = Autoencoder().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        main = AEModel(model, device)

        if not os.path.exists(output_dir):
            create_unet_dataset()

    else:
        # Split into training and validation sets
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size_autoencoder, shuffle=True, num_workers=8, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_autoencoder, shuffle=True, num_workers=8, drop_last=True)

        model = Autoencoder().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3 * 2)
        main = AEModel(model, device)
        train_losses = []
        val_losses = []

        for epoch in range(1, epochs + 1):
            loss, is_early_stop = main.train(epoch, train_loader, optimizer)
            if is_early_stop:
                print("Early stopped after " + str(epochs) + " epochs.")
                break
            # Validate
            val_loss = main.validate(val_loader)

            train_losses.append(loss)
            val_losses.append(val_loss)

        # Save losses for evaluation
        np.save("train_losses_ae", np.array(train_losses))
        np.save("val_losses_ae", np.array(val_losses))

        # Save model so we don't have to train every time
        torch.save(model.state_dict(), model_path)

        # Create dataset for U-Net
        create_unet_dataset()


    # Generate something
    # gen = main.generate("data" + sep + "piano" + sep + "chpn_op7_1.wav")
    # gen = librosa.util.normalize(gen)

    # Display (only works on IPython notebooks)
    # librosa.output.write_wav("output.wav", gen, sample_rate)
