import os
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

import Dataset
import DatasetCreator
from Hyperparameters import batch_size_cnn, epochs, log_interval, sep, device, model_path
# Initialise dataset
from Model import Model
from SpecVAE import SpecVAECNN

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
    # Iterate over piano songs and save generated spectrograms for use in U-Net
    print("Generating autoencoder outputs for use in U-Net...")
    piano_wavs = Path("data" + sep + "piano").rglob("*.wav")
    if not os.path.exists(output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    for wav in piano_wavs:
        path = str(wav)
        print("Generating autoencoder output for " + path)
        gen = main.generate(path, with_return=False)
        filename = path.split(sep)[-1][:-4]
        np.save(output_dir + sep + filename + "_output", gen)

if __name__ == '__main__':
    # Create dataset
    transform = Dataset.ToTensor()
    dataset = Dataset.AssignmentDataset(root_dir=root_dir, transform=transform)

    if os.path.exists(model_path):
        print("Model exists. Loading model...")
        model = SpecVAECNN(epochs, dataset.length).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        main = Model(model, device)

        if not os.path.exists(output_dir):
            create_unet_dataset()

    else:
        # Split into training and test sets
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size_cnn, shuffle=True, num_workers=8, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_cnn, shuffle=True, num_workers=8, drop_last=True)

        model = SpecVAECNN(epochs, dataset.length).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3 * 2)
        main = Model(model, device)
        train_losses = []
        test_losses = []

        for epoch in range(1, epochs + 1):
            loss, is_early_stop = main.train(epoch, train_loader, optimizer)
            if is_early_stop:
                print("Early stopped after " + str(epochs) + " epochs.")
                break
            # Test
            test_loss = main.test(test_loader, epoch)

            train_losses.append(loss)
            test_losses.append(test_loss)

        # Save losses
        np.save("train_losses_ae", np.array(train_losses))
        np.save("test_losses_ae", np.array(test_losses))

        # Save model so we don't have to train every time
        torch.save(model.state_dict(), model_path)

        # Create dataset for U-Net
        create_unet_dataset()


    # Generate something
    # gen = main.generate("data" + sep + "piano" + sep + "chpn_op7_1.wav")
    # gen = librosa.util.normalize(gen)

    # Display (only works on IPython notebooks)
    # librosa.output.write_wav("output.wav", gen, sample_rate)
