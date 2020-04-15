import os
import time

import librosa
import torch
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.nn import functional as F


import Dataset
import DatasetCreator
from Hyperparameters import batch_size_cnn, epochs, log_interval, sep, sample_rate, device, batch_size_gru, spec_width, \
    spec_height, log_epochs
# Initialise dataset
from Model import Model
from SpecGRU import SpecGRU, generate
from SpecVAE import SpecVAECNN

model_path = "gru.torch"

# Pytorch init
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

# device = "cpu"
# kwargs = {}

# Initialise dataset (create spectrograms if not exist)
DatasetCreator.initialise_dataset()
root_dir = "data" + sep + "generated"# + sep + "chpn_op7_1"

global main
global dataset
global batch_size

def loss_function(input, target):
    # MSE = F.mse_loss(input, target)
    MAE = F.l1_loss(input, target)
    return MAE

if __name__ == '__main__':
    # Create dataset
    transform = Dataset.ToTensor()
    dataset = Dataset.AssignmentDataset(root_dir=root_dir, transform=transform)

    # Set batch size
    # batch_size = batch_size_ann
    batch_size = batch_size_gru

    if os.path.exists(model_path):
        model = SpecGRU(epochs, dataset.length).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

    else:
        model = SpecGRU(epochs, dataset.length).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.95, 0.999))

        # Split into training and test sets
        train_size = int(len(dataset) * 0.99)
        test_size = len(dataset) - train_size
        # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_dataset = dataset
        l = len(train_dataset)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

        # generate(model, "data" + sep + "piano" + sep + "chpn_op7_1.wav", with_return=True)

        model.train()
        epoch_times = []
        # Start training loop
        for epoch in range(1, epochs + 1):
            start_time = time.clock()
            h = model.init_hidden(batch_size)
            avg_loss = 0.
            counter = 0
            for batch_idx, data in enumerate(train_loader):
                # Convert tensors to cuda
                data['sound'] = data['sound'].to(device)
                data['sound_synth'] = data['sound_synth'].to(device)
                piano = data['sound']
                # Get matching synth files
                synth = data['sound_synth']

                counter += 1
                h = h.data
                model.zero_grad()

                out, h = model(piano, h)
                loss = loss_function(out, synth)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                if counter % 24 == 0:
                    print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                               len(train_loader),
                                                                                               avg_loss / counter))

            current_time = time.clock()
            print("Epoch {}/{} Done, Total Loss: {}".format(epoch, epochs, avg_loss / len(train_loader)))
            print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
            print()
            epoch_times.append(current_time - start_time)

            if epoch % log_epochs == 0:
                generate(model, "data/piano/chpn_op7_1.wav", with_return=False)

        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    # Save model so we don't have to train every time
    # torch.save(model.state_dict(), model_path)

    # gen = generate(model, "data" + sep + "piano" + sep + "chpn_op7_1.wav", with_return=True)
    # gen = librosa.util.normalize(gen)

    # Display (only works on IPython notebooks)
    # librosa.output.write_wav("output.wav", gen, sample_rate)
