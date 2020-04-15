import os

import torch
from torch import optim
from torch.utils.data import DataLoader

import Dataset
import DatasetCreator
from Hyperparameters import batch_size_cnn, epochs, log_interval, sep, device
# Initialise dataset
from Model import Model
from SpecVAE import SpecVAECNN

model_path = "model.torch"

# Pytorch init
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

# device = "cpu"
# kwargs = {}

# Initialise dataset (create spectrograms if not exist)
DatasetCreator.initialise_dataset()
root_dir = "data" + sep + "generated" #+ sep + "chpn_op7_1"

global main
global dataset
global spec_width, spec_height

if __name__ == '__main__':
    # Create dataset
    transform = Dataset.ToTensor()
    dataset = Dataset.AssignmentDataset(root_dir=root_dir, transform=transform)

    # Set batch size
    # batch_size = batch_size_ann
    # Create and initialise VAE
    spec_width, spec_height = dataset.get_spec_dims()

    if os.path.exists(model_path):
        print("Model exists. Loading model...")
        model = SpecVAECNN(epochs, dataset.length).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        main = Model(model, device, log_interval)
        main.generate("data" + sep + "piano" + sep + "chpn_op7_1.wav", with_return=True)

    else:
        # Split into training and test sets
        # train_size = int(len(dataset) * 0.8)
        train_dataset = dataset
        # test_size = len(dataset) - train_size
        # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        # train_dataset = torch.utils.data.Subset(dataset, np.arange(train_size))
        # test_dataset = torch.utils.data.Subset(dataset, np.arange(train_size, dataset.length))

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size_cnn, shuffle=True, num_workers=8, drop_last=True)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

        model = SpecVAECNN(epochs, dataset.length).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3 * 2)
        main = Model(model, device, log_interval)

        for epoch in range(1, epochs + 1):
            is_early_stop = main.train(epoch, train_loader, optimizer)
            if is_early_stop:
                print("Early stopped after " + str(epochs) + " epochs.")
                break
            # main.test(test_loader, epoch)

        # Save model so we don't have to train every time
        # torch.save(model.state_dict(), model_path)

    # Generate something
    # gen = main.generate("data" + sep + "piano" + sep + "chpn_op7_1.wav")
    # gen = librosa.util.normalize(gen)

    # Display (only works on IPython notebooks)
    # librosa.output.write_wav("output.wav", gen, sample_rate)
