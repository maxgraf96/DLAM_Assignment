import numpy as np
import torch
from torch.nn import functional as F

from DatasetCreator import create_spectrogram
from Hyperparameters import spec_height, input_channels, spec_width, log_epochs, \
    batch_size_autoencoder, sep, top_db
from Util import map_to_range, plot_mel


class AEModel:
    """
    Wrapper for the Autoencoder module to simplify training and sample generation.
    """
    def __init__(self, model, device):
        """
        Initialise
        :param model: The Autoencoder module
        :param device: The PyTorch device (cpu or cuda)
        """
        self.model = model
        self.device = device
        self.prevloss = -1
        self.sameloss = False
        self.samelosscounter = 0

    def loss_function(self, input, target):
        """
        The mean absolute error (L1) loss function is used for training and evaluation.
        :param input: The incoming tensor
        :param target: The target tensr
        :return: The loss value
        """
        MAE = F.l1_loss(input, target)
        return MAE

    def train(self, current_epoch, train_loader, optimizer):
        """
        Train the Autoencoder model for one epoch using the given training data and optimizer.
        NB: Early stopping is implemented but not currently in use, as the obtained results were not satisfying.

        :param current_epoch: The current epoch in the training process
        :param train_loader: The training data loader
        :param optimizer: The PyTorch optimizer
        :return: The current loss averaged over all batches, whether the training process should be terminated early
        """
        # Set to training mode to enable autograd
        self.model.train()
        # Placeholder for current loss
        train_loss = 0

        # Iterate over all data points
        for batch_idx, data in enumerate(train_loader):
            # Convert tensors to cuda
            piano = data['piano_mel'].to(self.device)
            synth = data['synth_mel'].to(self.device)
            optimizer.zero_grad()
            mel, mu, logvar = self.model(piano)
            # Main point here: Loss function takes the synth sound as target, so the network learns
            # to map the piano sound to the synth sound.
            loss = self.loss_function(mel, synth)
            # Backpropagate
            loss.backward()
            # Add loss of current batch to cumulative
            train_loss += loss.item()
            optimizer.step()

        # Print training progress info
        print('====> Epoch: {} Average loss: {:.10f}'.format(
            current_epoch, train_loss / len(train_loader.dataset)))

        # Check whether early stopping should apply - Not in use right now
        current_avg_loss = train_loss / len(train_loader.dataset)
        self.sameloss = np.isclose(current_avg_loss, self.samelosscounter, atol=1e-3)
        if self.sameloss:
            self.samelosscounter += 1
            print("Same loss counter +1. Is now: " + str(self.samelosscounter))
        else:
            if self.samelosscounter > 0:
                print("Same loss counter reset")
            self.samelosscounter = 0

        # Update previous loss
        self.prevloss = current_avg_loss

        # Early stop if loss doesn't change for 3 epochs
        if self.samelosscounter == 10:
            print("Loss is the same since last three epochs. Early stopping")
            return current_avg_loss, True

        if current_epoch % log_epochs == 0:
            # Plot snapshot of current representation
            self.generate("data" + sep + "piano" + sep + "chpn_op7_1.wav", plot_original=False)

        return current_avg_loss, False

    def validate(self, val_loader):
        """
        Validate the current state of the model using the validation set
        :param val_loader: The PyTorch dataloader containg the validation set data
        :return: The validation loss averaged over the items in the validation set
        """
        # Set to evaluation mode
        self.model.eval()
        val_loss = 0
        # Disable autograd
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                # Convert tensors to cuda
                piano = data['piano_mel'].to(self.device)
                synth = data['synth_mel'].to(self.device)
                mel, mu, logvar = self.model(piano)
                loss = self.loss_function(mel, synth)
                val_loss += loss.item()

        val_loss /= len(val_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(val_loss))
        return val_loss

    def generate_sample(self, spec):
        """
        Helper function to forward a sample spectrogram through the network.
        Used for data generation for U-Net dataset and visualising the progress of the model during training.
        :param spec: The input spectrogram
        :return: The model's output
        """
        with torch.no_grad():
            sample = torch.from_numpy(spec).to(self.device)
            mel, mu, logvar = self.model.forward_sample(sample)
            return mel

    def generate(self, path, plot_original=True):
        """
        Routine for generating an output from a given *.wav input file and visualising the model's operation
        :param path: The path to the input *.wav file
        :param plot_original: If set to true, will plot the input spectrogram and attempt to plot the synth spectrogram
        of the input (only works if it can be located, i.e. if the input path points to
        the training/validation data).
        :return: The Mel spectrogram output from the autoencoder
        """
        # Set to evaluation mode
        self.model.eval()

        # Generate the a spectrogram of "limit_s" second length from the input *.wav file.
        mel = create_spectrogram(path)

        if plot_original:
            print("Original")
            # All data in the system are normalised to a range between 0 and -"top_db"
            # in order to be able to better visualise the model's operations
            mel_db = map_to_range(mel, 0, 1, -top_db, 0)
            plot_mel(mel_db)

            try:
                # Get synth version
                print("Original synth")
                synth_path_s = str(path).split(sep)
                # Replace 'piano' with 'synth'
                synth_path_s[-2] = 'synth'
                synth_path = sep.join(synth_path_s)
                mel_synth = create_spectrogram(synth_path)[0]
                mel_synth_db = map_to_range(mel_synth, 0, 1, -top_db, 0)
                plot_mel(mel_synth_db)
            except:
                print("No synth file found for input...")

        # Initialise result placeholder
        result = np.zeros(mel.shape)

        # Fill batches. Currently the batch size for the autoencoder is set to one.
        # Future releases could handle greater batch sizes
        current = np.zeros((batch_size_autoencoder, input_channels, spec_height, spec_width), dtype=np.float32)
        current[0, 0] = mel[:, 0 : spec_width]

        # Put data through autoencoder
        mel = self.generate_sample(current)
        # Convert PyTorch tensor back to numpy
        mel = mel.cpu().numpy()

        # Extract two-dimensional array
        result[:, 0  : spec_width] = mel[0]

        # Map to range and plot result
        inv_db_final = map_to_range(result, 0, 1, -top_db, 0)
        plot_mel(inv_db_final)

        return result