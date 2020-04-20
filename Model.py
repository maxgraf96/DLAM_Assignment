import librosa
import numpy as np
import torch
from torch.nn import functional as F

import DatasetCreator
from Util import map_to_range
from DatasetCreator import create_spectrogram
from Hyperparameters import spec_height, input_channels, n_fft, hop_size, spec_width, log_interval, log_epochs, \
    sample_rate, batch_size_cnn, sep, top_db
from Util import plot_final, plot_final_mel


class Model:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.prevloss = -1
        self.sameloss = False
        self.samelosscounter = 0

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x):

        MAE = F.l1_loss(recon_x, x)

        # return BCE + KLD
        return MAE # + MAE_STFT

    def train(self, epoch, train_loader, optimizer):
        # Set current epoch in model (used for plotting)
        self.model.set_epoch(epoch)
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            # Convert tensors to cuda
            piano = data['piano_mel'].to(self.device)
            synth = data['synth_mel'].to(self.device)
            optimizer.zero_grad()
            mel, mu, logvar = self.model(piano)
            # Main point here: Loss function takes the synth sound as target, so the network learns
            # to map the piano sound to the synth sound!
            loss = self.loss_function(mel, synth)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % log_interval == 0 and epoch % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(piano), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item() / len(piano)))

        print('====> Epoch: {} Average loss: {:.10f}'.format(
            epoch, train_loss / len(train_loader.dataset)))

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

        if epoch % log_epochs == 0:
            # Plot snapshot of current representation
            self.generate("data" + sep + "piano" + sep + "chpn_op7_1.wav", with_return=False)

        return current_avg_loss, False

    def validate(self, test_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # Convert tensors to cuda
                piano = data['piano_mel'].to(self.device)
                synth = data['synth_mel'].to(self.device)
                mel, mu, logvar = self.model(piano)
                loss = self.loss_function(mel, synth)
                val_loss += loss.item()

        val_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(val_loss))
        return val_loss

    def generate_sample(self, spec):
        with torch.no_grad():
            sample = torch.from_numpy(spec).to(self.device)
            mel, mu, logvar = self.model.forward_sample(sample)
            return mel

    def generate(self, path, with_return=True):
        self.model.eval()

        mel = create_spectrogram(path)
        if with_return:
            print("Original")
            inv_db = map_to_range(mel, 0, 1, -top_db, 0)
            # inv_pow = librosa.db_to_amplitude(inv_db)
            # plot_final(inv_mag)
            plot_final_mel(inv_db)

            try:
                # Get synth version
                print("Original synth")
                synth_path_s = str(path).split(sep)
                # Replace 'piano' with 'synth'
                synth_path_s[-2] = 'synth'
                synth_path = sep.join(synth_path_s)
                mel_synth = create_spectrogram(synth_path)[0]
                synth_inv_db = map_to_range(mel_synth, 0, 1, -top_db, 0)
                plot_final_mel(synth_inv_db)
            except:
                print("No synth file found for input")

        result = np.zeros(mel.shape)
        # Fill batches
        current = np.zeros((batch_size_cnn, input_channels, spec_height, spec_width), dtype=np.float32)
        current[0, 0] = mel[:, 0 : spec_width]

        # Calculate for whole batch
        mel = self.generate_sample(current)
        mel = mel.cpu().numpy()

        result[:, 0  : spec_width] = mel[0]

        # Mel
        inv_db_final = map_to_range(result, 0, 1, -top_db, 0)
        inv_pow = librosa.db_to_power(inv_db_final)

        # SHOW
        # plot_final(inv_mag)
        plot_final_mel(inv_db_final)

        # if with_return:
            # sig_result = librosa.feature.inverse.griffinlim(inv_mag, n_iter=100, hop_length=hop_size, win_length=n_fft)
            # When using mel specrogram
            # sig_result = librosa.feature.inverse.mel_to_audio(inv_pow, sample_rate, n_fft, hop_size, n_fft)
            # return sig_result

        return result