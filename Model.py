import librosa
import numpy as np
import torch
from torch.nn import functional as F

from Dataset import map_to_zero_one, map_to_range
from DatasetCreator import load_spectrogram
from Hyperparameters import spec_height, input_channels, sample_rate, n_fft, hop_size, spec_width, log_interval, n_mels
from SpecVAE import plot_final, plot_final_mel


class Model:
    def __init__(self, model, device, batch_size, log_interval):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.prevloss = -1
        self.sameloss = False
        self.samelosscounter = 0

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='mean')

        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise KLD by batch size and size of spectrogram
        KLD /= self.batch_size

        return BCE + KLD

    def train(self, epoch, train_loader, optimizer):
        # Set current epoch in model (used for plotting)
        self.model.set_epoch(epoch)
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(self.device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_function(recon_batch.view(self.batch_size, input_channels, spec_height, spec_width), data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % log_interval == 0 and epoch % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.10f}'.format(
            epoch, train_loss / len(train_loader.dataset)))

        current_avg_loss = train_loss / len(train_loader.dataset)
        self.sameloss = np.isclose(current_avg_loss, self.samelosscounter, rtol=0, atol=0.00001)
        if self.sameloss:
            self.samelosscounter += 1
        else:
            self.samelosscounter = 0

        # Update previous loss
        self.prevloss = current_avg_loss

        # Early stop if loss doesn't change for 3 epochs
        if self.samelosscounter == 3:
            print("Loss is the same since last three epochs. Early stopping")
            return True

        # if epoch % 30 == 0:
        # Plot snapshot of current representation
        self.generate("data/piano/chpn_op7_1.wav", with_return=False)

    def test(self, test_loader, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.loss_function(recon_batch.view(self.batch_size, input_channels, spec_height, spec_height), data, mu, logvar).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    # comparison = torch.cat([data[:n],
                    #                         recon_batch.view(self.batch_size, 1, SPEC_DIMS_W, SPEC_DIMS_H)[:n]])
                    # save_image(comparison.cpu(),
                    #            'results/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def generate_sample(self, spec):
        with torch.no_grad():
            sample = torch.from_numpy(spec).to(self.device)
            sample, mu, logvar = self.model(sample)
            return sample

    def generate_latent_sample(self, spec):
        with torch.no_grad():
            sample = torch.from_numpy(spec).to(self.device)
            sample, mu, logvar = self.model.bottleneck(self.model.encoder(sample))
            return sample

    def generate(self, path, with_return=True):
        self.model.eval()

        spec = load_spectrogram(path)
        if with_return:
            print("Original")
            inv_db = map_to_range(spec, 0, 1, -80, 0)
            inv_mag = librosa.db_to_amplitude(inv_db)
            plot_final(inv_mag)
            # plot_final_mel(spec)

        result = np.zeros(spec.shape)
        width = spec_width * self.batch_size
        for frame in range(0, spec.shape[1], width):
            if frame + width > spec.shape[1]:
                break

            # Fill batches
            current = np.zeros((self.batch_size, input_channels, spec_height, spec_width), dtype=np.float32)
            for batch in range(self.batch_size):
                start = frame + batch * spec_width
                end = start + spec_width
                current_batch = spec[:, start : end]
                current[batch, 0] = current_batch

            # Calculate for whole batch
            result_frames = self.generate_sample(current)
            result_frames = result_frames.cpu().numpy()

            for batch in range(self.batch_size):
                start = frame + batch * spec_width
                end = start + spec_width
                result[:, start : end] = result_frames[batch][0]

        # Invert result
        # STFT
        inv_db = map_to_range(result, 0, 1, -80, 0)
        inv_mag = librosa.db_to_amplitude(inv_db)

        # Mel
        # inv_mag = result

        # SHOW
        plot_final(inv_mag)
        # plot_final_mel(inv_mag)

        if with_return:
            sig_result = librosa.feature.inverse.griffinlim(inv_mag, n_iter=32, hop_length=hop_size, win_length=n_fft)

            # When using mel specrogram
            # sig_result = librosa.feature.inverse.mel_to_audio(inv_mag, sample_rate, n_fft, hop_size, n_fft)

            # orig_griffin_lim = librosa.feature.inverse.griffinlim(pow, n_iter=32, hop_length=hop_size, win_length=n_fft-2)
            # librosa.output.write_wav("orig_griffin_lim.wav", orig_griffin_lim, sample_rate)

            return sig_result