import librosa
import librosa.display
import torch
from torch import nn
import numpy as np

import DatasetCreator
from Dataset import map_to_range
from Hyperparameters import batch_size_gru, input_channels, spec_height, spec_width, device, hop_size, n_fft, \
    sample_rate
from SpecVAE import plot_final


class SpecGRU(nn.Module):
    def __init__(self, epochs, dataset_length):
        nn.Module.__init__(self)
        self.dataset_length = dataset_length

        self.n_features = spec_height
        self.seq_len = spec_width
        self.num_layers = 2
        self.hidden_dim = 256

        # Encoder
        self.encoder = nn.GRU(self.n_features, self.hidden_dim, self.num_layers, batch_first=True)

        test_encode = self.encoder(torch.randn(batch_size_gru, self.seq_len, self.n_features))
        output = test_encode[0]
        H_DIMS = output.flatten().shape[0]

        self.bottleneck = nn.Sequential(
            nn.Linear(H_DIMS, 128),
            nn.Linear(128, H_DIMS)
        )

        # Decoder
        self.decoder = nn.GRU(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fcout = nn.Linear(self.hidden_dim, self.n_features)

    def forward(self, x, h):
        # Test print
        # import matplotlib.pyplot as plt
        # plt.figure()
        # test = x.cpu().detach().numpy()[0]
        # librosa.display.specshow(test)
        # plt.show()
        out, h = self.encoder(x, h)
        # bottleneck = self.bottleneck(out.flatten())
        # bottleneck = bottleneck.reshape(batch_size_gru, spec_width, self.hidden_dim)
        out, h = self.decoder(out, h)
        result = self.fcout(out)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # test = result.cpu().detach().numpy()[0]
        # librosa.display.specshow(test)
        # plt.show()
        # a = 2
        return result, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()
        return hidden

def generate_sample(model, spec):
    with torch.no_grad():
        sample = torch.from_numpy(spec).float().to(device)
        h = model.init_hidden(batch_size_gru)
        sample, h = model.forward(sample, h)
        return sample

def generate(model, path, with_return=False):
    spec = DatasetCreator.create_spectrogram(path)

    if with_return:
        print("Original")
        orig_inv_db = map_to_range(spec, 0, 1, -120, 0)
        orig_inv_mag = librosa.db_to_amplitude(orig_inv_db)
        plot_final(orig_inv_mag)
        # Output orig griffin lim result
        orig_gl_result = librosa.feature.inverse.griffinlim(orig_inv_mag, n_iter=100, hop_length=hop_size, win_length=n_fft)
        gen = librosa.util.normalize(orig_gl_result)
        librosa.output.write_wav("output_gl.wav", gen, sample_rate)

    result = np.zeros(spec.shape)
    width = spec_width * batch_size_gru
    for frame in range(0, spec.shape[1], width):
        if frame + width > spec.shape[1]:
            break

        # Fill batches
        current = np.zeros((batch_size_gru, spec_height, spec_width), dtype=np.float32)
        for batch in range(batch_size_gru):
            start = frame + batch * spec_width
            end = start + spec_width
            current_batch = spec[:, start: end]
            current[batch] = current_batch

        endgschicht = np.zeros((batch_size_gru, spec_height, spec_width))
        # Calculate for whole batch
        current_original = current
        input = np.zeros((batch_size_gru, spec_width, spec_height))
        input = np.swapaxes(current_original, 1, 2).copy()
        result_frames = generate_sample(model, input)
        result_frames = result_frames.cpu().numpy()
        endgschicht = np.swapaxes(result_frames.copy(), 1, 2)


        # if frame > 0:
        #     import matplotlib.pyplot as plt
        #     plt.figure()
        #     # test = result_frames[0].copy()
        #     # test = np.swapaxes(np.flip(test, axis=0).copy(), 0, 1).copy()
        #
        #     librosa.display.specshow(endgschicht[0])
        #     plt.show()
        #     a = 2

        for batch in range(batch_size_gru):
            start = frame + batch * spec_width
            end = start + spec_width
            result[:, start: end] = endgschicht[batch]

    # Invert result
    # STFT
    inv_db = map_to_range(result, 0, 1, -120, 0)
    inv_mag = librosa.db_to_amplitude(inv_db)

    # Mel
    # inv_mag = result

    # SHOW
    plot_final(inv_mag)
    # plot_final_mel(inv_mag)

    if with_return:

        sig_result = librosa.feature.inverse.griffinlim(inv_mag, n_iter=100, hop_length=hop_size, win_length=n_fft)
        return sig_result

