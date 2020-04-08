import numpy as np
import matplotlib.pyplot as plt
from Hyperparameters import batch_size_cnn

from Freesound.FS_FreesoundDataset import SPEC_DIMS_W, SPEC_DIMS_H


def plot_latent_representations_cnn(original, data, is_plot_original=False):
    """
    Plot representation of latent space
    :param original: The original data
    :param data: The latent data
    :param is_plot_original: True if original data should be plotted
    :return: None
    """
    # Plot original input data
    if is_plot_original:
        num_items = original.shape[0]
        fig, ax = plt.subplots(nrows=2, ncols=int(num_items / 2))
        i = 0
        for row in ax:
            for col in row:
                spec = original[i, 0, :, :].cpu().detach().numpy()
                col.pcolormesh(spec)
                i = i + 1
        plt.title("Original")
        #plt.show()

    # Plot latent representation
    data = data.view(batch_size_cnn, 4, SPEC_DIMS_W, SPEC_DIMS_H)
    num_items = data.shape[0]
    fig, ax = plt.subplots(nrows=2, ncols=int(num_items / 2))
    i = 0
    for row in ax:
        for col in row:
            spec = data[i, 0, :, :].cpu().detach().numpy()
            col.pcolormesh(spec)
            i = i + 1
    fig.suptitle("Latent representation")
    plt.show()

def plot_latent_representations_ann(original, data, is_plot_original=False):
    """
    Plot representation of latent space
    :param original: The original data
    :param data: The latent data
    :param is_plot_original: True if original data should be plotted
    :return: None
    """
    # Plot latent representation
    data = data.cpu().detach().numpy()
    x = np.arange(data.shape[0])
    plt.plot(x, data)
    plt.title("Latent representation")
    plt.show()

# wavs = Path("data/Arturia MicroBrute Saw Oscillator").rglob("*.wav")
# for wav in wavs:
#     plot_mel_spectrogram(wav)