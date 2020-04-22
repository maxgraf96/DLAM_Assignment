import librosa
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from Util import map_to_range
from DatasetCreator import create_spectrogram
from Hyperparameters import device, top_db, spec_height, batch_size_autoencoder, input_channels, sample_rate, n_fft, hop_size, \
    unet_width
from Util import plot_mel


class UNet(nn.Module):
    """
    The denoising U-Net model, adapted from https://github.com/n0obcoder/UNet-based-Denoising-Autoencoder-In-PyTorch
    """
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=(1, 1))

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        output = self.last(x)

        return output

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        block.append(nn.Dropout2d(p=0.15)) # edited
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out

def generate_sample(model, spec):
    with torch.no_grad():
        sample = torch.from_numpy(spec).float().to(device)
        mel = model(sample)
        return mel

def generate(model, ae_output, mel_gt, plot_original=True):
    """
    Helper function to forward a sample spectrogram through the network during training.
    :param model: The U-Net PyTorch model object
    :param ae_output: The output spectrogram of the autoencoder
    :param gt_path: The path to the ground truth *.wav file
    :param plot_original: Whether to plot the original (ground truth) spectrogram
    :return: The denoised spectrogram
    """
    # Set mode to evaluation
    model.eval()

    # Create ground truth spectrogram
    if plot_original:
        print("Original ground truth")
        mel_db = map_to_range(mel_gt, 0, 1, -top_db, 0)
        plot_mel(mel_db)

    print("Autoencoder output")
    mel_db = map_to_range(ae_output, 0, 1, -top_db, 0)
    plot_mel(mel_db)

    # Initialise result placeholder
    result = np.zeros((spec_height, unet_width))

    # Prepare data for network
    batch_size = 1  # Use batch size of 1 for single item
    current = np.zeros((batch_size, input_channels, spec_height, unet_width), dtype=np.float32)
    current[0, 0] = ae_output[:, 0 : unet_width]

    # Feed to model
    mel = generate_sample(model, current)
    # Convert back to numpy array
    mel = mel.cpu().numpy()
    result[:, 0  : unet_width] = mel[0]

    # Map back to range [-top_db, 0]
    inv_db_final = map_to_range(result, 0, 1, -top_db, 0)

    # Plot result
    print("U-Net output")
    plot_mel(inv_db_final)

    return result