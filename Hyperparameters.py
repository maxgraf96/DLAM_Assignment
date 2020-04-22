import os

import torch

# OS specific path separator
sep = os.path.sep

# Generated directory
gen_dir = 'data' + sep + 'generated'
model_path = "ae.torch"


# STFT params
sample_rate = 22050
n_fft = 1024
hop_size = 256
limit_s = 30
width_seconds = 30
spec_width = int(width_seconds / (hop_size / sample_rate))
stft_height = int(n_fft / 2) + 1
# When using mel spectrogram
n_mels = 128
spec_height = n_mels

# Hyperparameters for torch
cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shared params
log_epochs = 20
log_interval = 60

# Autoencoder params
epochs = 600
input_channels = 1
batch_size_autoencoder = 1

# U-net params
epochs_unet = 50
batch_size_unet = 8
unet_width = 2576

# dB flooring for amp/pow to dB conversions
top_db = 120