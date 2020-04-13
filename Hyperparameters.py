import os

import torch

# OS specific path separator
sep = os.path.sep

# Generated directory
gen_dir = 'data' + os.path.sep + 'generated'

# STFT params
sample_rate = 22050
n_fft = 1024
hop_size = 256
limit_s = 10
width_seconds = 1
spec_width = int(width_seconds / (hop_size / sample_rate))
spec_height = int(n_fft / 2) + 1
# When using mel spectrogram
n_mels = 512
# spec_height = n_mels

# Hyperparameters for torch
cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size_cnn = 10
epochs = 300
log_interval = 60

# CNN params
input_channels = 1