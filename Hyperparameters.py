import os

import torch

# OS specific path separator
sep = os.path.sep

# Generated directory
gen_dir = 'data' + sep + 'generated'

# STFT params
sample_rate = 22050
n_fft = 1024
hop_size = 256
limit_s = 30
width_seconds = 1
spec_width = int(width_seconds / (hop_size / sample_rate))
spec_height = int(n_fft / 2) + 1
# When using mel spectrogram
n_mels = 512
# spec_height = n_mels

# Hyperparameters for torch
cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size_cnn = 30
epochs = 100
log_epochs = 20
log_interval = 60

# CNN params
input_channels = 1

# GRU params
batch_size_gru = 30