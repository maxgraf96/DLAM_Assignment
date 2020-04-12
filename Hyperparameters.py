import os

import torch

# OS specific path separator
sep = os.path.sep

# Generated directory
gen_dir = 'data' + os.path.sep + 'generated'

# STFT params
sample_rate = 22050
n_fft = 2048
hop_size = 512
limit_s = 30
width_seconds = 0.2
spec_width = int(width_seconds / (hop_size / sample_rate))
spec_height = int(n_fft / 2) + 1
# When using mel spectrogram
n_mels = 512
# spec_height = n_mels

# Hyperparameters for torch
cuda = torch.cuda.is_available()
batch_size_cnn = 160
epochs = 12
log_interval = 60

# CNN params
input_channels = 1