import os

import torch

# OS specific path separator
sep = os.path.sep

# Generated directory
gen_dir = 'data' + os.path.sep + 'generated'

# Hyperparameters for torch
model_path = "spec_vae.model"
cuda = torch.cuda.is_available()
batch_size_cnn = 4
batch_size_ann = 1
epochs = 10
seed = 1
log_interval = 10

# STFT params
sample_rate = 22050
n_fft = 256
hop_size = 2048