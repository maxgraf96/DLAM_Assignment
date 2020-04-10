import os

import torch

# OS specific path separator
sep = os.path.sep

# Generated directory
gen_dir = 'data' + os.path.sep + 'generated'

# STFT params
sample_rate = 22050
n_fft = 512
hop_size = 256
limit_s = 30
spec_dim = int(n_fft / 2) + 1

# Hyperparameters for torch
model_path = "spec_vae.model"
cuda = torch.cuda.is_available()
batch_size_cnn = 1
batch_size_ann = 10
epochs = 20
seed = 1
log_interval = 5