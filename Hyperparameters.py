import torch

# Hyperparameters
model_path = "spec_vae.model"
cuda = torch.cuda.is_available()
batch_size_cnn = 4
batch_size_ann = 1
epochs = 10
seed = 1
log_interval = 10