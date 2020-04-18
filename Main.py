import os

import librosa
import numpy as np
import torch

import Unet_Denoise
from Util import map_to_range
from Hyperparameters import epochs, device, top_db, sep, sample_rate, n_fft, hop_size
from Model import Model
from SpecVAE import SpecVAECNN
from Unet_Denoise import UNet
from Util import plot_final_mel

ae_path = "model.torch"
unet_path = "unet.torch"

global ae_wrapper
global unet

def pipeline(path):
    # Get the autoencoder output
    ae_output: np.ndarray = ae_wrapper.generate(path, with_return=True)

    # Feed into U-Net
    unet_input = np.expand_dims(ae_output, axis=0)
    unet_input = np.expand_dims(unet_input, axis=0)
    unet_output = Unet_Denoise.generate_sample(unet, unet_input).cpu().numpy()[0, 0]

    db = map_to_range(unet_output, 0, 1, -top_db, 0)
    print("Final output")
    plot_final_mel(db)

    print("Converting spectrogram back to signal...")
    power = librosa.db_to_power(db)
    sig = librosa.feature.inverse.mel_to_audio(power, sample_rate, n_fft, hop_size, n_fft)
    return sig

if __name__ == '__main__':
    if not os.path.exists(ae_path) or not os.path.exists(unet_path):
        print("Autoencoder and U-Net models not present. Please train the networks first.")

    print("Loading autoencoder model...")
    ae = SpecVAECNN(epochs).to(device)
    ae.load_state_dict(torch.load(ae_path))
    ae.eval()
    ae_wrapper = Model(ae, device)
    print("Autoencoder model successfully loaded.")

    print("Loading U-Net model...")
    unet = UNet(n_classes=1, depth=4, padding=True).to(device)
    unet.load_state_dict(torch.load(unet_path))
    unet.eval()
    print("U-Net model loaded.")

    # pipeline("data" + sep + "piano" + sep + "chp_op18.wav")
