import os

import librosa
import numpy as np
import torch

from AEModel import AEModel
from Autoencoder import Autoencoder
from Hyperparameters import device, top_db, sample_rate, n_fft, hop_size, sep
from UNet import UNet, generate_sample
from Util import map_to_range
from Util import plot_mel

ae_path = "ae.torch"
unet_path = "unet.torch"

global ae_wrapper
global unet

def pipeline(path):
    # Get the autoencoder output
    ae_output: np.ndarray = ae_wrapper.generate(path, plot_original=True)

    # Feed into U-Net
    unet_input = np.expand_dims(ae_output, axis=0)
    unet_input = np.expand_dims(unet_input, axis=0)
    unet_output = generate_sample(unet, unet_input).cpu().numpy()[0, 0]

    db = map_to_range(unet_output, 0, 1, -top_db, 0)
    print("Final output")
    plot_mel(db)

    print("Converting spectrogram back to signal...")
    power = librosa.db_to_power(db)
    sig = librosa.feature.inverse.mel_to_audio(power, sample_rate, n_fft, hop_size, n_fft)
    return sig

if __name__ == '__main__':
    if not os.path.exists(ae_path) or not os.path.exists(unet_path):
        print("Autoencoder and U-Net models not present. Please train the networks first.")

    print("Loading autoencoder model...")
    ae = Autoencoder().to(device)
    ae.load_state_dict(torch.load(ae_path))
    ae.eval()
    ae_wrapper = AEModel(ae, device)
    print("Autoencoder model successfully loaded.")

    print("Loading U-Net model...")
    unet = UNet(n_classes=1, depth=4, padding=True).to(device)
    unet.load_state_dict(torch.load(unet_path))
    unet.eval()
    print("U-Net model loaded.")

    # Test run a *.wav file through both models
    # This plots the interim results and returns the final output as a numpy array containing
    # the signal's time-series data
    output_sig = pipeline("data" + sep + "piano" + sep + "chp_op18.wav")