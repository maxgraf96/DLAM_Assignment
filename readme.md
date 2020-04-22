# End-to-End Raw Audio Instrument Resynthesis

This repository contains the codebase for an end-to-end raw audio resynthesis system 
written in Python 3. The project was done for the course 
_ECS7013P - Deep Learning For Audio And Music - 2019/20_ at Queen Mary, University of London.

The aim of the project is to resynthesise recordings of one instrument to another instrument's timbre.
The system is currently configured to synthesise electronic piano sounds from piano sounds.
However, by using different datasets, the technique can theoretically be applied to any two instrument types.

A detailed description of the system is given in the `Report.pdf` file.

## Quick Setup:
1. Install [pip](https://pip.pypa.io/en/stable/)
2. Run `pip install -r requirements.txt` to install the required libraries
3. If you want to use the pretrained models: 
    - Use [Main_Notebook.ipynb](Main_Notebook.ipynb) for testing both models combined
    - Use [AEMain_Notebook.ipynb](AEMain_Notebook.ipynb) for testing only the Autoencoder
    - Use [UNetMain_Notebook.ipynb](UNetMain_Notebook.ipynb) for testing only the U-Net model 
4. If you want to train the models yourself:
    - Delete (or backup) the pretrained model files `ae.torch` and `unet.torch`
    - **Option 1**: Synthesise _*.wav_ files from MIDI data using the [MIDI2Audio.py](MIDI2Audio.py) script. 
    **NB:** For this, [FluidSynth](http://www.fluidsynth.org/) needs to be installed on your system and
     you must provide training+validation and test _*.mid_ files in the [data/MIDI](data/MIDI) 
     and [data/MIDI_test](data/MIDI_test) directories respectively
    - **Option 2**: Use a separate dataset of pre-existing recordings
    In this case you can simply provide the piano and synth _*.wav_ files in the 
    [data/piano](data/piano) and [data/synth](data/synth) directories
    - _Optional: Adjust the hyperparameters for the STFT calculations and the models 
    in [Hyperparameters.py](Hyperparameters.py)_ 
    - Execute `python AEMain.py` to generate the spectrogram representations 
    and train the autoencoder. This will fill the [data/generated](data/generated) directory 
    with _*.npy_ files for both piano and synth spectrograms
    - Execute `python UNetMain.py` to train the U-Net
    - Execute `python Main.py` to load both models and input the standard test file
     [Clair_de_lune__Claude_Debussy__Suite_bergamasque.wav](Clair_de_lune__Claude_Debussy__Suite_bergamasque.wav).
     This will plot three spectrograms: One for the original input data, 
     one after the autoencoder processing and one after the U-Net processing
    - If you use jupyter notebooks: Include the line `%run Main.py` in your notebook in order to initialise the models, 
    then use the exposed `pipeline(path):` function to put any _*.wav_ file through the models.
    - If you use standard Python scripts: Either extend the [Main.py](Main.py) script, or copy the model loading code 
    and use it in your own script.

## License
The code is free to use under the MIT license.


## Attributions
The piano recording used used for evaluation (`Clair_de_lune__Claude_Debussy__Suite_bergamasque.wav`) 
was performed by Laurens Goedhart and is available in the [public domain](https://commons.wikimedia.org/wiki/File:Clair_de_lune_(Claude_Debussy)_Suite_bergamasque.ogg) under the [Creative Commons Attribution 3.0 Unported](https://creativecommons.org/licenses/by/3.0/deed.en) license.