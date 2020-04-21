import os
import subprocess

# Initialise Fluid Synth with piano soundfont
from pathlib import Path
from Hyperparameters import sep, limit_s

# The sample rate and number of output channels used by FluidSynth to synthesise *.wav files
sample_rate = 22050
num_channels = 2

# The paths to the SoundFont files for the piano and electronic piano
sf_piano = "SoundFonts" + sep + "steinway.sf2"
sf_synth = "SoundFonts" + sep + "JR_elepiano.sf2"

def synthesise_midi_to_audio(midi_path, output_path, soundfont_path):
    """
    Synthesise a *.mid file to a *.wav file using the specified SoundFont
    :param midi_path: The path to the *.mid file
    :param output_path: The desired output path for the *.wav file
    :param soundfont_path: The path to the SoundFont that FluidSynth should use
    :return: None
    """
    # Create destination folder if it doesn't exist
    path_str = str(output_path)
    dest_dir = path_str[:path_str.rfind(sep)]
    if not os.path.exists(dest_dir):
        Path(dest_dir).mkdir(parents=True, exist_ok=True)

    # Convert MIDI to raw PCM data
    raw_path = output_path[:-4] + ".dat"
    subprocess.call(['fluidsynth', '-F', raw_path, soundfont_path, midi_path, '-r', str(sample_rate)])

    # NB: This is necessary because windows has no easy support for libsndfile, the library that fluidsynth
    # uses internally to write valid wav files (file with headers).
    # Therefore we use this hack that relies on the sox package, which is platform independent
    # Convert raw PCM information into valid *.wav file
    pretrimmed = output_path[:-4] + '_full_length.wav'
    subprocess.call(['sox', '-t',  'raw', '-r', str(sample_rate),  '-e',  'signed-integer', '-b', '16', '-c',
                     str(num_channels), raw_path, pretrimmed])

    # Trim to "limit_s" seconds, specified in Hyperparameters.py
    subprocess.call(['sox', pretrimmed, output_path, 'trim', '0', str(limit_s)])

    # Delete interim raw audio *.dat and full length files
    os.remove(pretrimmed)
    os.remove(raw_path)

def synthesise_all(mode, soundfont_path):
    """
    Synthesise all *.mid files in the "data/MIDI" folder for a given mode (currently "piano" and "synth") using the
    specified SoundFont.
    This method will currently output *.wav files to a fixed folder ("data/"mode")
    :param mode: The mode of the synthesised instrument (either "piano" or "synth" at the moment)
    :param soundfont_path: The path to the SoundFont
    :return: None
    """
    # Get MIDI files
    midis = Path("data" + sep + "MIDI").rglob("*.mid")
    for midi in midis:
        # Convert path to string
        midi_str = str(midi)
        # Extract filename
        filename = midi_str.split(sep)[-1][:-4]

        # Check if *.wav file already exists
        if os.path.exists("data" + sep + mode + sep + filename + ".wav"):
            print("Wav file for " + filename + " already exists. Delete it if you want to regenerate it. Continuing...")
            continue

        print("Converting " + midi_str)
        synthesise_midi_to_audio(midi_str, "data" + sep + mode + sep + filename + ".wav", soundfont_path)

def create_test_set():
    """
    Synthesise the MIDI files used for the test set
    :return: None
    """
    midis = Path("data" + sep + "MIDI_test").rglob("*.mid")
    for midi in midis:
        # Convert path to string
        midi_str = str(midi)
        # Extract filename
        filename = midi_str.split(sep)[-1][:-4]

        # Check if wav already exists
        if os.path.exists("data" + sep + "test" + sep + filename + ".wav"):
            print("Wav file for " + filename + " already exists. Delete it if you want to regenerate it. Continuing...")
            continue

        print("Converting " + midi_str)
        synthesise_midi_to_audio(midi_str, "data" + sep + "test" + sep + filename + ".wav", sf_piano)

# Synthesise all piano files
synthesise_all("piano", sf_piano)

# Synthesise all synth files
synthesise_all("synth", sf_synth)

# Create test set with mozart songs
create_test_set()