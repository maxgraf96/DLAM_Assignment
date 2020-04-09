import os
import subprocess

# Initialise Fluid Synth with piano soundfont
from pathlib import Path
from Hyperparameters import sep

sample_rate = 22050
num_channels = 2

def synthesise_midi_to_audio(midi_path, output_path, soundfont_path):
    # Convert MIDI to raw PCM data
    raw_path = output_path[:-4] + ".dat"
    subprocess.call(['fluidsynth', '-F', raw_path, soundfont_path, midi_path, '-r', str(sample_rate)])

    # NB: This is necessary because windows has no easy support for libsndfile, the library that fluidsynth
    # uses internally to write valid wav files (file with headers).
    # Therefore we use this hack that relies on the sox package, which is platform independent
    # Convert raw PCM information into valid *.wav file
    subprocess.call(['sox', '-t',  'raw', '-r', str(sample_rate),  '-e',  'signed-integer', '-b', '16', '-c',
                     str(num_channels), raw_path, output_path])

    # Delete interim raw audio *.dat file
    os.remove(raw_path)

def synthesise_all(mode, soundfont_path):
    # Get midis
    midis = Path("data/MIDI").rglob("*.mid")
    for midi in midis:
        # Convert path to string
        midi_str = str(midi)
        # Get filename
        filename = midi_str.split(sep)[-1][:-4]
        # Only Chopin for now
        if filename[:3] != "chp":
            continue

        # Check if wav already exists
        if os.path.exists("data" + sep + mode + sep + filename + ".wav"):
            print("Wav file for " + filename + " already exists. Delete it if you want to regenerate it. Continuing...")
            continue

        print("Converting " + midi_str)
        synthesise_midi_to_audio(midi_str, "data" + sep + mode + sep + filename + ".wav", soundfont_path)

# Synthesise all piano files
# synthesise_all("piano", "SoundFonts/steinway.sf2")

# Synthesise all synth files
synthesise_all("synth", "SoundFonts/strings lyrical.sf2")

# synthesise_midi_to_audio("data/MIDI/chp_op18.mid", "data/synth/chp_op18.wav", "SoundFonts/strings lyrical.sf2")