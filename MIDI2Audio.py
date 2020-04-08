import os
import subprocess

# Initialise Fluid Synth with piano soundfont
soundfont_path = "Sound Fonts/steinway.sf2"
sample_rate = 44100
num_channels = 2

def synthesise_midi_to_audio(midi_path, output_path):
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

synthesise_midi_to_audio("data/MIDI/alb_esp1.mid", "data/piano/alb_esp1.wav")
