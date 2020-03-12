import os

import librosa
import pandas as pd
from pathlib import Path

def import_offline_pack(root_dir, path_pack_csv, instrument_category):
    """
    Import a downloaded pack and create a csv for it
    :param root_dir: Name of the subfolder containing the extracted WAV data in the "data" folder
    :return:
    """
    if not os.path.exists(path_pack_csv):
        # Create list for annotation data
        data = []
        sounds = Path("data" + os.path.sep + root_dir).rglob("*.wav")

        # Create annotation data
        for sound in sounds:
            path = str(sound)
            # Check validity
            try:
                librosa.load(path, sr=None, mono=True)
            except:
                # File invalid => delete and continue
                os.remove(path)
                print("INFO: Removed invalid wav file " + path + ".")
                continue
            # Store annotation data
            name = path[path.rindex(os.path.sep):].replace(os.path.sep, "")
            data.append([name, instrument_category, 0, ""])

            print("Successfully created annotation for " + path + ".")

        # Convert annotation data to dataframe
        df = pd.DataFrame(data, columns=["Name", "Instrument Class", "Class Number", "Tags"])
        # Set dataframe path
        path_df = root_dir + ".csv"
        if not os.path.exists(path_df):
            # Save dataframe to CSV
            df.to_csv(path_pack_csv, index=False)
        else:
            # Concat df to existing data
            existing = pd.read_csv(path_df)
            df = pd.concat([existing, df])
            df.to_csv(path_pack_csv, index=False)
        print("Done importing pack '" + root_dir + "'.")
    else:
        print("Pack " + root_dir + " already exists.")

def download_sounds(fs_interface, root_dir, terms, tags, instrument_categories, ac_single_event, ac_note_names, path_csv):
    # First check if any data for this dataset was queried yet
    # If not, query all the data
    if not os.path.exists(path_csv):
        for note in ac_note_names:
            for i in range(len(terms)):
                # Create part: This could be something like a set of files describing a bright synth
                create_part(fs_interface, root_dir, terms[i], tags, instrument_categories[i], class_nr=i,
                                           ac_single_event=ac_single_event, ac_note_name=note)

    # Used if a dataset should be extended
    else:
        # Some part of the data already exists => go over dataframe and check which parts already exist
        # And only add those parts that don't exist yet
        # Get instrument category column
        col_category = pd.read_csv(path_csv).values[:, 1]
        for note in ac_note_names:
            for i in range(len(terms)):
                if instrument_categories[i] not in col_category:
                    # Create part: This could be something like a set of files describing a bright synth
                    create_part(fs_interface, root_dir, terms[i], tags, instrument_categories[i],
                                               class_nr=i,
                                               ac_single_event=ac_single_event, ac_note_name=note)

def create_part(fs_interface, root_dir, term, tags, instrument_category, class_nr, ac_single_event, ac_note_name=None):
    # Get sounds
    sounds = fs_interface.search(term, tags, ac_single_event, ac_note_name)
    # Save sounds
    save_sounds(sounds, fs_interface, root_dir, term, tags, instrument_category, class_nr, ac_single_event, ac_note_name)

def save_sounds(sounds, fs_interface, root_dir, term, tags, instrument_category, class_nr, ac_single_event, ac_note_name=None):
    tags_str = ",".join(tags)
    # Number of files to download
    max_files = 30

    # Folder to which the files should be saved
    save_dir = "data/" + root_dir + "/"
    # Create list for annotation data
    data = []
    # Create folder (no effect if folder already exists)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Store the *.wav files
    counter = 0
    for sound in sounds:
        # Get sound data
        id = str(sound["id"])
        name = str(sound["name"])
        # Download raw audio data
        wav = fs_interface.download_sound(id)
        # Save raw audio data
        # Replace all file endings with ".wav" to avoid edge cases
        name = name[:-4] + "_" + id + ".wav"
        # Replace all slashes, backslashes and quotation marks with "_"
        name = name.replace("/", "_")
        name = name.replace("\\", "_")
        name = name.replace("\"", "_")
        name = name.replace("'", "_")
        f = open(save_dir + name, 'wb')
        f.write(wav)
        f.close()
        # Check validity
        try:
            librosa.load(save_dir + name, sr=None, mono=True)
        except:
            # File invalid => delete and continue
            os.remove(save_dir + name)
            print("INFO: Removed invalid wav file " + save_dir + name + ".")
            continue
        # Store annotation data
        data.append([name, instrument_category, class_nr, tags_str])

        print("Saved " + name + ".")
        counter = counter + 1
        if counter >= max_files:
            break

    # Convert annotation data to dataframe
    df = pd.DataFrame(data, columns=["Name", "Instrument Class", "Class Number", "Tags"])
    # Set dataframe path
    path_df = save_dir + root_dir + ".csv"
    if not os.path.exists(path_df):
        # Save dataframe to CSV
        df.to_csv(save_dir + root_dir + ".csv", index=False)
    else:
        # Concat df to existing data
        existing = pd.read_csv(path_df)
        df = pd.concat([existing, df])
        df.to_csv(save_dir + root_dir + ".csv", index=False)
    print("Done saving data for term '" + term + "'.")


def download_pack(fs_interface, pack_id, root_dir, path_pack_csv):
    """
    Download a pack from freesound.org
    TODO: Not implemented yet
    :param fs_interface:
    :param pack_id:
    :param root_dir:
    :param path_pack_csv:
    :return:
    """
    raise NotImplementedError

    # First check if this pack was queried yet
    # if not os.path.exists(path_pack_csv):
    #     Download pack
    # pack = fs_interface.download_pack(pack_id)
    # else:
    #     print("Data for pack " + pack_id + " already exists. Aborting download...")