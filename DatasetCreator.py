import os
import pandas as pd
from pathlib import Path

def create_part(fs_interface, root_dir, term, tags, instrument_category, class_nr):
    """

    :param fs_interface:
    :param root_dir:
    :param term:
    :param tags:
    :param instrument_category:
    :param class_nr: Number corresponding to the instrument_category! Needed for training in pytorch
    :return:
    """
    tags_str = ",".join(tags)
    sounds = fs_interface.search(term, tags=tags, ac_single_event=True)
    # Number of files to download
    max_files = 5

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
        # Append ".wav" if file name doesn't include it
        if ".wav" not in name:
            name = name + ".wav"
        f = open(save_dir + name, 'wb')
        f.write(wav)
        f.close()
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