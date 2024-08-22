import os
import mne
import pandas as pd
import numpy as np
from mne_bids import write_raw_bids, BIDSPath



def run_subject(sub):

    # Set the path to the raw data directory
    data_dir = '/home/nmuller/projects/fmg_storage/Data'

    # Load the raw data
    data = mne.io.read_raw_bdf(os.path.join(data_dir, f'sub_{sub}', f'sub_{sub}.bdf'))

    # Set the channel types
    data.set_channel_types(dict(zip(['left', 'right', 'above', 'below', 'left-ref', 'right-ref', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp'], ['eog']*4 + ['misc']*13)))

    # Set the channel names and montage
    montage = mne.channels.make_standard_montage('biosemi64')
    montage.rename_channels(dict(zip(montage.ch_names, data.ch_names)))

    data.set_montage(montage)


    # Initialize BIDS parameters
    subject_id = f"{str(sub).zfill(3)}"
    task = "RSVP"

    bids_root = '/home/nmuller/projects/fmg_storage/BIDS-Data/'
    bids_path = BIDSPath(subject=subject_id, task=task, root=bids_root)

    # Get events
    events = mne.find_events(data)

    # Get trigger id mapping codes
    sub_matrix_filename = [x for x in os.listdir(os.path.join(data_dir, f'sub_{sub}')) if x.endswith('csv') and 'randomized_matrix' in x][0]

    sub_images = pd.read_csv(os.path.join(data_dir, f'sub_{sub}', sub_matrix_filename), header=None)
    sub_images = sub_images.values.tolist()
    sub_images = [i for image in sub_images for i in image]   #Flatten the list

    all_images_df = pd.read_csv(os.path.join(data_dir, "EventsID_Dictionary.csv"), header = None)
    all_images = dict(zip(all_images_df[0], all_images_df[1]))


    # Convert trigger ids
    events_len = len(events)
    event_index = 0
    converted_events = []

    for i in range(events_len):

        converted_events.append(events[i, :])
        
        if events[i, 2] > 250:
            converted_events[i][2] = 5000 + events[i, 2]
            # print(events[i, 2])
            # all_images[converted_events[i][2]] = str(converted_events[i][2])
            continue
        
        elif 1 <= events[i, 2] <= 250:
            this_image = sub_images[event_index]
            if this_image.startswith('Stimuli\\Targets\\'):
                this_image = this_image[16:]
            elif this_image.startswith('Stimuli\\'):
                this_image = this_image[8:]
            
            converted_events[i][2] = all_images[this_image]
            
            event_index += 1

    converted_events = np.array(converted_events)


    event_id = {
        str(x): x for x in converted_events[:, 2]
    }


    # Write the BIDS data
    write_raw_bids(data, bids_path, overwrite=True, allow_preload=True, format='EDF', events=converted_events, event_id=event_id)


if __name__ == '__main__':
    for sub in range(7, 36):
        run_subject(sub)