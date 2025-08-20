import os
import sys
from mne import read_epochs
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle

home_path = os.path.expanduser('~')

def do_subject(sub, cleaning):
    
    if 'niklas' in home_path:
        eeg_dir = f'/mnt/z/Projects/2023_Scholte_FMG1441/Data/sub_{sub}/Preprocessed epochs/sub_{sub}-OC&CSD{cleaning}-epo.fif'
    elif 'nmuller' in home_path:
        eeg_dir = f'{home_path}/projects/fmg_storage/Data/sub_{sub}/Preprocessed epochs/sub_{sub}-OC&CSD{cleaning}-epo.fif'
        # eeg_dir = f'{home_path}/projects/data/oads_eeg/sub_{sub}/sub_{sub}-OC&CSD{cleaning}-epo.fif'

    epochs = read_epochs(fname=eeg_dir, preload=True, verbose=False)

    channel_names = epochs.ch_names
    visual_channel_names = ['O1', 'O2', 'Oz', 'Iz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO7', 'POz', 'PO4', 'PO8']
    visual_channel_indices = [i for i in range(len(channel_names)) if channel_names[i] in visual_channel_names]

    if 'niklas' in home_path:
        event_ids = pd.read_csv(f'/mnt/z/Projects/2023_Scholte_FMG1441/EventsID_Dictionary.csv', header=None)
    elif 'nmuller' in home_path:
        event_ids = pd.read_csv(f'/home/nmuller/projects/data/oads_eeg/EventsID_Dictionary.csv', header=None)
    event_ids = {id: filename for _, (filename, id) in event_ids.iterrows()}

    # target_filenames = os.listdir(f'/mnt/z/Projects/2023_Scholte_FMG1441/Stimuli/reduced/Targets (reduced)')
    # target_filenames = [x for x in target_filenames]
    target_filenames = ['0434262454981c92.tiff', '0c6c2c66e61e3133.tiff', '0e2e2f2931313d37.tiff', '1332a2d4c4c14322.tiff', '202024d9b333968c.tiff', '24247432670b3131.tiff', '2c2064d163c52c78.tiff', '42431879e191d3c3.tiff', '61662ece1f6e7870.tiff', '7890918716766325.tiff', '85cc4e533959b1e1.tiff', '8ece4062eee68292.tiff', '93939b9a101e87e0.tiff', '948cacbc94b42474.tiff', 'af8f3a3939313171.tiff', 'b23232b232332361.tiff', 'b371f8ecf4e0f0f0.tiff', 'bce2d2d2c393e3b2.tiff', 'c77131717179f173.tiff', 'c93b3b2b29b9b8f1.tiff', 'cc8c9290bcbcf4fc.tiff', 'dad8acb8b82e36b1.tiff', 'e2f8f8fcf8f8f8f8.tiff', 'e6d8d40438c0c8e2.tiff', 'e6e648c62ebacc38.tiff', 'ef3632c2c1476e78.tiff', 'f4ca83dc6731310d.tiff']

    epoch_filenames = [event_ids[x] for x in epochs.events[:, 2]]

    indices_per_id = {}

    for index, id in enumerate(epochs.events[:, 2]):
        # print(event_ids[id], filenames[0])
        if event_ids[id] in target_filenames:
            continue

        if id in indices_per_id:
            indices_per_id[id].append(index)
        else:
            indices_per_id[id] = [index]

    sub_data = []
    filenames = []
    is_test = []

    for id, indices in indices_per_id.items():
        # if use_all_repetitions:
        # if len(indices) > 5:
        # sub_data.append(np.mean([epochs[index].get_data() for index in indices], axis=0))
        sub_data.append({index: epochs[index].get_data() for index in indices})
        filenames.append(epoch_filenames[indices[0]])

        if len(indices) > 5:
            is_test.append(True)
        else:
            is_test.append(False)

        # else:
        #     if len(indices) > use_n_repetitions:
        #         # sub_data.append(np.mean(np.array([epochs[index].get_data() for index in indices])[np.random.choice(range(len(indices)), size=use_n_repetitions,  replace=False)], axis=0))
        #         filenames.append(epoch_filenames[indices[0]])
            
        #         if len(indices) > 5:
        #             is_test.append(True)
        #         else:
        #             is_test.append(False)


    # train_ids = [x.split('.')[0] for x in train_filenames]
    # test_ids = [x.split('.')[0] for x in test_filenames]

    # sub_data = np.array(sub_data)

    # oads_erp[sub] = sub_data
    # sub_filenames[sub] = filenames
    # sub_is_test[sub] = is_test

    return sub_data, filenames, is_test




if __name__ == '__main__':
    cleaning = ''# '-AutoReject'

    oads_erp = {}
    sub_filenames = {}
    sub_is_test = {}

    target_dir = f'/home/nmuller/projects/fmg_storage/osf_eeg_data/{cleaning if cleaning != "" else "Default"}'
    os.makedirs(target_dir, exist_ok=True)  

    for sub in tqdm(range(10, 36), total=31):
        try:
            sub_data, filenames, is_test = do_subject(sub, cleaning)
        except FileNotFoundError as e:
            print(e)
            continue

        oads_erp[sub] = sub_data
        sub_filenames[sub] = filenames
        sub_is_test[sub] = is_test

        # for subject in sub_filenames.keys():
        #### Save EEG DATA
        # erp = oads_erp[subject][:,0,:64,:]
        erp = np.array([
            np.mean([sub_data[image_index][rep_index][:64,:] for rep_index in sub_data[image_index].keys()], axis=0) for image_index in range(len(sub_data))
        ])
        filename = f'oads_eeg_rsvp_sub-{str(sub).zfill(2)}'
        np.save(file=os.path.join(target_dir, f"{filename}.npy"), arr=erp)

        with open(os.path.join(target_dir, f"{filename}.pkl"), 'wb') as f:
            pickle.dump(file=f, obj=sub_data)

        #### Save Filenames
        filename = f'filenames_oads_eeg_rsvp_sub-{str(sub).zfill(2)}.pkl'
        with open(os.path.join(target_dir, filename), 'wb') as f:
            pickle.dump(file=f, obj=filenames)

        #### Save is test
        filename = f'is_test_oads_eeg_rsvp_sub-{str(sub).zfill(2)}.pkl'
        with open(os.path.join(target_dir, filename), 'wb') as f:
            pickle.dump(file=f, obj=is_test)

