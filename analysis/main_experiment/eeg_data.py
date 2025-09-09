import yaml
import os
import pickle
import numpy as np

target_filenames = ['0434262454981c92.tiff', '0c6c2c66e61e3133.tiff', '0e2e2f2931313d37.tiff', '1332a2d4c4c14322.tiff', '202024d9b333968c.tiff', '24247432670b3131.tiff', '2c2064d163c52c78.tiff', '42431879e191d3c3.tiff', '61662ece1f6e7870.tiff', '7890918716766325.tiff', '85cc4e533959b1e1.tiff', '8ece4062eee68292.tiff', '93939b9a101e87e0.tiff', '948cacbc94b42474.tiff', 'af8f3a3939313171.tiff', 'b23232b232332361.tiff', 'b371f8ecf4e0f0f0.tiff', 'bce2d2d2c393e3b2.tiff', 'c77131717179f173.tiff', 'c93b3b2b29b9b8f1.tiff', 'cc8c9290bcbcf4fc.tiff', 'dad8acb8b82e36b1.tiff', 'e2f8f8fcf8f8f8f8.tiff', 'e6d8d40438c0c8e2.tiff', 'e6e648c62ebacc38.tiff', 'ef3632c2c1476e78.tiff', 'f4ca83dc6731310d.tiff']

def load_eeg_channel_and_timepoints():
    with open('../../eeg_info.yaml', 'r') as f:
        eeg_info = yaml.safe_load(f)['main_experiment']

    channel_names = eeg_info['channel_names']
    t = eeg_info['timepoints']


    return channel_names, t

def load_eeg_filenames(eeg_dir, sub):
    with open(os.path.join(eeg_dir, f'filenames_oads_eeg_rsvp_sub-{str(sub).zfill(2)}.pkl'), 'rb') as f:
        filenames = pickle.load(f)

    with open(os.path.join(eeg_dir, f'is_test_oads_eeg_rsvp_sub-{str(sub).zfill(2)}.pkl'), 'rb') as f:
        is_test = pickle.load(f)

    train_filenames = [filenames[i] for i in range(len(filenames)) if not is_test[i]]
    test_filenames = [filenames[i] for i in range(len(filenames)) if is_test[i]]

    train_ids = [x.split('.')[0] for x in train_filenames if f"{x.split('.')[0]}.tiff" not in target_filenames]
    test_ids = [x.split('.')[0] for x in test_filenames if f"{x.split('.')[0]}.tiff" not in target_filenames]

    return train_ids, test_ids

def load_eeg_data(sub, eeg_dir):
    with open(os.path.join(eeg_dir, f'filenames_oads_eeg_rsvp_sub-{str(sub).zfill(2)}.pkl'), 'rb') as f:
        filenames = pickle.load(f)

    with open(os.path.join(eeg_dir, f'is_test_oads_eeg_rsvp_sub-{str(sub).zfill(2)}.pkl'), 'rb') as f:
        is_test = pickle.load(f)

    data = np.load(os.path.join(eeg_dir, f'oads_eeg_rsvp_sub-{str(sub).zfill(2)}.npy'))
    
    train_filenames = [filenames[i] for i in range(len(filenames)) if not is_test[i]]
    test_filenames = [filenames[i] for i in range(len(filenames)) if is_test[i]]

    train_ids = [x.split('.')[0] for x in train_filenames if f"{x.split('.')[0]}.tiff" not in target_filenames]
    test_ids = [x.split('.')[0] for x in test_filenames if f"{x.split('.')[0]}.tiff" not in target_filenames]

    train_data = np.array([data[i] for i in range(len(data)) if not is_test[i]])
    test_data = np.array([data[i] for i in range(len(data)) if is_test[i]])

    return train_ids, test_ids, train_data, test_data