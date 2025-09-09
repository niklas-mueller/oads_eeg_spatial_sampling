import os
import pickle
import numpy as np
from mne import read_epochs

target_filenames = ['0434262454981c92.png', '0c6c2c66e61e3133.png', '0e2e2f2931313d37.png', '1332a2d4c4c14322.png', '202024d9b333968c.png', '24247432670b3131.png', '2c2064d163c52c78.png', '42431879e191d3c3.png', '61662ece1f6e7870.png', '7890918716766325.png', '85cc4e533959b1e1.png', '8ece4062eee68292.png', '948cacbc94b42474.png', 'af8f3a3939313171.png', 'b23232b232332361.png', 'b371f8ecf4e0f0f0.png', 'bce2d2d2c393e3b2.png', 'c77131717179f173.png', 'c93b3b2b29b9b8f1.png', 'cc8c9290bcbcf4fc.png', 'dad8acb8b82e36b1.png', 'e2f8f8fcf8f8f8f8.png', 'e6d8d40438c0c8e2.png', 'e6e648c62ebacc38.png', 'ef3632c2c1476e78.png', 'f4ca83dc6731310d.png', 'c863a5a58c1c4ecc.png', '767272e2a2363f39.png', 'ce8e8e8a9a1ccad1.png', 'f2eaa1e1e3e66c91.png', '89888998999b8eec.png', '6984989733339d9c.png', '8888848cccc8d941.png', '1c44d41c0cac24a5.png', 'e48c9c9de4e78f86.png']

def load_eeg_channel_and_timepoints(eeg_fif_dir, sub, exp_condition):
    # eeg_dir_res = f'data_SECOND_BATCH_manual/fif_files/sub012_ses{str(1).zfill(3)}center1-OC&CSD-epo.fif'
    epochs = read_epochs(os.path.join(eeg_fif_dir, f'sub{str(sub).zfill(3)}_ses{str(1).zfill(3)}{exp_condition}-OC&CSD-epo.fif'), preload=False)
    channel_names = epochs.ch_names
    n_timepoints = len(epochs.times)
    sample_rate = 1024
    t = [i/sample_rate - 0.1 for i in range(n_timepoints)]
    channel_names = epochs.ch_names

    return channel_names, t

def load_eeg_filenames(eeg_dir, exp_condition):
    train_ids = np.load(os.path.join(eeg_dir, f'train_filenames.npy'))
    test_ids = np.load(os.path.join(eeg_dir, f'test_filenames.npy'))
    
    condition_index = int(exp_condition[-1])-1
    condition_name = 'center_gray' if 'center' in exp_condition else 'periphery_gray'
    train_ids = [f'condition-{condition_name}/{x.split(".")[0]}_index-{condition_index}.png' for x in train_ids if x not in target_filenames]
    test_ids = [f'condition-{condition_name}/{x.split(".")[0]}_index-{condition_index}.png' for x in test_ids if x not in target_filenames]
    
    return train_ids, test_ids

def load_eeg_data(sub, exp_condition, eeg_dir):

    train_data = np.load(os.path.join(eeg_dir, f'sub{str(sub).zfill(3)}', f'sub{str(sub).zfill(3)}_{exp_condition}_traindata.npy'))
    test_data = np.load(os.path.join(eeg_dir, f'sub{str(sub).zfill(3)}', f'sub{str(sub).zfill(3)}_{exp_condition}_testdata.npy'))

    if 'SECOND' not in eeg_dir:
        train_data = np.nanmean(train_data, axis=1)
        test_data = np.nanmean(test_data, axis=1)   

    return train_data, test_data