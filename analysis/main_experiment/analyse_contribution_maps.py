import os
import numpy as np
from skimage import transform
from tqdm import tqdm

from eeg_data import load_eeg_channel_and_timepoints

def create_rotations(contribution, channel_names, visual_channel_indices, res_dir, sub, available_timepoints=None, is_small=False):
    if available_timepoints is None:
        available_timepoints = [x for x in contribution[visual_channel_indices[0]].keys() if contribution[visual_channel_indices[0]][x] is not None]
        

    if is_small:
        rotations_per_channel = np.zeros((len(channel_names), 16, len(available_timepoints)))
    else:
        rotations_per_channel = np.zeros((len(channel_names), 22, len(available_timepoints)))

    do_channels = channel_names if is_small else visual_channel_indices

    for channel in tqdm(do_channels, desc='Eccentricity Channels', total=len(do_channels)):
        height, width = contribution[channel][available_timepoints[0]].shape            

        center = np.array([height//2, width//2])
        linewidth = 2 if is_small else 10
        step_size = 1 if is_small else 4

        rotations = []
        for rotation in range(0, 360, 10):
            eccentricities = []

            for margin in range(0, height-center[0], step_size):
                if center[0]+margin+step_size > height:
                    break

                mask = np.zeros((height, width))
                mask[center[0]+margin:center[0]+margin+step_size, center[1]-linewidth:center[1]+linewidth] = 1
                new_mask = transform.rotate(mask, rotation).astype(bool)

                ecc_per_timepoint = []

                for timepoint in available_timepoints:
                    _mean = np.nanmean(contribution[channel][timepoint][new_mask])
                    ecc_per_timepoint.append(_mean)

                eccentricities.append(ecc_per_timepoint)

            rotations.append(eccentricities)

        rotations_per_channel[channel] = np.mean(rotations, axis=0)

    # Save
    
    np.save(os.path.join(res_dir, f'rotations_per_channel_sub-{sub}.npy'), rotations_per_channel, allow_pickle=False)


if __name__ == '__main__':
    model_type = 'alexnet_imagenet'  # Change to the model type you are using
    layer_name = 'across-layers'

    contribution_dir = '../../results/{sub}/{model_type}/{layer_name}/contribution_maps'

    channel_names, t = load_eeg_channel_and_timepoints()

    n_channels = len(channel_names)
    n_timepoints = len(t)

    visual_channel_names = ['O1', 'O2', 'Oz', 'Iz', 'Pz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO7', 'POz', 'PO4', 'PO8', 'I1', 'I2']
    visual_channel_indices = [channel_names.index(ch) for ch in visual_channel_names if ch in channel_names]

    ##################################### Load random patch contributions
    for sub in range(5, 36):
        contribution = np.load(os.path.join(contribution_dir.format(sub=sub, model_type=model_type, layer_name=layer_name), f'sub-{sub}_average_random_patch_contributions.npy'), allow_pickle=True).item()

        create_rotations(contribution, channel_names, range(len(channel_names)), contribution_dir.format(sub=sub, model_type=model_type, layer_name=layer_name), sub)