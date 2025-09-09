import numpy as np
import os
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models.feature_extraction import create_feature_extractor

import sys
sys.path.append('../../..')

from utils import OADSImageDataset, record_activations, collate_fn, CustomOADS
from eeg_data import load_eeg_filenames


def extract_features(model_type, oads_dir, save_dir, fileending='.ARW', save_to_file:bool=True, subjects:list=list(range(5,36)), batch_size:int=16, num_workers:int=8, device='cuda:1', image_width:int=2155, image_height:int=1440):
    # Class to load images
    oads = CustomOADS(basedir=oads_dir, n_processes=num_workers, ending=fileending)

    # DNN model setup
    # width = 2155
    # height = 1440
    size = (int(image_height), int(image_width))

    if model_type == 'alexnet' or model_type == 'alexnet_imagenet':

        return_nodes = {
            'features.2': 'layer1',
            'features.5': 'layer2',
            'features.12': 'layer3',
        }
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

    else:
        raise ValueError(f"Model type {model_type} is not implemented.")

    model = model.to(device)
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

    # Transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_list = []
    transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(transform_list)

    #### TODO: Change so that once, all features are extracted into one big file, such that for each subject, the respective subselection can be loaded. 
    # Maybe an HDF5 file is the best solution for this.

    # Gather all IDs from all subjects
    all_ids = []

    eeg_dir = '../../eeg_data/main_experiment'
    for sub in subjects:
        train_ids, test_ids = load_eeg_filenames(eeg_dir, sub)

        all_ids.extend(train_ids)
        all_ids.extend(test_ids)

    all_ids = list(set(all_ids)) # [:50]

    print(len(all_ids))


    dataset = OADSImageDataset(oads_access=oads, item_ids=all_ids, return_index=True, transform=transform, device=device)
    dataloader = DataLoader(dataset, collate_fn=collate_fn,
                            batch_size=batch_size, shuffle=False, num_workers=oads.n_processes)

    activations = record_activations(loader=dataloader, models=[(model_type, feature_extractor)], device=device, layer_names=return_nodes.values(), flatten=False) # type: ignore

    if save_to_file:
        # ##### Save Activations
        os.makedirs(save_dir, exist_ok=True)

        npy_activations = {
            layer_name: np.vstack([activations[layer_name][img_id] for img_id in activations[layer_name].keys()]) for layer_name in activations.keys()
        }

        np.savez_compressed(f'{save_dir}/main_experiment_{model_type}_activations.npz', **npy_activations)
        # Save image order
        with open(f'{save_dir}/main_experiment_image_id_order.pkl', 'wb') as f:
            pickle.dump(all_ids, f)

    return activations

if __name__ == '__main__':

    ### Main experiment
    oads_dir = '../../stimuli'
    save_dir = '../../dnn_features'
    extract_features(model_type='alexnet_imagenet', oads_dir=oads_dir, save_dir=save_dir, save_to_file=True, subjects=list(range(5,36)))