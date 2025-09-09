import os
import numpy as np
from PIL import Image
import tqdm
import pickle

from eeg_data import load_eeg_data
from feature_extraction import extract_features

def get_random_patch_mask(n_patches, size, shape, no_overlap:bool=True, no_center=None):
    rand_mask = np.zeros(shape).astype(bool)

    counter = 0
    while counter < n_patches:
        _mask = np.zeros(shape).astype(bool)


        start_choices_row = []
        start_choices_col = []

        center_size_row = int(shape[0] * 0.3)
        center_size_col = int(shape[1] * 0.3)

        if no_center is not None and no_center:

            p = np.random.rand()
            if p < 0.5:
                row_start = np.random.choice(range(shape[0]))
                row_lim = int(shape[0] / 2 - center_size_row/2 - size)
                col_lim = int(shape[1] / 2 - center_size_col/2 - size)
                
                if row_start + size >= int(shape[0] / 2 - center_size_row/2) and row_start <= row_lim+center_size_row+size:
                    start_choices_col.extend(range(col_lim))
                    start_choices_col.extend(range(col_lim+center_size_col+size, shape[1]))
                else:
                    start_choices_col = range(shape[1])
                
                col_start = np.random.choice(start_choices_col)
            else:
                col_start = np.random.choice(range(shape[1]))
                col_lim = int(shape[1] / 2 - center_size_col/2 - size)
                row_lim = int(shape[0] / 2 - center_size_row/2 - size)
                if col_start + size >=  int(shape[1] / 2 - center_size_col/2) and col_start <= col_lim+center_size_col+size:
                    start_choices_row.extend(range(row_lim))
                    start_choices_row.extend(range(row_lim+center_size_row+size, shape[0]))
                else:
                    start_choices_row = range(shape[0])
                
                row_start = np.random.choice(start_choices_row)

            #     start_choices_row.extend(range(row_lim))
            #     start_choices_row.extend(range(row_lim+center_size_row+size, shape[0]))

            #     start_choices_col = range(shape[1])

            # else:
            #     start_choices_row = range(shape[0])
                
                # col_lim = int(shape[1] / 2 - center_size_col/2 - size)
            #     start_choices_col.extend(range(col_lim))
            #     start_choices_col.extend(range(col_lim+center_size_col+size, shape[1]))
                
        elif no_center is not None and not no_center:
            pass
        else:
            start_choices_row = range(shape[0])
            start_choices_col = range(shape[1])
            # row_start = np.random.randint(0, row_lim)
            # col_start = np.random.randint(0, col_lim)
            # row_lim = shape[0]
            # col_lim = shape[1]

            row_start = np.random.choice(start_choices_row)
            col_start = np.random.choice(start_choices_col)
        # if 
        row_end = row_start + size
        col_end = col_start + size

        

        _mask[row_start:row_end, col_start:col_end] = True

        if row_end > shape[0]:
            _mask[:row_end-shape[0], col_start:col_end] = True

        if col_end > shape[1]:
            _mask[row_start:row_end, :col_end - shape[1]] = True

        if row_end > shape[0] and col_end > shape[1]:
            _mask[:row_end-shape[0], :col_end-shape[1]] = True

        if no_overlap and rand_mask[_mask].any():
            continue
        # if rand_mask[_mask].any():
        #     continue

        rand_mask[_mask] = True

        counter += 1

    return rand_mask


def iterate_load_subject_data(args):
    sub, encoding_model_dir = args

    model_type = 'alexnet_imagenet'
    layer_name = 'across-layers'
    crop_condition = 'feature'
    crop_instance = 'feature-full'
    fraction = 1.0

    filename = f'encoding_results_sub_{sub}_{layer_name}_{model_type}-feature-cropping-{crop_condition}-{crop_instance}-{fraction}.pkl'

    with open(os.path.join(encoding_model_dir, filename), 'rb') as f:
        results = pickle.load(f)

    pca = results['pca']
    lin_reg = results['lin_reg']

    return sub, pca, lin_reg



def iter(args):
    sub, all_masks, shape, n_iterations, results, filename, num_workers = args

    device = 'cuda:0'
    result_dir = '../../results'
    encoding_model_dir = f'../../results/sub-{sub}/alexnet_imagenet/across-layers/feature-feature-full-1.0'
    load_features_from_file = False
    
    model_type = 'alexnet_imagenet'

    _, pca, lin_reg = iterate_load_subject_data((sub, encoding_model_dir))


    eeg_dir = '../../eeg_data/main_experiment'
    train_ids, _, train_data, _ = load_eeg_data(sub=sub, eeg_dir=eeg_dir)
    _, n_channels, n_timepoints = train_data.shape

    feature_dir = '../../dnn_features'
    if load_features_from_file:
        # Load extracted features
        activations = np.load(os.path.join(feature_dir, f'main_experiment_{model_type}_activations.npz'), allow_pickle=True)
        
        with open(os.path.join(feature_dir, f'main_experiment_image_id_order.pkl'), 'rb') as f:
            image_id_order = pickle.load(f)

        train_id_indices = [image_id_order.index(x) for x in train_ids if x in image_id_order]

        activations = {
            layer_name: {
                image_index: activations[layer_name].item().get(image_index) for image_index in train_id_indices
            } for layer_name in activations.keys()
        }
        
    else:
        activations = extract_features(save_to_file=False, subjects=[sub], oads_dir='../stimuli', model_type=model_type, save_dir=feature_dir, device=device)

    # Divide extracted features into training and test sets
    activations = {
        layer_name: {
            image_index: feature for image_index, feature in activations[layer_name].items() if image_index in train_ids
        } for layer_name in activations.keys()
    }

    all_patch_correlations = {
        channel: {
            timepoint: [] for timepoint in range(n_timepoints)
        } for channel in range(n_channels)
    }

    make_new_mask = False
    if all_masks is None:
        all_masks = []
        make_new_mask = True

        n_patches = 10
        patch_size = 4

        if filename is None:
            filename = f'random_patch_contribution_{n_patches}_{patch_size}x{patch_size}.pkl'

    if shape is None:
        shape = {}

    for patch_index in tqdm.tqdm(range(n_iterations), total=n_iterations, desc='Patches'):
        if make_new_mask:
            rand_mask = {}
        else:
            rand_mask = all_masks[patch_index]
    
        no_center = None # if patch_index > 1000 else True

        all_patch_activations = []
        for layer in ['layer1', 'layer2', 'layer3']:
            
            all_images = []
            for image_index in range(len(activations[f'alexnet_{layer}'])):
                x = []
                for feature_index in range(len(activations[f'alexnet_{layer}'][image_index])):
                    feature = activations[f'alexnet_{layer}'][image_index][feature_index]

                    ##############
                    if layer not in shape:
                        shape[layer] = feature.shape
                        
                    if make_new_mask and layer not in rand_mask:
                        if layer == 'layer1':
                            rand_mask[layer] = get_random_patch_mask(n_patches, patch_size, shape[layer], no_overlap=True, no_center=no_center)
                        else:
                            rand_mask[layer] = np.array(Image.fromarray(rand_mask['layer1']).resize(list(shape[layer])[::-1]))
                        # if layer == 'layer1':
                        #     for _layer in ['layer2', 'layer3']:
                        #         rand_mask[_layer] = np.array(Image.fromarray(rand_mask['layer1']).resize(list(shape[_layer])[::-1]))
                    ##############
                    feature = np.where(rand_mask[layer], feature, 0)
                    x.append(feature.flatten())
                all_images.append(np.array(x).flatten())

            all_patch_activations.append(np.array(all_images))
            
        all_patch_activations = np.hstack(all_patch_activations)
        all_patch_activations = pca.transform(all_patch_activations)
        design_matrix = np.hstack((np.ones((all_patch_activations.shape[0], 1)), all_patch_activations))

        predictions = lin_reg.predict(design_matrix).reshape(-1, n_channels, n_timepoints)

        for channel in range(n_channels):
            for timepoint in range(n_timepoints):
                ############# TESTING #################
                c = np.corrcoef(predictions[:, channel, timepoint], train_data[:, channel, timepoint])[0, 1]

                all_patch_correlations[channel][timepoint].append(c)

        if make_new_mask:
            all_masks.append(rand_mask)

    # Save results
    results = {
        'all_masks': all_masks,
        'all_patch_correlations': all_patch_correlations,
        'shape': shape,
    }

    os.makedirs(os.path.join(result_dir, str(sub)), exist_ok=True)
    with open(os.path.join(result_dir, str(sub), f'sub-{sub}_{filename}'), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    num_workers = 5 # nproc
    
    for sub in range(5, 36):
        results = None
        all_masks = None
        shape = None
        n_iterations = 1000

        iter((sub, all_masks, shape, n_iterations, results, None, num_workers))
