import os
import numpy as np
from PIL import Image
import tqdm
import multiprocessing
import pickle

from eeg_data import load_eeg_data
from feature_extraction import extract_features

# class CustomOADS():
#     def __init__(self, basedir, n_processes):
#         self.basedir = basedir
#         self.image_dir = os.path.join(basedir, 'oads_arw', 'ARW')
#         self.n_processes = n_processes

#         self.image_names = os.listdir(self.image_dir)

#     def load_image(self, image_name):
#         # try:
#         with rawpy.imread(os.path.join(self.image_dir, f'{image_name}.ARW')) as raw:
#             img = raw.postprocess()
#             img = Image.fromarray(img)
#         # except Error as e:
#         #     print(e)
#         #     print(image_name)
#         #     exit(1)
#         # label = self.get_annotation(
#         #     dataset_name=dataset_name, image_name=image_name, is_raw=is_raw)
#         label = ''
#         tup = (img, label)

#         return tup

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

    # unique_values = {}
    rows = []
    # rows_pca = []
    for image_representation in ['rgb']:
        for image_quality in ['raw']: 
            for n_pca_components in [100]: 
                for model_type in ['alexnet']: # alexnet_scce_full-crop
                    for layer in ['across-layers']: # f'{model_type}_layer1',f'{model_type}_layer2',f'{model_type}_layer3', 
                    # for layer in tqdm.tqdm(['across-layers'], total=1):
                        for image_resolution in ['400']: # '_400'
                            # for crop_condition in ['gcs', 'gcs_inverse', 'fraction']:
                            for crop_condition in ['feature']: # 'gcs', 'gcs_inverse', 'fraction', 

                                crop_instances = ['gcs-full'] if 'gcs' in crop_condition else (['center', 'periphery', 'center_circ', 'periphery_circ'] if 'fraction' in crop_condition else ['feature-full'])
                                fractions = [1.0] if 'gcs' in crop_condition or 'feature' in crop_condition else [0.005, 0.01, 0.05, 0.1, 0.2] # 

                                for fraction in fractions:
                                    for crop_instance in crop_instances:


                                        results = result_manager.load_result(filename=f'encoding_results_pca_{n_pca_components}_sub_{sub}_{model_type}_feature-cropping_{layer}_{image_representation}_{image_quality}_{image_resolution}_{crop_condition}_{crop_instance}_{fraction}.pkl')
                                        folder = result_manager.root

                                        # print(results.keys())
                                        # return
                                        
                                        if results is not None:

                                            # return results['lin_reg']
                                            rows.append([folder, sub, model_type, layer, image_representation, image_quality, n_pca_components, image_resolution, crop_condition, crop_instance, fraction, 'lin_reg', results['lin_reg']])
                                            rows.append([folder, sub, model_type, layer, image_representation, image_quality, n_pca_components, image_resolution, crop_condition, crop_instance, fraction, 'pca', results['pca']])
                                            continue
                                            
                                            if 'timepoints' in results.keys():
                                                timepoints = range(len(results['timepoints']))
                                                
                                            else:
                                                timepoints = None

                                            if 't' in results.keys():
                                                t = results['t']
                                            else:
                                                t = [i/sample_rate - 0.1 for i in range(n_timepoints)]

                                            # if 'explained_variance' in results.keys():
                                            #     for layer_index in range(len(results['explained_variance'])):
                                            #         for component in range(len(results['explained_variance'][layer_index])):
                                            #             rows_pca.append([folder, sub, model_type, image_representation, image_quality, n_pca_components, component, image_resolution, layer, f'explained_variance', results['explained_variance'][layer_index][component]])
                                            #             # rows_pca.append([folder, sub, model_type, image_representation, image_quality, n_pca_components, component, image_resolution, layer, f'test_explained_variance', results['test_explained_variance'][layer_index][component]])

                                            for channel in results['corr_channels'].keys():
                                                if timepoints is None:
                                                    timepoints = range(len(results['corr_channels'][channel]))

                                                for timepoint in timepoints:
                                                    r2_train = results['corr_channels'][channel][timepoint]
                                                    rows.append([folder, sub, model_type, layer, image_representation, image_quality, n_pca_components, image_resolution, crop_condition, crop_instance, fraction, channel_names[channel], channel, t[timepoint], timepoint, -1, 'corr_train', r2_train])
                                                    # for col_idx, value in enumerate([folder, sub, model_type, layer, image_representation, image_quality, n_pca_components, image_resolution, crop_condition, crop_instance, fraction, channel_names[channel], channel, t[timepoint], timepoint, -1, 'r2_train', r2_train]):
                                                    #     if col_idx not in unique_values:
                                                    #         unique_values[col_idx] = [value]
                                                    #     elif value not in unique_values[col_idx]:
                                                    #         unique_values[col_idx].append(value)

                                                    
                                                    # r2_test = results['test_r2_channels'][channel][timepoint]
                                                    # rows.append([folder, sub, model_type, layer, image_representation, image_quality, n_pca_components, image_resolution, crop_condition, crop_instance, fraction, channel_names[channel], channel, t[timepoint], timepoint, -1, 'r2_test', r2_test])
                                                    # for col_idx, value in enumerate([folder, sub, model_type, layer, image_representation, image_quality, n_pca_components, image_resolution, crop_condition, crop_instance, fraction, channel_names[channel], channel, t[timepoint], timepoint, -1, 'r2_test', r2_test]):
                                                    #     if col_idx not in unique_values:
                                                    #         unique_values
                                                    #     elif value not in unique_values[col_idx]:
                                                    #         unique_values[col_idx].append(value)

                                                    
                                                    test_corr_train = results['test_corr_channels'][channel][timepoint]
                                                    rows.append([folder, sub, model_type, layer, image_representation, image_quality, n_pca_components, image_resolution, crop_condition, crop_instance, fraction, channel_names[channel], channel, t[timepoint], timepoint, -1, 'test_corr_train', test_corr_train])
                                                    # for col_idx, value in enumerate([folder, sub, model_type, layer, image_representation, image_quality, n_pca_components, image_resolution, crop_condition, crop_instance, fraction, channel_names[channel], channel, t[timepoint], timepoint, -1, 'test_corr_train', test_corr_train]):
                                                    #     if col_idx not in unique_values:
                                                    #         unique_values[col_idx] = [value]
                                                    #     elif value not in unique_values[col_idx]:
                                                    #         unique_values[col_idx].append(value)

                                                    
                                                    # beta = results['beta_channels'][channel][timepoint]
                                                    # rows.append([folder, sub, model_type, image_representation, image_quality, n_pca_components, channel_names[channel], t[timepoint], 'beta', beta])
                                                    # for col_idx, value in enumerate([folder, sub, model_type, image_representation, image_quality, n_pca_components, channel_names[channel], t[timepoint], 'beta', beta]):
                                                    #     if col_idx not in unique_values:
                                                    #         unique_values[col_idx] = [value]
                                                    #     elif value not in unique_values[col_idx]:
                                                    #         unique_values[col_idx].append(value)


                                                    # test_pred = results['test_pred_channels'][channel][timepoint]
                                                    # for pred_index, pred in enumerate(test_pred):
                                                    #     rows.append([folder, sub, model_type, layer, image_representation, image_quality, n_pca_components, image_resolution, crop_condition, crop_instance, fraction, channel_names[channel], channel, t[timepoint], timepoint, pred_index, 'test_pred', pred])
                                                    #     # for col_idx, value in enumerate([folder, sub, model_type, layer, image_representation, image_quality, n_pca_components, image_resolution, crop_condition, crop_instance, fraction, channel_names[channel], channel, t[timepoint], timepoint, pred_index, 'test_pred', pred]):
                                                    #     #     if col_idx not in unique_values:
                                                    #     #         unique_values[col_idx] = [value]
                                                    #     #     elif value not in unique_values[col_idx]:
                                                    #     #         unique_values[col_idx].append(value)


    # rows = {col_idx: [unique_values[col_idx].index(row[col_idx]) for row in rows] for col_idx in unique_values}
    return sub, result_manager, rows #, rows_pca


def iter(args):
    sub, all_masks, shape, n_iterations, results, filename, num_workers = args

    # oads_dir = '/home/nmuller/projects/data/oads'
    device = 'cuda:0'
    result_dir = '/home/nmuller/projects/fmg_storage/oads_experiment_analysis/'
    encoding_model_dir = f'/home/nmuller/projects/oads_eeg_spatial_sampling/results/sub-{sub}/alexnet_imagenet/across-layers/feature-feature-full-1.0'
    load_features_from_file = False
    
    model_type = 'alexnet_imagenet'

    # if all_masks is None:
    #     folder = 'encoding_267x400_alexnet_share-pca_partial-corr_feature-cropping-AutoReject'
    # else:
    #     folder = 'encoding_alexnet_share-pca_partial-corr_feature-cropping-AutoReject'

    # results = [iterate_load_subject_data((sub, result_manager)) for sub in [sub]]
    _, pca, lin_reg = iterate_load_subject_data((sub, encoding_model_dir))


    eeg_dir = '/home/nmuller/projects/fmg_storage/osf_eeg_data/AutoReject'
    train_ids, _, train_data, _ = load_eeg_data(sub=sub, eeg_dir=eeg_dir)
    _, n_channels, n_timepoints = train_data.shape

    feature_dir = '/home/nmuller/projects/fmg_storage/TEST_feature_extraction'
    if load_features_from_file:
        # Load extracted features
        with open(os.path.join(feature_dir, 'activations.pkl'), 'rb') as f:
            activations = pickle.load(f)

    else:
        activations = extract_features(save_to_file=False, subjects=[sub], oads_dir='/home/nmuller/projects/data/oads', model_type=model_type, save_dir=feature_dir, device=device)

    # Divide extracted features into training and test sets
    activations = {
        layer_name: {
            image_index: feature for image_index, feature in activations[layer_name].items() if image_index in train_ids
        } for layer_name in activations.keys()
    }

    # test_activations = {
    #     layer_name: {
    #         image_index: feature for image_index, feature in activations[layer_name].items() if image_index in test_ids
    #     } for layer_name in activations.keys()
    # }

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
    
        # p = np.random.rand()
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


def load_and_iter(args):
    filename, sub, result_dir, num_workers = args

    if os.path.exists(os.path.join('/home/nmuller/projects/fmg_storage/oads_experiment_analysis/', str(sub), f'sub-{sub}_{filename}')):
        return
    
    results = pickle.load(open(os.path.join(result_dir, filename), 'rb'))
    all_masks = results['all_masks']
    shape = results['shape']
    n_iterations = len(all_masks)

    iter((sub, all_masks, shape, n_iterations, results, filename, num_workers))

def run_new_subjects():
    num_workers = 5 # nproc
    # for sub in range(5, 15):
    # for sub in range(15, 25):
    for sub in range(5, 36):
        results = None
        all_masks = None
        shape = None
        n_iterations = 1000

        iter((sub, all_masks, shape, n_iterations, results, None, num_workers))

# if __name__ == '__main__':
def run_other_subjects():
    # with open(os.path.join(result_dir, sub, f'random_patch_contribution_{n_patches}_{patch_size}x{patch_size}_so_no_center.pkl'), 'wb') as f:
    #     pickle.dump(results, f)
    names = [
        'random_patch_contribution_10_20x20_so_no_center.pkl',
        'random_patch_contribution_10_40x40_so_no_center.pkl',
        'random_patch_contribution_20_20x20_no_center.pkl',
        'random_patch_contribution_20_20x20_some_no_center.pkl',
        'random_patch_contribution_2000_some_no_center.pkl',
        'random_patch_contribution.pkl',
    ]

    result_dir = '/home/nmuller/projects/fmg_storage/oads_experiment_analysis/random_patch_contributions/sub-5'
    num_workers = 2 # nproc
    
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(load_and_iter, [(filename, sub, result_dir, num_workers) for sub in [12, 26] for filename in names]) # range(5, 36)

    # for filename in tqdm.tqdm(names, total=len(names), desc='Files'):
    #     results = pickle.load(open(os.path.join(result_dir, filename), 'rb'))
    #     all_masks = results['all_masks']
    #     shape = results['shape']
    #     n_iterations = len(all_masks)

    #     # for sub in tqdm.tqdm(range(5, 36), total=31, desc='Subjects'):
    #     #     # sub = 5

    #     #     # n_patches = 10
    #     #     # patch_size = 20
    #     #     # n_iterations = 5000

    #     #     iter((sub, all_masks, shape, n_iterations, results))

    #     subs = range(6, 36)
    #     if filename == 'random_patch_contribution_10_20x20_so_no_center.pkl':
    #         subs = [9] + list(range(11, 36))

    #     with multiprocessing.Pool(nproc) as pool:
    #         pool.map(iter, [(sub, all_masks, shape, n_iterations, results, filename) for sub in subs])


if __name__ == '__main__':
    run_other_subjects()

    # run_new_subjects()