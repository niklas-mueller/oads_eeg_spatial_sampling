import os
import pickle
import numpy as np
import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from scipy.stats import zscore
from feature_extraction import extract_features
from GDS import ToRetinalGanglionCellSampling

def convert_to_df(sub, results, folder):
    rows = []
    cols = ['folder', 'subject', 'model_type', 'layer', 'image_representation', 'image_quality', 'n_pca_components', 'resolution', 'crop_condition', 'crop_instance', 'fraction', 'channel', 'channel_index', 'timepoint', 'timepoint_index', 'pred_index', 'metric', 'value']
    dtypes = {col: 'category' for col in cols if col != 'value'}

    # Retrieve information
    image_representation = results['image_representation']
    image_quality = results['image_quality']
    n_pca_components = results['n_components']
    model_type = results['model_type']
    layer = results['layer']
    image_resolution = '400'
    crop_condition = results['crop_condition']
    fraction = results['fraction']
    crop_instance = results['crop_instance']

    # Set EEG properties
    sample_rate = 1024
    t = [i/sample_rate - 0.1 for i in range(513)]
    timepoints = None

    # Loop over channels and timepoints and save results
    for channel in results['corr_channels'].keys():
        if timepoints is None:
            timepoints = range(len(results['corr_channels'][channel]))

        for timepoint in timepoints:
            r2_train = results['corr_channels'][channel][timepoint]
            rows.append([folder, sub, model_type, layer, image_representation, image_quality, n_pca_components, image_resolution, crop_condition, crop_instance, fraction, channel_names[channel], channel, t[timepoint], timepoint, -1, 'corr_train', r2_train])
            
            test_corr_train = results['test_corr_channels'][channel][timepoint]
            rows.append([folder, sub, model_type, layer, image_representation, image_quality, n_pca_components, image_resolution, crop_condition, crop_instance, fraction, channel_names[channel], channel, t[timepoint], timepoint, -1, 'test_corr_train', test_corr_train])

    # Convert to DataFrame
    df = pd.DataFrame(data=rows, columns=cols)

    dtypes = {col: 'category' for col in cols if col != 'value'}
    dtypes['value'] = 'float32'
    for col in cols:
        df[col] = df[col].astype(dtypes[col])

    return df

def iter(args):
    layer_name, current_activations, train_data, test_data, n_channels, n_timepoints, use_pls_regression, n_components, home_path, model_type, sub, image_representation, image_quality, crop_condition, crop_instance, fraction = args

    # Create Design Matrix for training set
    if not use_pls_regression:                                   
        pca: PCA
        pca = PCA(n_components=n_components)
        activations = pca.fit_transform(current_activations['train'])
        explained_variance = [pca.explained_variance_ratio_]

        design_matrix = np.hstack((np.ones((activations.shape[0], 1)), activations))
    else:
        design_matrix = np.hstack((np.ones((current_activations['train'].shape[0], 1)), current_activations['train']))

    # Create Design Matrix for test set
    if not use_pls_regression:
        comp_test_activations = pca.transform(current_activations['test'])
        test_design_matrix = np.hstack((np.ones((comp_test_activations.shape[0], 1)), comp_test_activations))
    else:
        test_design_matrix = np.hstack((np.ones((current_activations['test'].shape[0], 1)), current_activations['test']))

    if use_pls_regression:
        lin_reg = PLSRegression(n_components=2, scale=False)
    else:
        lin_reg = LinearRegression(fit_intercept=False)

    x = design_matrix

    # Standardize EEG data
    y = zscore(train_data.reshape(-1, n_channels * n_timepoints), axis=0)

    # Fit the regression model
    lin_reg.fit(design_matrix, y)

    # Extract beta weights
    beta = lin_reg.coef_

    # Reshape beta weights for each channel and timepoint
    beta = beta.reshape(-1, n_channels, n_timepoints)

    # Predict EEG data for training and test set
    predictions = lin_reg.predict(design_matrix).reshape(-1, n_channels, n_timepoints)
    test_predictions = lin_reg.predict(test_design_matrix).reshape(-1, n_channels, n_timepoints)
    

    corr_channels = {}
    pred_channels = {}

    test_corr_channels = {}
    test_pred_channels = {}

    # Calculate correlation between predicted and actual EEG data for each channel and timepoint
    for channel in tqdm.tqdm(range(n_channels), total=n_channels):
        corrs = []
        preds = []

        test_corrs = []
        test_preds = []
        
        for timepoint in range(n_timepoints):
            # Train set
            pred = predictions[:, channel, timepoint]
            preds.append(pred)

            c = np.corrcoef(pred, train_data[:, channel, timepoint])[0, 1]
            corrs.append(c)

            # Test set
            test_pred = test_predictions[:, channel, timepoint]
            test_preds.append(test_pred)

            test_c = np.corrcoef(test_pred, test_data[:, channel, timepoint])[0, 1]
            test_corrs.append(test_c)


        corr_channels[channel] = corrs
        pred_channels[channel] = preds
        
        test_corr_channels[channel] = test_corrs
        test_pred_channels[channel] = test_preds

    # Save results
    save_dir = f'{home_path}/projects/fmg_storage/oads_experiment_analysis/correct_size_new_fit/{"PLS_" if use_pls_regression else ""}encoding_{model_type}_share-pca_partial-corr_feature-cropping{cleaning}'
    os.makedirs(save_dir, exist_ok=True)

    results = {
        'sub': sub,
        'model_type': model_type,
        'image_representation': image_representation,
        'image_quality': image_quality,

        'layer': layer_name,
        'crop_condition': crop_condition,
        'crop_instance': crop_instance,
        'fraction': fraction,

        'lin_reg': lin_reg,
        'beta': beta,
        'corr_channels': corr_channels,
        'test_corr_channels': test_corr_channels,
        'design_matrix': design_matrix,
        'test_design_matrix': test_design_matrix,
        'n_components': n_components,
        'explained_variance': explained_variance,

        'pca': pca,
        'pred_channels': pred_channels,
        'test_pred_channels': test_pred_channels,
        }

    # # Save to pickle
    # filename = f'{"PLS_" if use_pls_regression else ""}encoding_results_pca_{n_components}_sub_{sub}_{model_type}_feature-cropping_{layer_name}_{image_representation}_{image_quality}_{image_resolution}_{crop_condition}_{crop_instance}_{fraction}.pkl'
    # with open(os.path.join(save_dir, filename), 'wb') as f:
    #     pickle.dump(results, f)

    # Convert to DataFrame
    df = convert_to_df(sub, results, folder=save_dir.split('/')[-1])

    # # Save to parquet
    # filename = f'{"PLS_" if use_pls_regression else ""}encoding_results_pca_{n_components}_sub_{sub}_{model_type}_feature-cropping_{layer_name}_{image_representation}_{image_quality}_{image_resolution}_{crop_condition}_{crop_instance}_{fraction}.parquet'
    # df.to_parquet(os.path.join(save_dir, filename))

def get_rectangular_mask(shape, fraction):
    _fraction = np.sqrt(fraction)

    mask = np.zeros(shape).astype(bool)

    row_center = shape[0] / 2
    row_start = int(row_center - row_center * _fraction)
    row_end = int(row_center + row_center * _fraction)
    row_size = row_end - row_start

    col_center = shape[1] / 2
    col_start = int(col_center - col_center * _fraction)
    col_end = int(col_center + col_center * _fraction)
    col_size = col_end - col_start

    mask[row_start:row_end, col_start:col_end] = True

    return mask

def get_circular_mask(size, center_fraction):

    # Calculate the area of the mask
    mask_area = size[0] * size[1]

    # Calculate the center size based on the percentage of the area
    center_area = mask_area * center_fraction

    # Calculate the radius of the circular center
    center_radius = int(np.sqrt(center_area / np.pi))

    # # Calculate the padding needed to center the circular center within the mask
    # padding = (size - center_radius) // 2

    # Create the mask
    mask = np.zeros((size[0], size[1]), dtype=bool)
    y, x = np.ogrid[:size[0], :size[1]]
    mask[((x - size[1] // 2) ** 2 + (y - size[0] // 2) ** 2) < center_radius ** 2] = True

    return mask
    
# if __name__ == '__main__':
def run_regression(sub, load_features_from_file:bool=True):
    
    # Paths
    home_path = os.path.expanduser('~')
    basedir = f'{home_path}/projects/data/oads'
    eeg_dir = '/home/nmuller/projects/fmg_storage/osf_eeg_data'

    # Fitting specs
    n_components = 100

    

    gcs = {}
    gcs_inverse = {}

    target_filenames = ['0434262454981c92.tiff', '0c6c2c66e61e3133.tiff', '0e2e2f2931313d37.tiff', '1332a2d4c4c14322.tiff', '202024d9b333968c.tiff', '24247432670b3131.tiff', '2c2064d163c52c78.tiff', '42431879e191d3c3.tiff', '61662ece1f6e7870.tiff', '7890918716766325.tiff', '85cc4e533959b1e1.tiff', '8ece4062eee68292.tiff', '93939b9a101e87e0.tiff', '948cacbc94b42474.tiff', 'af8f3a3939313171.tiff', 'b23232b232332361.tiff', 'b371f8ecf4e0f0f0.tiff', 'bce2d2d2c393e3b2.tiff', 'c77131717179f173.tiff', 'c93b3b2b29b9b8f1.tiff', 'cc8c9290bcbcf4fc.tiff', 'dad8acb8b82e36b1.tiff', 'e2f8f8fcf8f8f8f8.tiff', 'e6d8d40438c0c8e2.tiff', 'e6e648c62ebacc38.tiff', 'ef3632c2c1476e78.tiff', 'f4ca83dc6731310d.tiff']

    # Load EEG data and filenames
    eeg_dir = '/home/nmuller/projects/fmg_storage/osf_eeg_data'
    with open(os.path.join(eeg_dir, f'filenames_oads_eeg_rsvp_sub-{str(sub).zfill(2)}.pkl'), 'rb') as f:
        filenames = pickle.load(f)

    with open(os.path.join(eeg_dir, f'is_test_oads_eeg_rsvp_sub-{str(sub).zfill(2)}.pkl'), 'rb') as f:
        is_test = pickle.load(f)

    data = np.load(os.path.join(eeg_dir, f'oads_eeg_rsvp_sub-{str(sub).zfill(2)}.npy'))
    
    train_filenames = [filenames[i] for i in range(len(filenames)) if not is_test[i]]
    test_filenames = [filenames[i] for i in range(len(filenames)) if is_test[i]]

    train_ids = [x.split('.')[0] for x in train_filenames if f"{x.split('.')[0]}.tiff" not in target_filenames]
    test_ids = [x.split('.')[0] for x in test_filenames if f"{x.split('.')[0]}.tiff" not in target_filenames]

    print(len(train_ids), len(test_ids))

    train_data = np.array([data[i] for i in range(len(data)) if not is_test[i]])
    test_data = np.array([data[i] for i in range(len(data)) if is_test[i]])

    _, n_channels, n_timepoints = train_data.shape
    sample_rate = 1024
    t = [i/sample_rate - 0.1 for i in range(n_timepoints)]

    if load_features_from_file:
        # Load extracted features
        with open('/home/nmuller/projects/fmg_storage/TEST_feature_extraction/activations.pkl', 'rb') as f:
            activations = pickle.load(f)

    else:
        activations = extract_features(save_to_file=False, subjects=[sub])

    # Divide extracted features into training and test sets
    train_activations = {
        layer_name: {
            image_index: feature for image_index, feature in activations[layer_name].items() if image_index in train_ids
        } for layer_name in activations.keys()
    }

    test_activations = {
        layer_name: {
            image_index: feature for image_index, feature in activations[layer_name].items() if image_index in test_ids
        } for layer_name in activations.keys()
    }
    

    
    base_layer_name = list(train_activations.keys())[0] if len(train_activations) > 0 else list(test_activations.keys())[0] # f'{model_type}_feature'
    base_image_index = list(train_activations[base_layer_name].keys())[0] if len(train_activations[base_layer_name]) > 0 else list(test_activations[base_layer_name].keys())[0]

    # n_train_images = len(train_activations[base_layer_name])
    # n_test_images = len(test_activations[base_layer_name])

    # print(f'Processing {sub} with {n_train_images} training images and {n_test_images} test images')
    # exit(1)
    
    outputs = {}
    for split_name, _activations in [('train', train_activations), ('test', test_activations)]:
        outputs[split_name] = {}
        for layer_name, layer in _activations.items():
            outputs[split_name][layer_name] = {}

            shape = None
            for image_index, image in layer.items():
                outputs[split_name][layer_name][image_index] = {}

                for feature_index, feature in enumerate(image):
                    if shape is None:
                        shape = feature.shape
                        _max = max(shape)
                        out_size = (_max, _max)

                    ##################### Spatial sampling methods

                    # (a) Convert to GCS
                    if layer_name not in gcs:
                        gcs[layer_name] = ToRetinalGanglionCellSampling(image_shape=out_size + ((1,)), out_size=_max, series=1, dtype=np.float32)

                    if layer_name not in gcs_inverse:
                        gcs_inverse[layer_name] = ToRetinalGanglionCellSampling(image_shape=out_size + ((1,)), out_size=_max, series=1, dtype=np.float32, decomp=1)

                    gcs_output = gcs[layer_name](feature).flatten()
                    gcs_inverse_output = gcs_inverse[layer_name](feature).flatten()

                    # (b) cropping
                    fraction_outputs = {}

                    # Increasing crop sizes
                    for fraction in [0.005, 0.01, 0.05, 0.1, 0.2]: # This refers to the percentage of the area of the feature map
                        if shape is None:
                            print(f'No shape for {layer_name} {image_index} {feature_index}')                                            

                        # (b.1) Circular mask
                        small_circ_mask = get_circular_mask(shape, fraction)
                        large_circ_mask = get_circular_mask(shape, fraction+0.15)

                        intersect_circ_mask = np.where(small_circ_mask, 0, large_circ_mask).astype(bool)

                        intersect_circ_flat = feature[intersect_circ_mask.astype(bool)]
                        center_circ_flat = feature[small_circ_mask.astype(bool)]
                        periphery_circ_flat = feature[~small_circ_mask.astype(bool)]

                        # (b.2) Rectangular mask
                        mask = get_rectangular_mask(shape, fraction)

                        center_fraction = fraction
                        mask_small = get_rectangular_mask(shape, center_fraction)

                        out_fraction = center_fraction + 0.15
                        mask_large = get_rectangular_mask(shape, center_fraction + out_fraction)

                        mask_intersect = np.where(mask_small, 0, mask_large).astype(bool)

                        center_flat = feature[mask]
                        periphery_flat = feature[~mask]
                        intersect_flat = feature[mask_intersect]
                        # ####################################

                        fraction_outputs[fraction] = {
                            'center': center_flat, 
                            'periphery': periphery_flat, 
                            'intersect_flat': intersect_flat, 
                            'center_circ': center_circ_flat,
                            'periphery_circ': periphery_circ_flat,
                            'intersect_circ': intersect_circ_flat,
                            # 'config': {
                            #     'mask': mask, 
                            #     # 'row_size': row_size, 'col_size': col_size,
                            #     'mask_intersect': mask_intersect, 
                            # },
                        }

                    # Save spatially sampled outputs
                    outputs[split_name][layer_name][image_index][feature_index] = {
                        'feature': {1.0: {'feature-full': feature}},
                        'fraction': fraction_outputs,
                        'gcs': {1.0: {'gcs-full': gcs_output}},
                        'gcs_inverse': {1.0: {'gcs-full': gcs_inverse_output}},
                    }

    # Loop over spatial sampling methods
    for crop_condition in tqdm.tqdm(outputs['train'][base_layer_name][base_image_index][0].keys(), total=len(outputs['train'][base_layer_name][base_image_index][0].keys())):

        # Loop over sizes
        for fraction in outputs['train'][base_layer_name][base_image_index][0][crop_condition].keys():
            
            crop_instances = [x for x in outputs['train'][base_layer_name][base_image_index][0][crop_condition][fraction].keys() if 'config' not in x]

            for crop_instance in crop_instances:
                
                # Collect all activations per layer and across all layers
                all_activations = {
                    'train': None, 
                    'test': None,
                    }
                layer_activations = {
                    'train': {},
                    'test': {},
                    }
                
                # Loop over training and test set
                for split_name in all_activations.keys():

                    # Loop over layers
                    for layer in outputs[split_name].keys():
                        
                        # Flatten all features (columns) per image (rows)
                        x = np.array([np.array([outputs[split_name][layer][image_index][feature_index][crop_condition][fraction][crop_instance]
                                                for feature_index in outputs[split_name][layer][image_index].keys()]).flatten()
                                                for image_index in outputs[split_name][layer].keys()]
                                )
                        
                        # Save across all layers
                        if all_activations[split_name] is None:
                            all_activations[split_name] = x
                        else:
                            all_activations[split_name] = np.hstack((all_activations[split_name], x))

                        # Save per layer
                        layer_activations[split_name][layer] = x

                activation_pairs = [(layer_name, {split: layer_activations[split][layer_name] for split in layer_activations.keys()}) for layer_name in layer_activations[list(layer_activations.keys())[0]].keys()]
                activation_pairs.append(('across-layers', all_activations))
                

                # Loop over layers - Regression Model
                for layer_name, current_activations in activation_pairs:
                    iter((layer_name, current_activations, train_data, test_data, n_channels, n_timepoints, use_pls_regression, n_components, home_path, model_type, sub, image_representation, image_quality, crop_condition, crop_instance, fraction))
                
                # with multiprocessing.Pool(nproc) as pool:
                #     pool.map(iter, [(layer_name, current_activations, train_data, test_data, n_channels, n_timepoints, use_pls_regression, n_components, home_path, model_type, sub, image_representation, image_quality, crop_condition, crop_instance, fraction) for layer_name, current_activations in activation_pairs])


if __name__ == '__main__':
    load_features_from_file = False

    for sub in range(5,6):
        run_regression(sub, load_features_from_file)