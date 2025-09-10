import h5py
import os
import pickle
import numpy as np
import tqdm
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy.stats import zscore

from GDS import ToRetinalGanglionCellSampling

from feature_extraction import extract_features
from eeg_data import load_eeg_data, load_eeg_channel_and_timepoints


def convert_to_df(sub, results):
    rows = []
    cols = ['subject', 'model_type', 'layer', 'n_pca_components', 'crop_condition', 'crop_instance', 'fraction', 'channel', 'channel_index', 'timepoint', 'timepoint_index', 'pred_index', 'metric', 'value']
    dtypes = {col: 'category' for col in cols if col != 'value'}

    # Retrieve information
    n_pca_components = results['n_components']
    model_type = results['model_type']
    layer = results['layer']
    crop_condition = results['crop_condition']
    fraction = results['fraction']
    crop_instance = results['crop_instance']

    channel_names, t = load_eeg_channel_and_timepoints()

    # Loop over channels and timepoints and save results
    for channel in results['corr_channels'].keys():
        timepoints = range(len(results['corr_channels'][channel]))

        for timepoint in timepoints:
            r2_train = results['corr_channels'][channel][timepoint]
            rows.append([sub, model_type, layer, n_pca_components, crop_condition, crop_instance, fraction, channel_names[channel], channel, t[timepoint], timepoint, -1, 'corr_train', r2_train])
            
            test_corr_train = results['test_corr_channels'][channel][timepoint]
            rows.append([sub, model_type, layer, n_pca_components, crop_condition, crop_instance, fraction, channel_names[channel], channel, t[timepoint], timepoint, -1, 'test_corr_train', test_corr_train])

            for pred_index, pred in enumerate(results['test_pred_channels'][channel][timepoint]):
                rows.append([sub, model_type, layer, n_pca_components, crop_condition, crop_instance, fraction, channel_names[channel], channel, t[timepoint], timepoint, pred_index, 'test_pred_channels', pred])

    # Convert to DataFrame
    df = pd.DataFrame(data=rows, columns=cols)

    dtypes = {col: 'category' for col in cols if col != 'value'}
    dtypes['value'] = 'float32'
    for col in cols:
        df[col] = df[col].astype(dtypes[col]) # type: ignore

    return df

def iter(args):
    save_dir, layer_name, current_activations, train_data, test_data, n_channels, n_timepoints, n_components, model_type, sub, crop_condition, crop_instance, fraction = args

    # Create Design Matrix for training set
    pca = PCA(n_components=n_components)
    activations = pca.fit_transform(current_activations['train'])
    explained_variance = [pca.explained_variance_ratio_]

    design_matrix = np.hstack((np.ones((activations.shape[0], 1)), activations))

    # Create Design Matrix for test set
    comp_test_activations = pca.transform(current_activations['test'])
    test_design_matrix = np.hstack((np.ones((comp_test_activations.shape[0], 1)), comp_test_activations))

    lin_reg = LinearRegression(fit_intercept=False)

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
    # save_dir = f'{save_dir}/encoding_{model_type}_feature-cropping'
    # '../../results/sub-{sub}/{layer}/{encoding_model}/{filename}'
    save_dir = save_dir.format(sub=sub, model_type=model_type, layer=layer_name, encoding_model=f'{crop_condition}-{crop_instance}-{fraction}')
    os.makedirs(save_dir, exist_ok=True)
    
    save_filename = os.path.join(save_dir, f'encoding_results_sub_{sub}_{layer_name}_{model_type}-feature-cropping-{crop_condition}-{crop_instance}-{fraction}.pkl')

    results = {
        'sub': sub,
        'model_type': model_type,

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

    # Save to pickle
    # filename = f'encoding_results_pca_{n_components}_sub_{sub}_{model_type}_feature-cropping_{layer_name}_{crop_condition}_{crop_instance}_{fraction}.pkl'
    with open(os.path.join(save_filename), 'wb') as f:
        pickle.dump(results, f)

    # Convert to DataFrame
    df = convert_to_df(sub, results)

    # # Save to parquet
    # filename = f'encoding_results_pca_{n_components}_sub_{sub}_{model_type}_feature-cropping_{layer_name}_{crop_condition}_{crop_instance}_{fraction}.parquet'
    df.to_parquet(os.path.join(save_filename.replace('.pkl', '.parquet')), index=False)

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
    
def run_regression(sub, eeg_dir, result_dir, load_features_from_file:bool=True):

    device = 'cuda:1'
    
    # Fitting specs
    n_components = 100

    ################ Load EEG data and filenames
    train_ids, test_ids, train_data, test_data = load_eeg_data(sub=sub, eeg_dir=eeg_dir)

    _, n_channels, n_timepoints = train_data.shape
    # sample_rate = 1024
    # t = [i/sample_rate - 0.1 for i in range(n_timepoints)]


    ################ Load features
    model_type = 'alexnet_imagenet'
    feature_dir = '../../dnn_features'

    if load_features_from_file:

        activations = np.load(os.path.join(feature_dir, f'main_experiment_{model_type}_activations.npz'), allow_pickle=True)
        
        with open(os.path.join(feature_dir, f'main_experiment_image_id_order.pkl'), 'rb') as f:
            image_id_order = pickle.load(f)

        train_id_indices = [image_id_order.index(x) for x in train_ids if x in image_id_order]
        test_id_indices = [image_id_order.index(x) for x in test_ids if x in image_id_order]

        train_activations = {
            layer_name: {
                image_index: activations[layer_name].item().get(image_index) for image_index in train_id_indices
            } for layer_name in activations.keys()
        }
        
        test_activations = {
            layer_name: {
                image_index: activations[layer_name].item().get(image_index) for image_index in test_id_indices
            } for layer_name in activations.keys()
        }

    else:
        activations = extract_features(save_to_file=False, subjects=[sub], oads_dir='../../stimuli', model_type=model_type, save_dir=feature_dir, device=device)

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
    
    gcs = {} # Only need to compute the GCS transform matrix once per layer
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
                    out_size = (_max, _max, 1)

                    ##################### Spatial sampling methods

                    # (a) Convert to GCS
                    if layer_name not in gcs:
                        gcs[layer_name] = ToRetinalGanglionCellSampling(image_shape=out_size, out_size=_max, series=1, dtype=np.float32) # type: ignore


                    gcs_output = gcs[layer_name](feature).flatten()

                    # (b) cropping
                    fraction_outputs = {}

                    # Increasing crop sizes
                    # for fraction in [0.005, 0.01, 0.05, 0.1, 0.2]: # This refers to the percentage of the area of the feature map
                    for fraction in [0.005]: # This refers to the percentage of the area of the feature map

                        # (b.1) Circular mask
                        small_circ_mask = get_circular_mask(shape, fraction)
                        large_circ_mask = get_circular_mask(shape, fraction+0.15)
                        intersect_circ_mask = np.where(small_circ_mask, 0, large_circ_mask).astype(bool)

                        intersect_circ_flat = feature[intersect_circ_mask.astype(bool)]
                        center_circ_flat = feature[small_circ_mask.astype(bool)]
                        periphery_circ_flat = feature[~small_circ_mask.astype(bool)]

                        # # (b.2) Rectangular mask
                        # mask = get_rectangular_mask(shape, fraction)

                        # center_fraction = fraction
                        # out_fraction = center_fraction + 0.15
                        # mask_small = get_rectangular_mask(shape, center_fraction)
                        # mask_large = get_rectangular_mask(shape, center_fraction + out_fraction)
                        # mask_intersect = np.where(mask_small, 0, mask_large).astype(bool)

                        # center_flat = feature[mask]
                        # periphery_flat = feature[~mask]
                        # intersect_flat = feature[mask_intersect]
                        # ####################################

                        fraction_outputs[fraction] = {
                            # 'center': center_flat, 
                            # 'periphery': periphery_flat, 
                            # 'intersect_flat': intersect_flat, 
                            'center_circ': center_circ_flat,
                            'periphery_circ': periphery_circ_flat,
                            # 'intersect_circ': intersect_circ_flat,
                        }

                    # Save spatially sampled outputs
                    outputs[split_name][layer_name][image_index][feature_index] = {
                        'feature': {1.0: {'feature-full': feature}},
                        'fraction': fraction_outputs,
                        'gcs': {1.0: {'gcs-full': gcs_output}},
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
                            all_activations[split_name] = x # type: ignore
                        else:
                            all_activations[split_name] = np.hstack((all_activations[split_name], x)) # type: ignore

                        # Save per layer
                        layer_activations[split_name][layer] = x

                activation_pairs = [(layer_name, {split: layer_activations[split][layer_name] for split in layer_activations.keys()}) for layer_name in layer_activations[list(layer_activations.keys())[0]].keys()]
                activation_pairs.append(('across-layers', all_activations))
                

                # Loop over layers - Regression Model
                for layer_name, current_activations in activation_pairs:
                    iter((result_dir, layer_name, current_activations, train_data, test_data, n_channels, n_timepoints, n_components, model_type, sub, crop_condition, crop_instance, fraction))
                
                # with multiprocessing.Pool(nproc) as pool:
                #     pool.map(iter, [(layer_name, current_activations, train_data, test_data, n_channels, n_timepoints, n_components, model_type, sub, crop_condition, crop_instance, fraction) for layer_name, current_activations in activation_pairs])


if __name__ == '__main__':
    load_features_from_file = False
    eeg_dir = '../../eeg_data/main_experiment'

    for sub in range(5, 36):
        result_dir = '../../results/sub-{sub}/{model_type}/{layer}/{encoding_model}'

        run_regression(sub=sub, eeg_dir=eeg_dir, result_dir=result_dir, load_features_from_file=load_features_from_file)