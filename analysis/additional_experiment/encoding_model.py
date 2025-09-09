import h5py
import pandas as pd
import pickle
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import OADSImageDataset
from torchvision.models import alexnet
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from pytorch_utils.pytorch_utils import collate_fn, record_activations, ToRetinalGanglionCellSampling
from sklearn.decomposition import PCA
import tqdm
from sklearn.linear_model import LinearRegression
import rawpy
from PIL import Image

from eeg_data import load_eeg_channel_and_timepoints, load_eeg_filenames, load_eeg_data
from feature_extraction import extract_features

def center_size_image(img, shape=(1440, 2155, 3)):
    full_image = np.ones(np.array(shape), dtype=np.uint8) * np.mean(img).astype(np.uint8) 
    # past the current image in to the center of the full_image
    height, width, _ = np.array(img).shape
    center_x = full_image.shape[1] // 2
    center_y = full_image.shape[0] // 2
    start_x = center_x - width // 2
    start_y = center_y - height // 2
    full_image[start_y:start_y + height, start_x:start_x + width] = img

    return Image.fromarray(full_image)

class CenterSizeImage(object):
    def __init__(self, size=(1440, 2155, 3)):
        self.size = size

    def __call__(self, img):
        full_image = center_size_image(img, self.size)
        return full_image

def convert_to_df(sub, exp_condition, results):
    rows = []
    cols = ['subject', 'model_type', 'exp_condition', 'layer', 'n_pca_components', 'crop_condition', 'crop_instance', 'fraction', 'channel', 'channel_index', 'timepoint', 'timepoint_index', 'pred_index', 'metric', 'value']
    dtypes = {col: 'category' for col in cols if col != 'value'}

    # Retrieve information
    n_pca_components = results['n_components']
    model_type = results['model_type']
    layer = results['layer']
    crop_condition = results['crop_condition']
    fraction = results['fraction']
    crop_instance = results['crop_instance']

    eeg_dir = f'/home/nmuller/projects/fmg_storage/Data/sub_{sub}/Preprocessed epochs' # /sub_{sub}-OC&CSD-AutoReject-epo.fif
    channel_names, t = load_eeg_channel_and_timepoints(eeg_fif_dir=eeg_dir, sub=sub, exp_condition=exp_condition)

    # Loop over channels and timepoints and save results
    for channel in results['corr_channels'].keys():
        timepoints = range(len(results['corr_channels'][channel]))

        for timepoint in timepoints:
            r2_train = results['corr_channels'][channel][timepoint]
            rows.append([sub, model_type, exp_condition, layer, n_pca_components, crop_condition, crop_instance, fraction, channel_names[channel], channel, t[timepoint], timepoint, -1, 'corr_train', r2_train])
            
            test_corr_train = results['test_corr_channels'][channel][timepoint]
            rows.append([sub, model_type, exp_condition, layer, n_pca_components, crop_condition, crop_instance, fraction, channel_names[channel], channel, t[timepoint], timepoint, -1, 'test_corr_train', test_corr_train])

            for pred_index, pred in enumerate(results['test_pred_channels'][channel][timepoint]):
                rows.append([sub, model_type, exp_condition, layer, n_pca_components, crop_condition, crop_instance, fraction, channel_names[channel], channel, t[timepoint], timepoint, pred_index, 'test_pred_channels', pred])

    # Convert to DataFrame
    df = pd.DataFrame(data=rows, columns=cols)

    dtypes = {col: 'category' for col in cols if col != 'value'}
    dtypes['value'] = 'float32'
    for col in cols:
        df[col] = df[col].astype(dtypes[col]) # type: ignore

    return df

def iter(args):
    save_dir, layer_name, current_activations, train_data, test_data, n_channels, n_timepoints, n_components, model_type, sub, exp_condition, crop_condition, crop_instance, fraction = args

    pca: PCA
    pca = PCA(n_components=n_components)
    activations = pca.fit_transform(current_activations['train'])
    explained_variance = [pca.explained_variance_ratio_]

    design_matrix = np.hstack((np.ones((activations.shape[0], 1)), activations))


    comp_test_activations = pca.transform(current_activations['test'])
    test_design_matrix = np.hstack((np.ones((comp_test_activations.shape[0], 1)), comp_test_activations))

    lin_reg = LinearRegression(fit_intercept=False)

    mask = np.any(np.isnan(train_data), axis=(1,2))
    design_matrix = design_matrix[~mask]
    train_data = train_data[~mask]
    
    y = train_data.reshape(-1, n_channels * n_timepoints)

    lin_reg.fit(design_matrix, y)

    beta = lin_reg.coef_
    beta = beta.reshape(-1, n_channels, n_timepoints)

    test_mask = np.any(np.isnan(test_data), axis=(1,2))
    test_design_matrix = test_design_matrix[~test_mask]
    test_data = test_data[~test_mask]

    predictions = lin_reg.predict(design_matrix).reshape(-1, n_channels, n_timepoints)
    test_predictions = lin_reg.predict(test_design_matrix).reshape(-1, n_channels, n_timepoints)
    

    corr_channels = {}
    pred_channels = {}

    test_corr_channels = {}
    test_pred_channels = {}

    for channel in tqdm.tqdm(range(n_channels), total=n_channels):
        corrs = []
        preds = []

        test_corrs = []
        test_preds = []
        
        for timepoint in range(n_timepoints):
            ############# TRAINING #################
            pred = predictions[:, channel, timepoint]
            preds.append(pred)

            c = np.corrcoef(pred, train_data[:, channel, timepoint])[0, 1]
            corrs.append(c)

            ############# TESTING #################
            test_pred = test_predictions[:, channel, timepoint]
            test_preds.append(test_pred)

            test_c = np.corrcoef(test_pred, test_data[:, channel, timepoint])[0, 1]
            test_corrs.append(test_c)


        corr_channels[channel] = corrs
        pred_channels[channel] = preds
        
        test_corr_channels[channel] = test_corrs
        test_pred_channels[channel] = test_preds


    # save_dir = f'../../results/{"/400_pixels" if reduce_size else ""}/encoding{mode}{"" if eye_reject is None else "eye_reject-rejected" if eye_reject else "eye_reject_accepted"}_{model_type}_share-pca_partial-corr_feature-cropping{cleaning}_{exp_condition}'
    save_dir = save_dir.format(sub=sub, model_type=model_type, exp_condition=exp_condition, layer=layer_name, encoding_model=f'{crop_condition}-{crop_instance}-{fraction}')
    save_filename = os.path.join(save_dir, f'encoding_results_sub_{sub}_{layer_name}_{model_type}_{exp_condition}_feature-cropping_{crop_condition}-{crop_instance}-{fraction}.pkl')

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


    with open(save_filename, 'wb') as f:
        pickle.dump(results, f)

    # Convert to DataFrame
    df = convert_to_df(sub=sub, exp_condition=exp_condition, results=results)

    # # Save to parquet
    # filename = f'encoding_results_pca_{n_components}_sub_{sub}_{model_type}_feature-cropping_{layer_name}_{crop_condition}_{crop_instance}_{fraction}.parquet'
    df.to_parquet(os.path.join(save_filename.replace('.pkl', '.parquet')), index=False)

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
    
def main(result_dir):

    ############################################
    num_workers = 8
    
    gpu_name = 'cuda:1'
    device = torch.device(gpu_name if torch.cuda.is_available() else 'cpu')
    batch_size = 32 #512 # 512

    load_features_from_file = True
    model_type = 'alexnet_imagenet' # 'alexnet_imagenet'

    manual = True
    
    for eye_reject in [True, False]:

        mode = '_manual' if manual else ''
        
        for exp_condition in ['center1', 'center2', 'center3', 'peri1', 'peri2', 'peri3', 'size1', 'size2', 'size3']: 

            gcs = {}

            # eeg_dir = 'data_clean/base_dir_reordered/data_with_reject/'
            eeg_dir = f"data_SECOND_BATCH{mode}{'' if eye_reject is None else 'eye_reject-rejected' if eye_reject else 'eye_reject_accepted'}/dataclean/preprocessed_for_screening/data_with_reject/"
            
            print(eeg_dir)

            filenames_dir = 'data_clean/base_dir_reordered/data_with_reject/'
            train_ids, test_ids = load_eeg_filenames(eeg_dir=filenames_dir, exp_condition=exp_condition)

            for sub in [12, 15, 16, 17, 18, 19]:
                train_data, test_data = load_eeg_data(eeg_dir=eeg_dir, exp_condition=exp_condition, sub=sub)

                print(f'Train data: {train_data.shape}, Test data: {test_data.shape}')
                ####################


                print(len(train_ids), len(test_ids))

                # (3, 453, 1, 64, 511)
                _, n_channels, n_timepoints = train_data.shape
                sample_rate = 1024
                t = [i/sample_rate - 0.1 for i in range(n_timepoints)]

                if load_features_from_file:
                    feature_dir = '/home/nmuller/projects/fmg_storage/TEST_feature_extraction'

                    with h5py.File(os.path.join(feature_dir, 'additional_activations.h5'), 'r') as activations:
                        # Divide extracted features into training and test sets
                        train_activations = {
                            layer_name: {
                                image_index: feature[()] for image_index, feature in activations[layer_name].items() if image_index in train_ids
                            } for layer_name in activations.keys()
                        }

                        test_activations = {
                            layer_name: {
                                image_index: feature[()] for image_index, feature in activations[layer_name].items() if image_index in test_ids
                            } for layer_name in activations.keys()
                        }
                else:
                    activations = extract_features(model_type=model_type, oads_dir='oads_resolution_experiment/stimuli', fileending='.png', num_workers=num_workers)

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


                for n_components in [100]:
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


                                    if layer_name not in gcs:
                                        gcs[layer_name] = ToRetinalGanglionCellSampling(image_shape=out_size + ((1,)), out_size=_max, series=1, dtype=np.float32)

                                    gcs_output = gcs[layer_name](feature).flatten()

                                    outputs[split_name][layer_name][image_index][feature_index] = {
                                        'feature': {1.0: {'feature-full': feature}},
                                        'gcs': {1.0: {'gcs-full': gcs_output}},
                                    }


                    for crop_condition in tqdm.tqdm(outputs['train'][base_layer_name][base_image_index][0].keys(), total=len(outputs['train'][base_layer_name][base_image_index][0].keys())):
                        for fraction in outputs['train'][base_layer_name][base_image_index][0][crop_condition].keys():
                            
                            crop_instances = [x for x in outputs['train'][base_layer_name][base_image_index][0][crop_condition][fraction].keys() if 'config' not in x]
                            for crop_instance in crop_instances:
                                
                                all_activations = {
                                    'train': None, 
                                    'test': None,
                                    }
                                # layer_activations = {
                                #     'train': {},
                                #     'test': {},
                                #     }
                                
                                for split_name in all_activations.keys():
                                    for layer in outputs[split_name].keys():
                                        # Flatten all features (columns) per image (rows)
                                        x = np.array([np.array([outputs[split_name][layer][image_index][feature_index][crop_condition][fraction][crop_instance]
                                                                for feature_index in outputs[split_name][layer][image_index].keys()]).flatten()
                                                                for image_index in outputs[split_name][layer].keys()]
                                                )
                                        
                                        
                                        if all_activations[split_name] is None:
                                            all_activations[split_name] = x
                                        else:
                                            all_activations[split_name] = np.hstack((all_activations[split_name], x))

                                        # layer_activations[split_name][layer] = x

                                # activation_pairs = [(layer_name, {split: layer_activations[split][layer_name] for split in layer_activations.keys()}) for layer_name in layer_activations[list(layer_activations.keys())[0]].keys()]
                                activation_pairs = []
                                activation_pairs.append(('across-layers', all_activations))
                                
                                for layer_name, current_activations in activation_pairs:
                                    iter((result_dir, layer_name, current_activations, train_data, test_data, n_channels, n_timepoints, n_components, model_type, sub, exp_condition, crop_condition, crop_instance, fraction))



if __name__ == '__main__':
    result_dir = '../TEST_results/additional_experiment/{exp_condition}/sub-{sub}/{model_type}/{layer}/{encoding_model}'

    # Since all subjects have seen the same images, we are not looping over subjects here, but instead load the CNN features once and then iterate over subjects
    main(result_dir)