from eeg_data import load_eeg_channel_and_timepoints, load_eeg_data
from feature_extraction import extract_features
from encoding_model import convert_to_df
from utils import CustomOADS

from sklearn.decomposition import PCA
import torch
import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil

# Get optimal number of CPU cores (leave some for system)
nproc = max(1, psutil.cpu_count(logical=False) - 2)  # type: ignore # Physical cores minus 2 for system
logical_cores = psutil.cpu_count(logical=True)

# Set threading environment variables for optimal CPU performance
os.environ["OMP_NUM_THREADS"] = str(nproc)
os.environ["OPENBLAS_NUM_THREADS"] = str(nproc)
os.environ["MKL_NUM_THREADS"] = str(nproc)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(nproc)
os.environ["NUMEXPR_NUM_THREADS"] = str(nproc)

import numpy as np

import tqdm
from scipy.stats import zscore
import pickle
from time import time
from PIL import Image

# Configure PyTorch for CPU optimization
torch.set_num_threads(nproc)
torch.set_num_interop_threads(max(1, nproc // 4))


def fit_linear_regression_batch(X, Y, add_bias=False):
    """Vectorized linear regression for multiple targets - CPU optimized"""
    if add_bias:
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    else:
        X_bias = X
    
    # Use solve instead of pinv for better numerical stability and speed
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    # Use regularization for numerical stability
    reg_term = 1e-8 * np.eye(X_bias.shape[1])
    coef = np.linalg.solve(X_bias.T @ X_bias + reg_term, X_bias.T @ Y)
    return coef

def predict_linear_regression_batch(X, coef, add_bias=False):
    """Vectorized prediction for multiple targets"""
    if add_bias:
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    else:
        X_bias = X
    return X_bias @ coef

def precompute_resized_contributions(contribution, shape_dict):
    """CPU-optimized precomputation of resized contributions"""
    n_channels, n_timepoints = len(contribution), len(contribution[0])
    resized_contribs = {}
    
    # print("Precomputing resized contributions on CPU...")
    
    # Use parallel processing for resizing operations
    def resize_contribution(args):
        channel, timepoint, layer, resize_size = args
        resized = Image.fromarray(contribution[channel][timepoint]).resize(
            resize_size, Image.Resampling.BILINEAR)
        return (channel, timepoint, layer, np.array(resized, dtype=np.float32))
    
    # Create all resize tasks
    resize_tasks = []
    for channel in range(n_channels):
        for timepoint in range(n_timepoints):
            for layer, resize_size in shape_dict.items():
                resize_tasks.append((channel, timepoint, layer, resize_size))
    
    # Process resizing in parallel
    with ThreadPoolExecutor(max_workers=min(nproc, 8)) as executor:
        results = list(tqdm.tqdm(
            executor.map(resize_contribution, resize_tasks),
            total=len(resize_tasks),
            desc="Resizing contributions"
        ))
    
    # Organize results
    for channel, timepoint, layer, resized_data in results:
        if (channel, timepoint) not in resized_contribs:
            resized_contribs[(channel, timepoint)] = {}
        resized_contribs[(channel, timepoint)][layer] = resized_data
    
    return resized_contribs

def spatially_weight_activations(outputs, contribution, total_dim, n_images, 
                                   layer_names, layer_offsets, batch_size):
    """CPU-optimized spatial weighting with efficient memory usage"""
    split_names = list(outputs.keys())
    
    # Pre-allocate output arrays
    weighted = {}
    for split_name in split_names:
        weighted[split_name] = np.zeros((n_images[split_name], total_dim), dtype=np.float32)
    
    # Determine optimal batch size based on available memory
    available_memory = psutil.virtual_memory().available
    # Conservative estimate: each sample needs ~4 bytes per feature
    max_batch_size = min(batch_size, available_memory // (total_dim * 4 * 4))  # Safety factor
    safe_batch_size = max(1, min(max_batch_size, 32))
    
    print(f"Using batch size: {safe_batch_size} (available memory: {available_memory // 1e9:.1f}GB)")
    
    # Process each layer
    for layer in tqdm.tqdm(layer_names, desc="Processing layers"):
        contrib_map = contribution[layer]
        start_idx, end_idx = layer_offsets[layer]
        
        for split_name in split_names:
            layer_outputs = outputs[split_name][layer]
            n_img = len(layer_outputs)
            
            # Process in batches with CPU optimization
            for b_start in range(0, n_img, safe_batch_size):
                b_end = min(b_start + safe_batch_size, n_img)
                
                # Convert to numpy if needed and ensure CPU processing
                if isinstance(layer_outputs, torch.Tensor):
                    batch_data = layer_outputs[b_start:b_end].cpu().numpy()
                else:
                    batch_data = layer_outputs[b_start:b_end]
                
                # Efficient element-wise multiplication and flattening
                # Use einsum for optimal performance
                if len(batch_data.shape) == 4:  # (batch, channels, height, width)
                    weighted_batch = np.einsum('bchw,hw->bchw', batch_data, contrib_map)
                    weighted_batch = weighted_batch.reshape(b_end - b_start, -1)
                else:
                    # Fallback for other shapes
                    weighted_batch = (batch_data * contrib_map).reshape(b_end - b_start, -1)
                
                # Store results
                weighted[split_name][b_start:b_end, start_idx:end_idx] = weighted_batch
    
    return weighted

def process_timepoint_batch(args):
    """CPU-optimized timepoint batch processing"""
    timepoint_batch, channel, resized_contribs, outputs, pca, n_components, train_data, test_data, \
    total_dim, n_images, layer_names, layer_offsets, batch_size = args
    
    results = {'corrs': [], 'preds': [],
               'test_corrs': [], 'test_preds': []}
    
    for timepoint in timepoint_batch:
        start = time()
        
        # Get contribution map for this timepoint/channel
        current_contribution = {}
        for layer in layer_names:
            current_contribution[layer] = resized_contribs[(channel, timepoint)][layer]
        
        # Spatial weighting
        current_activations = spatially_weight_activations(
            outputs, current_contribution, total_dim, n_images, 
            layer_names, layer_offsets, batch_size
        )
        
        print(f'Spatially weighted activations for ch {channel}, tp {timepoint} in {time() - start:.2f}s')

        if pca is None:
            # Fit PCA if not provided
            pca = PCA(n_components=n_components)
            pca.fit(current_activations['train'].astype(np.float32))

        # PCA transform with memory efficiency
        activations = pca.transform(current_activations['train'].astype(np.float32))
        comp_test_activations = pca.transform(current_activations['test'].astype(np.float32))
        
        # Prepare design matrices
        design_matrix = np.hstack((np.ones((activations.shape[0], 1)), activations))
        test_design_matrix = np.hstack((np.ones((comp_test_activations.shape[0], 1)), comp_test_activations))
        
        # Fit model
        y = zscore(train_data[:, channel, timepoint], axis=0)
        coef = fit_linear_regression_batch(X=design_matrix, Y=y, add_bias=False)
        
        # Predictions
        pred = predict_linear_regression_batch(X=design_matrix, coef=coef, add_bias=False).flatten()
        test_pred = predict_linear_regression_batch(X=test_design_matrix, coef=coef, add_bias=False).flatten()
        
        # Compute metrics
        c = np.corrcoef(pred, train_data[:, channel, timepoint])[0, 1]
        test_c = np.corrcoef(test_pred, test_data[:, channel, timepoint])[0, 1]
        
        results['corrs'].append(c)
        results['preds'].append(pred)
        results['test_corrs'].append(test_c)
        results['test_preds'].append(test_pred)
    
    return results

def iter_optimized(args):
    """CPU-optimized main iteration function"""
    layer_name, outputs, contribution, train_data, test_data, do_channels, do_timepoints, n_channels, n_timepoints, n_components, model_type, sub, crop_condition, crop_instance, fraction = args

    # Precompute shapes and resize contributions
    shape = {layer: outputs['train'][layer][0][0].shape[::-1] for layer in outputs['train'].keys()}
    # print("Precomputing resized contributions...")
    resized_contribs = precompute_resized_contributions(contribution, shape)
    
    # Compute metadata
    layer_names = list(outputs['train'].keys())
    split_names = list(outputs.keys())
    n_images = {split_name: len(outputs[split_name][layer_names[0]]) for split_name in split_names}
    layer_shapes = {layer: outputs['train'][layer][0].shape for layer in layer_names}
    # layer_sizes = {layer: outputs['train'][layer][0].numel() for layer in layer_names}
    layer_sizes = {layer: np.prod(layer_shapes[layer]) for layer in layer_names}
    
    # CPU-optimized batch size
    available_memory = psutil.virtual_memory().available
    estimated_batch_size = min(64, available_memory // (sum(layer_sizes.values()) * 4 * 8))  # Conservative
    batch_size = max(4, estimated_batch_size)
    
    # print(f"Using batch size: {batch_size}")
    
    # Compute layer offsets
    layer_offsets = {}
    offset = 0
    for layer in layer_names:
        size = layer_sizes[layer]
        layer_offsets[layer] = (offset, offset + size)
        offset += size
    total_dim = offset
    

    pca = None
    
    # Process channels with CPU optimization
    corr_channels = {}
    pred_channels = {}
    test_corr_channels = {}
    test_pred_channels = {}
    
    # Determine optimal number of processes
    # Balance between parallelism and memory usage
    max_processes = min(nproc // 2, 8)  # Don't over-subscribe
    n_timepoint_processes = max(1, max_processes)
    timepoint_batch_size = max(1, n_timepoints // n_timepoint_processes)
    
    # print(f"Using {n_timepoint_processes} processes for timepoint processing")
    
    # for channel in tqdm.tqdm(range(n_channels), desc="Processing channels"):
    for channel in tqdm.tqdm(do_channels, desc="Processing channels"):
        # Create batches of timepoints for parallel processing
        timepoint_batches = [
            [do_timepoints[i] for i in range(i, min(i + timepoint_batch_size, n_timepoints))]
            # list(range(i, min(i + timepoint_batch_size, n_timepoints)))
            for i in range(0, n_timepoints, timepoint_batch_size)
        ]
        
        # Prepare arguments for parallel processing
        args_list = [
            (batch, channel, resized_contribs, outputs, pca, n_components, train_data, test_data,
             total_dim, n_images, layer_names, layer_offsets, batch_size)
            for batch in timepoint_batches
        ]
        
        # Process timepoint batches in parallel
        if len(args_list) > 1 and n_timepoint_processes > 1:
            with ProcessPoolExecutor(max_workers=n_timepoint_processes) as executor:
                batch_results = list(executor.map(process_timepoint_batch, args_list))
        else:
            # Sequential processing for small datasets
            batch_results = [process_timepoint_batch(args) for args in args_list]
        
        # Aggregate results
        corrs, preds = [], []
        test_corrs, test_preds = [], []
        
        for batch_result in batch_results:
            corrs.extend(batch_result['corrs'])
            preds.extend(batch_result['preds'])
            test_corrs.extend(batch_result['test_corrs'])
            test_preds.extend(batch_result['test_preds'])
        
        corr_channels[channel] = corrs
        pred_channels[channel] = preds
        test_corr_channels[channel] = test_corrs
        test_pred_channels[channel] = test_preds

    result_dir = f'../results/sub-{sub}/alexnet_imagenet/across-layers/spatially-optimized'
    
    results = {
        'sub': sub,
        'model_type': model_type,
        'layer': layer_name,
        'crop_condition': crop_condition,
        'crop_instance': crop_instance,
        'fraction': fraction,
        'do_channels': do_channels,
        'do_timepoints': do_timepoints,
        'corr_channels': corr_channels,
        'test_corr_channels': test_corr_channels,
        'n_components': n_components,
        'pca': pca,
        'pred_channels': pred_channels,
        'test_pred_channels': test_pred_channels,
    }
    
    filename=f'encoding_results_sub_{sub}_{layer_name}_{model_type}_{crop_condition}-{crop_instance}-{fraction}.pkl'

    with open(os.path.join(result_dir, filename), 'wb') as f:
        pickle.dump(results, f)

    df = convert_to_df(sub=sub, results=results)
    df.to_parquet(os.path.join(result_dir, filename.replace('.pkl', '.parquet')))

    print(f"Completed processing for subject {sub}")
    
if __name__ == '__main__':
    # Initialize multiprocessing
    mp.set_start_method('spawn', force=True)

    oads_dir = '/home/nmuller/projects/data/oads'

    model_type = 'alexnet_imagenet'
    load_features_from_file = False

    num_workers = min(nproc, 32)  # Limit workers to prevent oversubscription

    # CPU-optimized batch size
    batch_size = 8
    
    oads = CustomOADS(basedir=oads_dir, n_processes=num_workers)
    
    # OADS Crops (400,400) mean, std
    mean = [0.3410, 0.3123, 0.2787]
    std = [0.2362, 0.2252, 0.2162]
    
    cleaning = '-AutoReject'



    ###########################################
    eeg_dir = f'../../eeg_data/main_experiment'
    channel_names, t = load_eeg_channel_and_timepoints()

    n_channels = len(channel_names)
    n_timepoints = len(t)
    do_timepoints = [i for i in range(150, n_timepoints-50, 20)]

    visual_channel_names = ['O1', 'O2', 'Oz', 'Iz', 'Pz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO7', 'POz', 'PO4', 'PO8'] # , 'I1', 'I2']
    do_channels = [channel_names.index(ch) for ch in visual_channel_names if ch in channel_names]
    ###########################################
    
    # Process subjects in parallel or sequentially based on memory constraints
    subjects = range(5, 36)
    
    for sub in subjects:
        print(f"Starting processing for subject {sub}")
        
        # Load contribution maps
        result_dir = '../../results/'
        contribution = np.load(os.path.join(result_dir, f'sub-{sub}', f'sub-{sub}_average_random_patch_contributions.npy'), allow_pickle=True).item()
        
        train_ids, test_ids, train_data, test_data = load_eeg_data(eeg_dir=eeg_dir, sub=sub)
        
        _, total_n_channels, total_n_timepoints = train_data.shape

        
        cont_shape = contribution[list(contribution.keys())[0]][list(contribution[list(contribution.keys())[0]].keys())[0]].shape
        print(f'Contribution shape: {cont_shape}')
        
        contribution_flat = np.ones((total_n_channels, total_n_timepoints, cont_shape[0], cont_shape[1]))
        for channel in contribution.keys():
            for timepoint in contribution[channel].keys():
                # if timepoint in do_timepoints and channel in do_channels:
                if contribution[channel][timepoint] is not None:
                    _cont = contribution[channel][timepoint]
                    _cont = (_cont - np.min(_cont)) / (np.max(_cont) - np.min(_cont))
                    contribution_flat[channel, timepoint] = _cont # contribution[channel][timepoint]
        
        n_timepoints = len(do_timepoints)
        n_channels = len(do_channels)


        feature_dir = '../../dnn_features'
        if load_features_from_file:
            # Load extracted features
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
            activations = extract_features(save_to_file=False, subjects=[sub], oads_dir='../../stimuli', model_type=model_type, save_dir=feature_dir, device='cuda:0')

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

        del activations
                
        # Prepare outputs structure
        base_layer_name = list(train_activations.keys())[0] if len(train_activations) > 0 else list(test_activations.keys())[0]
        base_image_index = list(train_activations[base_layer_name].keys())[0] if len(train_activations[base_layer_name]) > 0 else list(test_activations[base_layer_name].keys())[0]
        
        n_images = {
            'train': len(train_activations[base_layer_name]),
            'test': len(test_activations[base_layer_name])
        }
        
        # for n_components in [100]:
        n_components = 100
        outputs = {}
        
        for split_name, _activations in [('train', train_activations), ('test', test_activations)]:
            outputs[split_name] = {}
            for layer_name, layer in _activations.items():
                n_features = len(layer[base_image_index])
                shape = layer[base_image_index][0].shape
                
                print(f"Layer {layer_name}: {n_features} features, shape {shape}")
                
                # Pre-allocate with optimal memory layout
                outputs[split_name][layer_name] = np.zeros(
                    (n_images[split_name], n_features, shape[0], shape[1]),
                    dtype=np.float32,
                )
                
                for image_index, image in layer.items():
                    for feature_index, feature in enumerate(image):
                        outputs[split_name][layer_name][image_index, feature_index] = feature.cpu().numpy().astype(np.float32)
        
        crop_condition = 'spatially-optimized'
        crop_instance = 'full'
        fraction = 1.0

        del train_activations, test_activations
        
        print('Starting optimized processing...')
        
        # Call optimized function
        iter_optimized((
            'across-layers', outputs, contribution_flat, train_data, test_data,
            do_channels, do_timepoints,
            n_channels, n_timepoints, n_components,
            model_type, sub, crop_condition,
            crop_instance, fraction
        ))
        
        print(f"Completed subject {sub}")

    
    print("All processing completed!")