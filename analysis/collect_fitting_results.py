import os
import numpy as np
from PIL import Image
import torchvision
# from torchvision.models import alexnet
# import numpy as np
import matplotlib.pyplot as plt
# from pytorch_utils.resnet10 import ResNet10
import torch
from torch.utils.data import DataLoader
from result_manager.result_manager import ResultManager
# from oads_access.oads_access import OADS_Access, OADSImageDataset
from torchvision.models import resnet18, resnet50, alexnet
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from pytorch_utils.pytorch_utils import ToJpeg, ToOpponentChannel, collate_fn, record_activations, EdgeResize, ToRetinalGanglionCellSampling, EdgeCrop
from sklearn.decomposition import PCA
from lgnpy.CEandSC.lgn_statistics import regress, get_field_of_view
import tqdm
from scipy.stats import zscore
from mne import read_epochs
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import rawpy
import seaborn as sns
import multiprocessing

import matplotlib as mpl
import matplotlib.patches as patches

from scipy.stats import zscore, ttest_ind, ttest_1samp
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import patches
from mne import create_info
from mne.viz import plot_topomap
from mne.channels import get_builtin_montages, make_dig_montage, make_standard_montage

from oads_access.oads_access import OADS_Access
import pickle
import cProfile
from memory_profiler import profile

def iterate_load_subject_data(args):
    sub, result_manager = args

    rows = []
    # rows_pca = []
    for image_representation in ['rgb']:
        for image_quality in ['raw']: 
            for n_pca_components in [100]: 
                for model_type in ['alexnet']:
                    for layer in [f'{model_type}_layer1',f'{model_type}_layer2',f'{model_type}_layer3', 'across-layers']:
                        for image_resolution in ['400']:
                            for crop_condition in ['feature', 'gcs', 'fraction']: # 'gcs_inverse'

                                crop_instances = ['gcs-full'] if 'gcs' in crop_condition else (['center', 'periphery', 'center_circ', 'periphery_circ'] if 'fraction' in crop_condition else ['feature-full'])
                                fractions = [1.0] if 'gcs' in crop_condition or 'feature' in crop_condition else [0.005, 0.01, 0.05, 0.1, 0.2] # 

                                for fraction in fractions:
                                    for crop_instance in crop_instances:

                                        results = result_manager.load_result(filename=f'encoding_results_pca_{n_pca_components}_sub_{sub}_{model_type}_feature-cropping_{layer}_{image_representation}_{image_quality}_{image_resolution}_{crop_condition}_{crop_instance}_{fraction}.pkl')
                                        folder = result_manager.root
                                        
                                        if results is not None:
                                            
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
                                                    
                                                    test_corr_train = results['test_corr_channels'][channel][timepoint]
                                                    rows.append([folder, sub, model_type, layer, image_representation, image_quality, n_pca_components, image_resolution, crop_condition, crop_instance, fraction, channel_names[channel], channel, t[timepoint], timepoint, -1, 'test_corr_train', test_corr_train])

    return sub, result_manager, rows

if __name__ == '__main__':

    home_path = os.path.expanduser('~')
    # result_dir = '/home/nmuller/projects/fmg_storage/oads_experiment_analysis/'
    result_dir = '/home/nmuller/projects/oads_eeg_spatial_sampling/results/encoding_alexnet_feature-croppingAutoReject'

    eeg_dir = f'{home_path}/projects/data/oads_eeg/sub_13/sub_13-OC&CSD-AutoReject-epo.fif'

    epochs = read_epochs(fname=eeg_dir, preload=False)

    n_channels = len(epochs.ch_names)
    n_timepoints = len(epochs.times)
    sample_rate = 1024
    t = [i/sample_rate - 0.1 for i in range(n_timepoints)]

    channel_names = epochs.ch_names
    visual_channel_names = ['O1', 'O2', 'Oz', 'Iz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO7', 'POz', 'PO4', 'PO8', 'I1', 'I2']

    if os.path.exists(os.path.join(result_dir, f'all_encoding_model_data_feature_cropping-AutoReject.parquet')):
        df = pd.read_parquet(os.path.join(result_dir, f'all_encoding_model_data_feature_cropping-AutoReject.parquet'))
    else:
        df = pd.DataFrame()
    
    cols = ['folder', 'subject', 'model_type', 'layer', 'image_representation', 'image_quality', 'n_pca_components', 'resolution', 'crop_condition', 'crop_instance', 'fraction', 'channel', 'channel_index', 'timepoint', 'timepoint_index', 'pred_index', 'metric', 'value']
    dtypes = {col: 'category' for col in cols if col != 'value'}
    # rows = []

    n_components = 100
    model_type = 'alexnet'
    layer = 'across-layers'
    image_representation = 'rgb'
    image_quality = 'raw'
    crop_condition = 'feature'
    crop_instance = 'feature-full'
    fraction = 1.0
    for sub in range(5, 6):
        _df = pd.read_parquet(os.path.join(result_dir, f'encoding_results_pca_{n_components}_sub_{sub}_{model_type}_feature-cropping_{layer}_{image_representation}_{image_quality}_{crop_condition}_{crop_instance}_{fraction}.parquet'))
        df = pd.concat([_df, df], axis=0)

    # result_manager = ResultManager(root=os.path.join(result_dir, 'correct_size_new_fit', folder))

    # results = [iterate_load_subject_data((sub, result_manager)) for sub in range(5, 8)]
    # with multiprocessing.Pool(4) as pool:
    #     results = list(tqdm.tqdm(pool.map(iterate_load_subject_data, [(sub, result_manager) for sub in range(5, 36)]), total=31))

    # for sub, result_manager, _rows in results:
    #     rows.extend(_rows)

    # df = pd.DataFrame(data=rows, columns=cols)
    # df = pd.concat([_df, df], axis=0)

    dtypes = {col: 'category' for col in cols if col != 'value'}
    dtypes['value'] = 'float32'
    for col in cols:
        df[col] = df[col].astype(dtypes[col])


    df.to_parquet(os.path.join(result_dir, f'all_encoding_model_data_feature_cropping-AutoReject.parquet'))
    