import os
nproc = 16

os.environ["OMP_NUM_THREADS"] = str(nproc)
os.environ["OPENBLAS_NUM_THREADS"] = str(nproc)
os.environ["MKL_NUM_THREADS"] = str(nproc)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(nproc)
os.environ["NUMEXPR_NUM_THREADS"] = str(nproc)

import pingouin as pg
import os
import numpy as np
from result_manager.result_manager import ResultManager
import tqdm
from scipy.stats import zscore
from mne import read_epochs
import pandas as pd
import multiprocessing

import pickle

def load_sub_eeg(sub, cleaning = '-AutoReject'):
    target_filenames = ['0434262454981c92.tiff', '0c6c2c66e61e3133.tiff', '0e2e2f2931313d37.tiff', '1332a2d4c4c14322.tiff', '202024d9b333968c.tiff', '24247432670b3131.tiff', '2c2064d163c52c78.tiff', '42431879e191d3c3.tiff', '61662ece1f6e7870.tiff', '7890918716766325.tiff', '85cc4e533959b1e1.tiff', '8ece4062eee68292.tiff', '93939b9a101e87e0.tiff', '948cacbc94b42474.tiff', 'af8f3a3939313171.tiff', 'b23232b232332361.tiff', 'b371f8ecf4e0f0f0.tiff', 'bce2d2d2c393e3b2.tiff', 'c77131717179f173.tiff', 'c93b3b2b29b9b8f1.tiff', 'cc8c9290bcbcf4fc.tiff', 'dad8acb8b82e36b1.tiff', 'e2f8f8fcf8f8f8f8.tiff', 'e6d8d40438c0c8e2.tiff', 'e6e648c62ebacc38.tiff', 'ef3632c2c1476e78.tiff', 'f4ca83dc6731310d.tiff']

    eeg_dir = '/home/nmuller/projects/fmg_storage/osf_eeg_data'
    with open(os.path.join(eeg_dir, f'filenames_oads_eeg_rsvp_sub-{str(sub).zfill(2)}.pkl'), 'rb') as f:
        filenames = pickle.load(f)

    with open(os.path.join(eeg_dir, f'is_test_oads_eeg_rsvp_sub-{str(sub).zfill(2)}.pkl'), 'rb') as f:
        is_test = pickle.load(f)

    data = np.load(os.path.join(eeg_dir, f'oads_eeg_rsvp_sub-{str(sub).zfill(2)}.npy'))
    test_filenames = [filenames[i] for i in range(len(filenames)) if is_test[i]]
    test_ids = [x.split('.')[0] for x in test_filenames if f"{x.split('.')[0]}.tiff" not in target_filenames]
    test_data = np.array([data[i] for i in range(len(data)) if is_test[i]])

    return test_data

def iter(args):
    (crop_condition_pair, result_manager, folder, test_data, n_pca_components, sub, model_type, layer, image_representation, image_quality, image_resolution) = args

    rows = []

    crop_pairs_results = {}
    for crop_condition, crop_instance, fraction in crop_condition_pair:
        results = result_manager.load_result(filename=f'encoding_results_pca_{n_pca_components}_sub_{sub}_{model_type}_feature-cropping_{layer}_{image_representation}_{image_quality}_{image_resolution}_{crop_condition}_{crop_instance}_{fraction}.pkl')
        crop_pairs_results[(crop_condition, crop_instance, fraction)] = results

    for channel in crop_pairs_results[list(crop_pairs_results.keys())[0]]['test_corr_channels'].keys():
        for timepoint in range(513):

            y = zscore(test_data[:, channel, timepoint])

            _data = [crop_pairs_results[(crop_condition, crop_instance, fraction)]['test_pred_channels'][channel][timepoint] for crop_condition, crop_instance, fraction in crop_condition_pair]
            _data.append(y)

            cols = [crop_condition if 'gcs' in crop_condition else f'{crop_condition}_{crop_instance}_{fraction}' for crop_condition, crop_instance, fraction in crop_condition_pair]
            cols.append('y')

            df_corr = pd.DataFrame(columns=cols, data=np.array(_data).T)

            for f in cols:
                if f == 'y':
                    continue
                else:
                    other_f = [other_f for other_f in cols if other_f != f and other_f != 'y'][0]
                    par_corr = pg.partial_corr(data=df_corr, x=f, y='y', covar=other_f).r

                rows.append([
                    folder, sub, model_type, layer, image_representation, image_quality, n_pca_components, image_resolution, f, other_f, channel_names[channel], channel, t[timepoint], timepoint, 'partial_corr', par_corr
                ])

    return rows

def iterate_load_subject_data(args):
    sub, result_manager = args

    test_data = load_sub_eeg(sub)

    t = [i/1024 - 0.1 for i in range(513)]

    rows = []
    # rows_pca = []
    for image_representation in ['rgb']:
        for image_quality in ['raw']: 
            for n_pca_components in [100]: 
                for model_type in [f'alexnet']:
                    for layer in ['across-layers']: #  f'{model_type}_layer1',f'{model_type}_layer2',f'{model_type}_layer3', 
                        for image_resolution in ['400']:
                            
                            crop_condition_pairs = [
                                # [('gcs', 'gcs-full', 1.0), ('gcs_inverse', 'gcs-full', 1.0)],
                            ]

                            crop_condition_pairs.append([('feature', 'feature-full', 1.0), ('gcs', 'gcs-full', 1.0)])

                            for _fraction in [0.005, 0.01, 0.05, 0.1, 0.2]:
                                crop_condition_pairs.append([('feature', 'feature-full', 1.0), ('fraction', 'center_circ', _fraction)])
                                crop_condition_pairs.append([('feature', 'feature-full', 1.0), ('fraction', 'periphery_circ', _fraction)])
                                crop_condition_pairs.append([('feature', 'feature-full', 1.0), ('fraction', 'intersect_circ', _fraction)])
                                crop_condition_pairs.append([('feature', 'feature-full', 1.0), ('fraction', 'center', _fraction)])
                                crop_condition_pairs.append([('feature', 'feature-full', 1.0), ('fraction', 'periphery', _fraction)])
                                crop_condition_pairs.append([('feature', 'feature-full', 1.0), ('fraction', 'intersect_flat', _fraction)])

                                crop_condition_pairs.append([('fraction', 'center_circ', _fraction), ('gcs', 'gcs-full', 1.0)])
                                crop_condition_pairs.append([('fraction', 'periphery_circ', _fraction), ('gcs', 'gcs-full', 1.0)])
                                crop_condition_pairs.append([('fraction', 'intersect_circ', _fraction), ('gcs', 'gcs-full', 1.0)])

                                crop_condition_pairs.append([('fraction', 'center', _fraction), ('fraction', 'periphery', _fraction)])
                                crop_condition_pairs.append([('fraction', 'center', _fraction), ('fraction', 'intersect_flat', _fraction)])
                                crop_condition_pairs.append([('fraction', 'periphery', _fraction), ('fraction', 'intersect_flat', _fraction)])

                                crop_condition_pairs.append([('fraction', 'center_circ', _fraction), ('fraction', 'periphery_circ', _fraction)])
                                crop_condition_pairs.append([('fraction', 'center_circ', _fraction), ('fraction', 'intersect_circ', _fraction)])
                                crop_condition_pairs.append([('fraction', 'periphery_circ', _fraction), ('fraction', 'intersect_circ', _fraction)])

                                # add also a comparison to the normal feature model

                                for other_fraction in [0.005, 0.01, 0.05, 0.1, 0.2]:
                                    # pass
                                    if _fraction != other_fraction:
                                        crop_condition_pairs.append([('fraction', 'center', _fraction), ('fraction', 'center', other_fraction)])
                                        crop_condition_pairs.append([('fraction', 'periphery', _fraction), ('fraction', 'periphery', other_fraction)])
                                        crop_condition_pairs.append([('fraction', 'intersect_flat', _fraction), ('fraction', 'intersect_flat', other_fraction)])
                                        crop_condition_pairs.append([('fraction', 'center_circ', _fraction), ('fraction', 'center_circ', other_fraction)])
                                        crop_condition_pairs.append([('fraction', 'periphery_circ', _fraction), ('fraction', 'periphery_circ', other_fraction)])
                                        crop_condition_pairs.append([('fraction', 'intersect_circ', _fraction), ('fraction', 'intersect_circ', other_fraction)])


                            with multiprocessing.Pool(nproc) as pool:
                                results = list(tqdm.tqdm(pool.imap(iter, [(crop_condition_pair, result_manager, folder, test_data, n_pca_components, sub, model_type, layer, image_representation, image_quality, image_resolution) for crop_condition_pair in crop_condition_pairs]), total=len(crop_condition_pairs)))

                            for _rows in results:
                                rows.extend(_rows)

    return sub, result_manager, rows #, rows_pca

###############
if __name__ == '__main__':
    home_path = os.path.expanduser('~')
    eeg_dir = f'{home_path}/projects/data/oads_eeg/sub_13/sub_13-OC&CSD-AutoReject-epo.fif'

    epochs = read_epochs(fname=eeg_dir, preload=False, verbose=False)

    n_channels = len(epochs.ch_names)
    n_timepoints = len(epochs.times)
    sample_rate = 1024
    t = [i/sample_rate - 0.1 for i in range(n_timepoints)]

    channel_names = epochs.ch_names
    visual_channel_names = ['O1', 'O2', 'Oz', 'Iz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO7', 'POz', 'PO4', 'PO8', 'I1', 'I2']

    result_dir = '/home/nmuller/projects/fmg_storage/oads_experiment_analysis/'
    cleaning = '-AutoReject'

    folder = f'correct_size_new_fit/encoding_alexnet_imagenet_share-pca_partial-corr_feature-cropping{cleaning}'


    cols = ['folder', 'subject', 'model_type', 'layer', 'image_representation', 'image_quality', 'n_pca_components', 'resolution', 'condition', 'given', 'channel', 'channel_index', 'timepoint', 'timepoint_index', 'metric', 'value']
    dtypes = {col: 'category' for col in cols if col != 'value'}
    dtypes['value'] = 'float32'

    df = pd.read_parquet(os.path.join(result_dir, folder, f'all_encoding_model_data_alexnet_imagenet_partial-corr_feature_cropping{cleaning}.parquet'))

    result_manager = ResultManager(root=os.path.join(result_dir, folder))

    for sub in tqdm.tqdm(range(5, 36), total=31):
        _, _, _rows = iterate_load_subject_data((sub, result_manager))

        if df is None:
            df = pd.DataFrame(_rows, columns=cols)
            for col in cols:
                df[col] = df[col].astype(dtypes[col])
        else:
            df = pd.concat((df, pd.DataFrame(_rows, columns=cols)))
            for col in cols:
                df[col] = df[col].astype(dtypes[col])

        os.makedirs(os.path.join(result_dir, folder), exist_ok=True)
        df.to_parquet(os.path.join(result_dir, folder, f'all_encoding_model_data_alexnet_imagenet_partial-corr_feature_cropping{cleaning}.parquet'))