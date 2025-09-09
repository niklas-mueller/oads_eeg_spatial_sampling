import os
import pingouin as pg
import os
import numpy as np
import tqdm
from scipy.stats import zscore
from mne import read_epochs
import pandas as pd
from time import time
import pickle

from eeg_data import load_eeg_data, load_eeg_channel_and_timepoints

def load_sub_eeg(sub, cleaning = '-AutoReject'):
    target_filenames = ['0434262454981c92.tiff', '0c6c2c66e61e3133.tiff', '0e2e2f2931313d37.tiff', '1332a2d4c4c14322.tiff', '202024d9b333968c.tiff', '24247432670b3131.tiff', '2c2064d163c52c78.tiff', '42431879e191d3c3.tiff', '61662ece1f6e7870.tiff', '7890918716766325.tiff', '85cc4e533959b1e1.tiff', '8ece4062eee68292.tiff', '93939b9a101e87e0.tiff', '948cacbc94b42474.tiff', 'af8f3a3939313171.tiff', 'b23232b232332361.tiff', 'b371f8ecf4e0f0f0.tiff', 'bce2d2d2c393e3b2.tiff', 'c77131717179f173.tiff', 'c93b3b2b29b9b8f1.tiff', 'cc8c9290bcbcf4fc.tiff', 'dad8acb8b82e36b1.tiff', 'e2f8f8fcf8f8f8f8.tiff', 'e6d8d40438c0c8e2.tiff', 'e6e648c62ebacc38.tiff', 'ef3632c2c1476e78.tiff', 'f4ca83dc6731310d.tiff']

    eeg_dir = '/home/nmuller/projects/fmg_storage/osf_eeg_data/AutoReject'
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
    (crop_condition_pair, result_dir, test_data, n_pca_components, sub, model_type, layer) = args

    rows = []
    channels = []
    timepoints = []

    crop_pairs_results = {}
    for crop_condition, crop_instance, fraction in crop_condition_pair:
        
        results = pd.read_parquet(os.path.join(result_dir.format(sub=sub, model_type=model_type, layer=layer, encoding_model=f'{crop_condition}-{crop_instance}-{fraction}'), f'encoding_results_sub_{sub}_{layer}_{model_type}-feature-cropping-{crop_condition}-{crop_instance}-{fraction}.parquet'))

        if len(channels) == 0:
            channels = results.channel_index.unique()
        if len(timepoints) == 0:
            timepoints = results.timepoint_index.unique()

        crop_pairs_results[(crop_condition, crop_instance, fraction)] = results[results['metric'] == 'test_pred_channels']

    # for channel in crop_pairs_results[list(crop_pairs_results.keys())[0]]['test_corr_channels'].keys():
    #     for timepoint in range(513):
    for channel in tqdm.tqdm(channels, total=len(channels), desc=f'Channels'):
        # start_time = time()
        for timepoint in timepoints:
            y = zscore(test_data[:, channel, timepoint])

            _data = []
            for crop_condition, crop_instance, fraction in crop_condition_pair:
                __data = crop_pairs_results[(crop_condition, crop_instance, fraction)]
                __data = __data[(__data['channel_index'] == channel) & (__data['timepoint_index'] == timepoint)].value.values
                _data.append(__data)
            # _data = 
            # _data = [crop_pairs_results[(crop_condition, crop_instance, fraction)]['test_pred_channels'][channel][timepoint] for crop_condition, crop_instance, fraction in crop_condition_pair]
            # _data = [crop_pairs_results[(crop_condition, crop_instance, fraction)]['test_pred_channels'][channel][timepoint] for crop_condition, crop_instance, fraction in crop_condition_pair]
            _data.append(y)

            cols = [crop_condition if 'gcs' in crop_condition else f'{crop_condition}_{crop_instance}_{fraction}' for crop_condition, crop_instance, fraction in crop_condition_pair]
            cols.append('y')

            df_corr = pd.DataFrame(columns=cols, data=np.array(_data).T)

            for f in cols:
                if f == 'y':
                    continue
                else:
                    other_f = [other_f for other_f in cols if other_f != f and other_f != 'y'][0]
                    par_corr = pg.partial_corr(data=df_corr, x=f, y='y', covar=other_f).r.pearson

                rows.append([
                    sub, model_type, layer, n_pca_components, f, other_f, channel_names[channel], channel, t[timepoint], timepoint, 'partial_corr', par_corr
                ])

        # end_time = time()
        # print(f'{end_time - start_time} seconds')

    return rows

def iterate_load_subject_data(args):
    sub, result_dir = args

    test_data = load_sub_eeg(sub)

    rows = []
    
    for n_pca_components in [100]: 
        for model_type in [f'alexnet_imagenet']:
            for layer in ['across-layers']: #  f'{model_type}_layer1',f'{model_type}_layer2',f'{model_type}_layer3', 
                    
                crop_condition_pairs = [
                    # [('gcs', 'gcs-full', 1.0), ('gcs_inverse', 'gcs-full', 1.0)],
                ]

                crop_condition_pairs.append([('feature', 'feature-full', 1.0), ('gcs', 'gcs-full', 1.0)])

                # for _fraction in [0.005, 0.01, 0.05, 0.1, 0.2]:
                for _fraction in [0.005]:
                    # crop_condition_pairs.append([('feature', 'feature-full', 1.0), ('fraction', 'center_circ', _fraction)])
                    # crop_condition_pairs.append([('feature', 'feature-full', 1.0), ('fraction', 'periphery_circ', _fraction)])
                    # crop_condition_pairs.append([('feature', 'feature-full', 1.0), ('fraction', 'intersect_circ', _fraction)])
                    crop_condition_pairs.append([('feature', 'feature-full', 1.0), ('fraction', 'center', _fraction)])
                    crop_condition_pairs.append([('feature', 'feature-full', 1.0), ('fraction', 'periphery', _fraction)])
                    # crop_condition_pairs.append([('feature', 'feature-full', 1.0), ('fraction', 'intersect_flat', _fraction)])

                    # crop_condition_pairs.append([('fraction', 'center_circ', _fraction), ('gcs', 'gcs-full', 1.0)])
                    # crop_condition_pairs.append([('fraction', 'periphery_circ', _fraction), ('gcs', 'gcs-full', 1.0)])
                    # crop_condition_pairs.append([('fraction', 'intersect_circ', _fraction), ('gcs', 'gcs-full', 1.0)])

                    crop_condition_pairs.append([('fraction', 'center', _fraction), ('fraction', 'periphery', _fraction)])
                    # crop_condition_pairs.append([('fraction', 'center', _fraction), ('fraction', 'intersect_flat', _fraction)])
                    # crop_condition_pairs.append([('fraction', 'periphery', _fraction), ('fraction', 'intersect_flat', _fraction)])

                    # crop_condition_pairs.append([('fraction', 'center_circ', _fraction), ('fraction', 'periphery_circ', _fraction)])
                    # crop_condition_pairs.append([('fraction', 'center_circ', _fraction), ('fraction', 'intersect_circ', _fraction)])
                    # crop_condition_pairs.append([('fraction', 'periphery_circ', _fraction), ('fraction', 'intersect_circ', _fraction)])

                    # for other_fraction in [0.005, 0.01, 0.05, 0.1, 0.2]:
                    #     # pass
                    #     if _fraction != other_fraction:
                    #         crop_condition_pairs.append([('fraction', 'center', _fraction), ('fraction', 'center', other_fraction)])
                    #         crop_condition_pairs.append([('fraction', 'periphery', _fraction), ('fraction', 'periphery', other_fraction)])
                    #         crop_condition_pairs.append([('fraction', 'intersect_flat', _fraction), ('fraction', 'intersect_flat', other_fraction)])
                    #         crop_condition_pairs.append([('fraction', 'center_circ', _fraction), ('fraction', 'center_circ', other_fraction)])
                    #         crop_condition_pairs.append([('fraction', 'periphery_circ', _fraction), ('fraction', 'periphery_circ', other_fraction)])
                    #         crop_condition_pairs.append([('fraction', 'intersect_circ', _fraction), ('fraction', 'intersect_circ', other_fraction)])

                for crop_condition_pair in crop_condition_pairs:
                    results = iter((crop_condition_pair, result_dir, test_data, n_pca_components, sub, model_type, layer))
                    rows.extend(results)
                    
                # with multiprocessing.Pool(nproc) as pool:
                #     results = list(tqdm.tqdm(pool.imap(iter, [(crop_condition_pair, result_manager, folder, test_data, n_pca_components, sub, model_type, layer) for crop_condition_pair in crop_condition_pairs]), total=len(crop_condition_pairs)))
                # for _rows in results:
                #     rows.extend(_rows)


    return sub, rows #, rows_pca

###############
if __name__ == '__main__':

    eeg_dir = f'../../eeg_data/main_experiment'
    channel_names, t = load_eeg_channel_and_timepoints()

    encoding_model_dir = '../../results/sub-{sub}/{model_type}/{layer}/{encoding_model}'
    result_dir = '../../results/sub-{sub}/{model_type}/{layer}'

    cols = ['subject', 'model_type', 'layer', 'n_pca_components', 'condition', 'given', 'channel', 'channel_index', 'timepoint', 'timepoint_index', 'metric', 'value']
    dtypes = {col: 'category' for col in cols if col != 'value'}

    model_type = 'alexnet_imagenet'

    for sub in tqdm.tqdm(range(5, 6), total=1):
        _, _rows = iterate_load_subject_data((sub, encoding_model_dir))

        # if df is None:
        df = pd.DataFrame(_rows, columns=cols)
        for col in cols:
            if col in dtypes:
                df[col] = df[col].astype(dtypes[col]) # type: ignore


        os.makedirs(result_dir.format(sub=sub, model_type=model_type, layer='across-layers'), exist_ok=True)
        df.to_parquet(os.path.join(result_dir.format(sub=sub, model_type=model_type, layer='across-layers'), f'partial_correlation_feature_cropping.parquet'))