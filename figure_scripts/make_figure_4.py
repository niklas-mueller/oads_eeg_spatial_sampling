import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import matplotlib as mpl


from figure_utils import *

if __name__ == '__main__':

    home_path = os.path.expanduser('~')
    result_dir = '/home/nmuller/projects/fmg_storage/oads_experiment_analysis/'

    figure_dir = '/home/nmuller/projects/fmg_storage/tux20_oads_eeg_paper_figures'
    os.makedirs(figure_dir, exist_ok=True)

    tab10 = mpl.cm.get_cmap('tab10')
    tab20 = mpl.cm.get_cmap('tab20')
    tab20c = mpl.cm.get_cmap('tab20c')
    tab20b = mpl.cm.get_cmap('tab20b')

    gist_heat = mpl.cm.get_cmap('gist_heat')
    gist_heat = [gist_heat(i) for i in np.linspace(0,1,5,endpoint=False)]

    sns.set(font_scale=1.5)
    sns.set_style('ticks')

    plt.rcParams.update({
        'axes.titlesize': 'large',
        'axes.labelsize': 'large',
        'xtick.labelsize':'large',
        'ytick.labelsize':'large'
        })

    eeg_dir = '/home/nmuller/projects/fmg_projects/2024_Scholte_FMG-6506_oads_res/oads_resolution_experiment/data/fif_files/sub005_ses001size1-OC-epo.fif'
    epochs = read_epochs(fname=eeg_dir, preload=False)
    n_channels = len(epochs.ch_names)
    n_timepoints = len(epochs.times)
    sample_rate = 1024
    t = [i/sample_rate - 0.1 for i in range(n_timepoints)]
    channel_names = epochs.ch_names
    visual_channel_names = ['O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO7', 'POz', 'Pz', 'PO4', 'PO8'] # , 'I1', 'I2', 'Iz'
    visual_channel_indices = [channel_names.index(ch) for ch in visual_channel_names if ch in channel_names]

    # Load the data
    # df_sizes = pd.read_parquet(os.path.join(result_dir, 'eeg_resolution_encoding_results_all-subs_sizes.parquet'))
    df = pd.read_parquet(os.path.join(result_dir, 'eeg_resolution_encoding_results_all-subs_center-peri.parquet'))

    # df = pd.concat([df_sizes, df_center_peri])

    # Load noise ceiling
    noise_ceilings_subs = {}

    # size_conds = ['size1', 'size2', 'size3']
    # res_conds = ['quality1', 'quality2']
    peri_conds = ['peri1', 'peri2', 'peri3']
    center_conds = ['center1', 'center2', 'center3']

    eeg_dir = '/home/nmuller/projects/fmg_projects/2024_Scholte_FMG-6506_oads_res/oads_resolution_experiment/data_clean/base_dir_reordered/data_with_reject/'        

    for sub in range(1, 5):
        noise_ceilings_subs[sub] = {}
        for cond in peri_conds+center_conds:
            lower = np.load(os.path.join(eeg_dir, f'sub{str(sub).zfill(3)}', f'sub{str(sub).zfill(3)}_{cond}_lowerbound.npy'))
            upper = np.load(os.path.join(eeg_dir, f'sub{str(sub).zfill(3)}', f'sub{str(sub).zfill(3)}_{cond}_upperbound.npy'))

            noise_ceilings_subs[sub][cond] = (lower, upper)

    lower = {cond: 
        np.array([
            [
                np.mean([noise_ceilings_subs[sub][cond][0][channel][timepoint] for sub in noise_ceilings_subs])
                for timepoint in range(len(noise_ceilings_subs[sub][cond][0][channel]))
            ]
            for channel in range(64)
        ])
        for cond in size_conds+center_conds+peri_conds
    }

    upper = {cond:
        np.array([
            [
                np.mean([noise_ceilings_subs[sub][cond][1][channel][timepoint] for sub in noise_ceilings_subs])
                for timepoint in range(len(noise_ceilings_subs[sub][cond][1][channel]))
            ]
            for channel in range(64)
        ]) for cond in size_conds+center_conds+peri_conds
    }

    corrected_data = df[(df['channel'].isin(visual_channel_indices)) & (df['metric'] == 'test_corr_train') & (df['crop_instance'].str.contains('full'))].copy()
    corrected_data.crop_instance = corrected_data.crop_instance.str.replace('gcs-full', 'GCS').str.replace('feature-full', 'Full')
    corrected_data['corrected_value'] = corrected_data.apply(lambda x: x['value'] / upper[x['exp_condition']][x['channel'], x['timepoint_index']], axis=1)


    # ACTUAL FIGURE CODE
    fig = plt.figure(layout='constrained', figsize=(28, 24))

    _data = corrected_data[(corrected_data['channel'].isin(visual_channel_indices)) & (corrected_data['metric'] == 'test_corr_train')].copy()
    _data['size'] = _data['exp_condition'].apply(lambda x: x[-1].replace('1', 'Small').replace('2', 'Medium').replace('3', 'Large'))
    _data['exp_condition'] = _data.apply(lambda x: (x['exp_condition'][:-1].replace('center', 'Center').replace('peri', 'Periphery') + '-' + x['size'] if 'center' in x['exp_condition'] or 'peri' in x['exp_condition'] else x['exp_condition']), axis=1)
    _data.exp_condition = _data.exp_condition.astype('category')
    _data.exp_condition = _data.exp_condition.cat.remove_unused_categories()


    subfigs = fig.subfigures(2, 1, height_ratios=[1, 3])

    extents = ['Small', 'Medium', 'Large']
    images_subfigs = subfigs[0].subfigures(1, 3, wspace=0.1)
    for i in range(3):
        images_subfigs[i].suptitle(extents[i], fontsize=36, y=0.93, fontweight='bold')
    image_ax = np.array([images_subfigs[i].subplots(1,2) for i in range(3)]).flatten()

    ax = subfigs[1].subplots(3, 3, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [2, 2, 4], 'wspace': 0.1}, sharey='row')
    box_ax = ax[0, :]
    swarm_ax = ax[1, :]
    lineax = ax[2, :]


    make_center_peri_boxplot(box_ax)
    make_center_peri_swarmplot(swarm_ax, titles=False)
    plot_center_example_images(image_ax[::2].flatten())
    plot_periphery_example_images(image_ax[1::2].flatten())

    colormaps = {
        'Center-Small': tab20c.colors[4],
        'Center-Medium': tab20c.colors[5],
        'Center-Large': tab20c.colors[6],
        'Periphery-Small': tab20c.colors[0],
        'Periphery-Medium': tab20c.colors[1],
        'Periphery-Large': tab20c.colors[2],
    }

    for ax_index, size in enumerate(extents):
        
        g = sns.lineplot(data=_data[_data['size'] == size].groupby(['timepoint', 'exp_condition', 'crop_instance', 'subject']).corrected_value.mean().reset_index(), 
                        x='timepoint', y='corrected_value', hue='exp_condition', hue_order=[x for x in colormaps.keys() if size in x], 
                        style='crop_instance', ax=lineax[ax_index], linewidth=3, palette=colormaps)

        lineax[ax_index].set_xlabel('Time (s)')
        handles, labels = lineax[ax_index].get_legend_handles_labels()
        handles = [handles[i] for i in [1,2,4,5]]
        labels = [labels[i].replace('-Small', '').replace('-Medium', '').replace('-Large', '') for i in [1,2,4,5]]
        lineax[ax_index].legend(handles=handles, labels=labels, title='', ncols=2)
        if ax_index == 0:
            lineax[ax_index].set_ylabel('Prediction performance (r)')
        else:
            lineax[ax_index].set_ylabel('')
        
        lineax[ax_index].set_title(size)

    for ax_index, _ax in enumerate(lineax):
        _ax.axvline(0, color='gray', linestyle='--', zorder=-10)
        _ax.axhline(0, color='gray', linestyle='--', zorder=-10)

    sns.despine()

    ax[0, 0].text(0.0, 0.95, 'a)', color='black', size=40, weight='bold', clip_on=False, transform=fig.transFigure, in_layout=False)
    ax[1, 0].text(0.0, 0.75, 'b)', color='black', size=40, weight='bold', clip_on=False, transform=fig.transFigure, in_layout=False)
    ax[-1, 0].text(0.0, 0.35, 'c)', color='black', size=40, weight='bold', clip_on=False, transform=fig.transFigure, in_layout=False)

    # fig.savefig(os.path.join(figure_dir, 'oads_center-periphery_full-vs-gcs_normalized.png'), dpi=300)
    plt.show()