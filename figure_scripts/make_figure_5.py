import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import matplotlib as mpl

from mne.viz import plot_topomap

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
    df = pd.read_parquet(os.path.join(result_dir, 'eeg_resolution_encoding_results_all-subs_sizes.parquet'))
    # df_center_peri = pd.read_parquet(os.path.join(result_dir, 'eeg_resolution_encoding_results_all-subs_center-peri.parquet'))

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
        for cond in size_conds:
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
    fig, ax = plt.subplots(4,6, figsize=(30,12), sharey=False, sharex=False, gridspec_kw={'width_ratios': [2, 2, 2, 0.5, 2, 2], 'height_ratios': [1, 0.2, 0.5, 0.5]})

    for _ax in ax[1, :].flatten():
        _ax.axis('off')

    corrected_data['exp_condition'] = corrected_data['exp_condition'].str.replace('size1', 'Small').str.replace('size2', 'Medium').str.replace('size3', 'Large')
    sizes = ['size1', 'size2', 'size3']

    colormaps = {
        'Full': tab10.colors[5],
        'GCS': 'black',
    }

    lineplot_channel = 'Iz'

    for ax_index, condition in enumerate(['Small', 'Medium', 'Large']):
        _data = corrected_data[((corrected_data['exp_condition'] == condition) & (corrected_data['channel'] == channel_names.index(lineplot_channel)))]
        
        f = sns.lineplot(data=_data, x='timepoint', y='corrected_value', hue='crop_instance', ax=ax[2, ax_index], linewidth=3, palette=colormaps, legend='full' if ax_index == 1 else False)
        ax[2, ax_index].set_title(condition)
        # ax[2, ax_index].set_xlabel('Time (s)')

        ax[2, ax_index].set_xlabel('')
        ax[2, ax_index].set_ylabel('')
        ax[2, ax_index].set_xticks([])

        if ax_index == 1:
            f.legend(title='', loc='upper left')
        else:
            f.legend().set_visible(False)
        
        if ax_index > 0:
            ax[2, ax_index].set_yticks([])

        ax[2, ax_index].axvline(0, color='gray', linestyle='--')
        ax[2, ax_index].axhline(0, color='gray', linestyle='--')

        # ax[2, ax_index].axis('off')

        corrected_lower = lower[sizes[ax_index]][[channel_names.index(lineplot_channel)], :] / upper[sizes[ax_index]][[channel_names.index(lineplot_channel)], :]
        ax[2, ax_index].fill_between(corrected_data.timepoint.unique(), 1, corrected_lower.mean(axis=0), color='gray', alpha=0.5)
        ax[2, ax_index].plot(corrected_data.timepoint.unique(), corrected_lower.mean(axis=0), color=tab10.colors[ax_index], linestyle='--', zorder=-1, linewidth=2)
        
        # ===========================================================================
        average_data = corrected_data[((corrected_data['exp_condition'] == condition) & (corrected_data['channel'].isin(visual_channel_indices)))].groupby(['subject', 'crop_instance', 'timepoint']).corrected_value.mean().reset_index()
        f = sns.lineplot(data=average_data, x='timepoint', y='corrected_value', hue='crop_instance', ax=ax[3, ax_index], linewidth=3, palette=colormaps, legend=False)

        ax[3, ax_index].set_xlabel('')
        ax[3, ax_index].set_ylabel('')
        if ax_index == 1:
            ax[3, ax_index].set_xlabel('Time (s)')

        # if ax_index == 0:
        #     ax[3, ax_index].set_ylabel('Normalized\nPrediction performance (r)')
            # f.legend().set_title('')
        # else:
        #     ax[3, ax_index].set_ylabel('')
        if ax_index > 0:
            ax[3, ax_index].set_yticks([])
        # f.legend().set_visible(False)
        
        ax[3, ax_index].axvline(0, color='gray', linestyle='--')
        ax[3, ax_index].axhline(0, color='gray', linestyle='--')
        
        corrected_lower = lower[sizes[ax_index]][visual_channel_indices, :] / upper[sizes[ax_index]][visual_channel_indices, :]
        ax[3, ax_index].fill_between(corrected_data.timepoint.unique(), 1, corrected_lower.mean(axis=0), color='gray', alpha=0.5)
        ax[3, ax_index].plot(corrected_data.timepoint.unique(), corrected_lower.mean(axis=0), color=tab10.colors[ax_index], linestyle='--', zorder=-1, linewidth=2)
        # ===========================================================================
        


    # ============================================================================================== single channel
    left, bottom, width, height = [0.11, 0.4, 0.075, 0.075]
    inset_ax0 = fig.add_axes([left, bottom, width, height])

    info, tvalue_channel_names = make_custom_info(visual_channel_names)
    mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k',
            linewidth=0, markersize=6)
    channel_mask = np.array([x in [lineplot_channel] for x in visual_channel_names])

    _ax = plot_topomap(data=np.zeros((len(visual_channel_names), )), pos=info, axes=inset_ax0, mask_params=mask_params, mask=channel_mask, sensors=False, show=False)
    inset_ax0.set_title(lineplot_channel, fontweight='bold', y=0.2)
    # ==============================================================================================

    # ============================================================================================== all channels
    left, bottom, width, height = [0.11, 0.2, 0.075, 0.075]
    inset_ax1 = fig.add_axes([left, bottom, width, height])

    info, tvalue_channel_names = make_custom_info(visual_channel_names)
    mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k',
            linewidth=0, markersize=6)
    channel_mask = np.array([True for x in visual_channel_names])

    _ax = plot_topomap(data=np.zeros((len(visual_channel_names), )), pos=info, axes=inset_ax1, mask_params=mask_params, mask=channel_mask, sensors=False, show=False)
    # inset_ax1.set_title(lineplot_channel, fontweight='bold')
    # ==============================================================================================


    # ==============================================================================================
    plot_size_example_images(ax[0, :-3], 3)

    gs = ax[0, -2].get_gridspec()
    for _ax in ax[0, -2:]:
        _ax.remove()
    _ax = fig.add_subplot(gs[0, -2:])
    # plot_normalized_size_noise_ceilings(ax[0, -1])
    plot_normalized_size_noise_ceilings(_ax, False)
    _ax.set_title('Normalized Lower Noise Ceiling')
    _ax.set_ylabel('Correlation (r)')


    # ================================================
    pairs = [
        [('Small', 'Full'), ('Small', 'GCS')],
        [('Medium', 'Full'), ('Medium', 'GCS')],
        [('Large', 'Full'), ('Large', 'GCS')],
    ]

    plotting_parameters = {
        'data':    corrected_data.groupby(['subject', 'crop_instance', 'exp_condition']).mean(numeric_only=True).reset_index(),
        # 'data':    corrected_data,
        'x':       'exp_condition',
        'y':       'corrected_value',
        'hue':     'crop_instance',
        'palette':  colormaps,
        'order':    ['Small', 'Medium', 'Large'],
        'hue_order': ['Full', 'GCS'],
        # 'ax': ax,
    }

    gs = ax[2, -2].get_gridspec()
    for _ax in ax[2:, -2:].flatten():
        _ax.remove()
    _axbox = fig.add_subplot(gs[2:, -2])
    _axswarm = fig.add_subplot(gs[2:, -1])

    f = sns.boxplot(ax=_axbox, **plotting_parameters)
    g = sns.swarmplot(ax=_axswarm, **plotting_parameters, dodge=True, linewidth=3, edgecolor=None, size=8) # , legend=None

    f.set_xlabel('')
    f.set_ylabel('Normalized\nPrediction performance (r)')
    g.set_xlabel('')
    g.set_ylabel('')

    n_subs = 4
    gist_heat = mpl.colormaps['gist_heat'].resampled(n_subs+1)
    sub_cmap = gist_heat(range(n_subs))

    for index, condition in enumerate(corrected_data.exp_condition.unique()):
        for sub_index, sub in enumerate(corrected_data.subject.unique()):
            _data = corrected_data[((corrected_data.exp_condition == condition) & (corrected_data.subject == sub))].groupby(['crop_instance']).corrected_value.mean().reset_index()
            _gcs_mean = _data[_data.crop_instance == 'GCS'].corrected_value.values
            _full_mean = _data[_data.crop_instance == 'Full'].corrected_value.values

            # _axswarm.plot([index-0.2, index+0.2], [_full_mean.mean(), _gcs_mean.mean()], color='gray', alpha=0.8, linewidth=1.5)
            _axswarm.plot([index-0.2, index+0.2], [_full_mean.mean(), _gcs_mean.mean()], color=sub_cmap[sub_index], linewidth=2, label=f'Subj. {sub}' if index == 0 else '')

    _axbox.legend(title='', loc='upper right', ncol=1, bbox_to_anchor=(0.6, 1.2))
    _axswarm.legend(title='', loc='upper right', ncol=3, bbox_to_anchor=(1.1, 1.2))
    # ===========================================================

    sns.despine()

    ax[0, -3].axis('off')
    ax[2, -3].axis('off')
    ax[3, -3].axis('off')

    for _ax in ax[0, :-3]:
        _ax.spines[['top', 'right', 'left', 'bottom']].set_visible(True)
        _ax.set_xticks([])
        _ax.set_yticks([])

    fig.text(-0.2, 1.2, 'a)', size=28, weight='bold', clip_on=False, transform=ax[0, 0].transAxes)
    fig.text(3.5, 1.2, 'b)', size=28, weight='bold', clip_on=False, transform=ax[0, 0].transAxes)
    fig.text(-0.2, 1.2, 'c)', size=28, weight='bold', clip_on=False, transform=ax[-2, 0].transAxes)
    fig.text(3.5, 1.2, 'd)', size=28, weight='bold', clip_on=False, transform=ax[-2, 0].transAxes)

    fig.text(-0.2, 1.1, 'Normalized Prediction performance (r)', va='center', ha='center', rotation='vertical', wrap=True, transform=ax[-1, 0].transAxes)

    fig.subplots_adjust(hspace=0.2)

    # fig.savefig(os.path.join(figure_dir, 'normalized_prediction_performance_fig5.png'), dpi=300, bbox_inches='tight')  

    plt.show()