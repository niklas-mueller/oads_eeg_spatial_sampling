import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from scipy.stats import ttest_1samp
import matplotlib.patches as mpatches
from mne.viz import plot_topomap
import pickle
import tqdm
from mne import read_epochs
from figure_utils import *

if __name__ == '__main__':

    home_path = os.path.expanduser('~')
    # result_dir = '/home/nmuller/projects/fmg_storage/oads_experiment_analysis/'
    result_dir = '/home/nmuller/projects/oads_eeg_spatial_sampling/results'

    # figure_dir = '/home/nmuller/projects/fmg_storage/tux20_oads_eeg_paper_figures'
    figure_dir = '/home/nmuller/projects/oads_eeg_spatial_sampling/figures'
    os.makedirs(figure_dir, exist_ok=True)

    tab10 = mpl.colormaps.get_cmap('tab10')
    tab20 = mpl.colormaps.get_cmap('tab20')
    tab20c = mpl.colormaps.get_cmap('tab20c')
    tab20b = mpl.colormaps.get_cmap('tab20b')

    gist_heat = mpl.colormaps.get_cmap('gist_heat')
    gist_heat = [gist_heat(i) for i in np.linspace(0,1,5,endpoint=False)]

    sns.set(font_scale=1.5)
    sns.set_style('ticks')

    plt.rcParams.update({
        'axes.titlesize': 'large',
        'axes.labelsize': 'large',
        'xtick.labelsize':'large',
        'ytick.labelsize':'large'
        })

    eeg_dir = f'{home_path}/projects/data/oads_eeg/sub_13/sub_13-OC&CSD-AutoReject-epo.fif'
    epochs = read_epochs(fname=eeg_dir, preload=False)
    n_channels = len(epochs.ch_names)
    n_timepoints = len(epochs.times)
    sample_rate = 1024
    t = [i/sample_rate - 0.1 for i in range(n_timepoints)]
    channel_names = epochs.ch_names
    visual_channel_names = ['O1', 'O2', 'Oz', 'Iz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO7', 'Pz', 'POz', 'PO4', 'PO8', 'I1', 'I2']

    # Load data
    # folder = 'encoding_alexnet_imagenet_share-pca_partial-corr_feature-cropping-AutoReject'
    folder = 'encoding_alexnet_feature-croppingAutoReject'
    # df_imagenet = pd.read_parquet(os.path.join(result_dir, 'correct_size_new_fit', folder, f'all_encoding_model_data_alexnet_imagenet_feature_cropping-AutoReject.parquet'))
    df_imagenet = pd.read_parquet(os.path.join(result_dir, folder, f'all_encoding_model_data_feature_cropping-AutoReject.parquet'))

    _main_data = df_imagenet[(df_imagenet['metric'] == 'test_corr_train') & 
                            (df_imagenet['layer'] == 'across-layers') & 
                            (df_imagenet['crop_condition'].isin(['feature', 'fraction', 'gcs']) & 
                            (df_imagenet['crop_instance'].isin(['feature-full', 'center', 'periphery', 'gcs-full'])))].copy()

    _main_data.crop_condition = _main_data.crop_condition.cat.remove_unused_categories()
    _main_data.crop_instance = _main_data.crop_instance.cat.remove_unused_categories()

    # Load Noise Ceiling
    noise_ceiling_dir = '/home/nmuller/projects/fmg_storage/oads_experiment_analysis/oads_eeg'
    # noise_ceing_result_manager = ResultManager(root='/home/nmuller/projects/fmg_storage/oads_experiment_analysis/oads_eeg')
    noise_ceiling_filename = 'correlation_repetition_noise_ceiling_no_intercept.pkl'
    # noise_ceiling = noise_ceing_result_manager.load_result(filename='correlation_repetition_noise_ceiling_no_intercept.pkl')
    with open(os.path.join(noise_ceiling_dir, noise_ceiling_filename), 'rb') as f:
        noise_ceiling = pickle.load(f)

    upper_bound_per_electrode = noise_ceiling['upper_bound']
    lower_bound_per_electrode = noise_ceiling['lower_bound']
    upper_corr_subs = noise_ceiling['upper_corr_subs']
    lower_corr_subs = noise_ceiling['lower_corr_subs']


    new_channel_names = ['I1' if x == 'F5' else ('I2' if x == 'F6' else x) for x in channel_names]
    oads_ch_indices = []
    for ch in visual_channel_names:
        # if ch not in new_channel_names:
        #     print(ch)
        oads_ch_index = [i for i, x in enumerate(new_channel_names) if x == ch][0]
        oads_ch_indices.append(oads_ch_index)
        
    visual_channel_lower_bound_average = lower_bound_per_electrode[:, oads_ch_indices].mean(axis=1)
    visual_channel_upper_bound_average = upper_bound_per_electrode[:, oads_ch_indices].mean(axis=1)

    # ACTUAL FIGURE CODE
    channels = _main_data.channel.unique()
    channel_mask = np.array([x in ['Iz', 'Pz'] for x in channels])
    mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k',
            linewidth=0, markersize=6)
    channel_names = None # [x if channel_mask[i] else '' for i, x in enumerate(channels)]


    timepoints = [200, 225, 250]
    do_channels = ['Iz', 'Pz', 'Oz', 'CPz']

    hue_order = ['feature-full', 'center', 'periphery', 'gcs-full']

    colormaps = {
        'feature-full': tab10.colors[5],
        'gcs-full': 'black', #tab10.colors[0],
        'center': tab10.colors[1],
        'periphery': tab10.colors[0], #tab10.colors[7],
    }

    linewidth = 3
    n_rows = 6
    n_cols = 8 # 3 # len(timepoints) + 4

    # fig, ax = plt.subplots(n_rows, n_cols, figsize=(30, 18), gridspec_kw={"width_ratios":[10,8,8,8,8,8,8,2], 'height_ratios':[2,2,2,2,3,3]})
    # =========================================================
    fig = plt.figure(layout='constrained', figsize=(33, 24))
    # fig = plt.figure(layout='constrained', figsize=(25, 18))

    subfigs = fig.subfigures(1, 2, width_ratios=[1, 7], wspace=0.1)

    feature_plots = subfigs[0].subplots(4, 1, gridspec_kw={'height_ratios': [2,2,2,3]}).flatten()

    subfigsnest = subfigs[1].subfigures(2, 1, height_ratios=[3, 2], hspace=0.2)
    lineplots = subfigsnest[0].subplots(2, 2, sharey=True).flatten()

    topoplot_subfigs = subfigsnest[1].subfigures(1, 2, width_ratios=[8, 9], wspace=0.15)

    axsnest2 = topoplot_subfigs[0].subplots(2, 3, sharey=True, gridspec_kw={'wspace': 0})
    axsnest3 = topoplot_subfigs[1].subplots(2, 4, sharey=True, gridspec_kw={'wspace': -0.1, 'width_ratios': [8, 8, 8, 2]})
    axsnest4 = axsnest3[:, -1]
    gs = axsnest4[0].get_gridspec()
    for _ax in axsnest4:
        _ax.remove()
    axsnest4 = fig.add_subplot(gs[:, -1])
    axsnest3 = axsnest3[:, :-1]
    # axsnest4 = topoplot_subfigs[2].subplots(1, 1, sharey=True, gridspec_kw={'wspace': -0.65})
    # =========================================================

    info, tvalue_channel_names = make_custom_info(channels)
    mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k',
            linewidth=0, markersize=6)

    # row_index = 0
    counter = 0
    for ch_index, channel in enumerate(do_channels):
        # _ax = all_axes[counter]
        _ax = lineplots[counter]
        counter += 1

        # ================ channel insets
        iax = _ax.inset_axes([0.1, 0.6, .25, .25])
        iax.set_aspect('equal', anchor="NW")
        channel_mask = np.array([x in [channel] for x in channels])
        iax.axis('off')
        _ = plot_topomap(data=np.zeros((len(channels), )), pos=info, axes=iax, mask_params=mask_params, mask=channel_mask, sensors=False, show=False)
        # ==============================


        if ch_index%2 == 0:
            # r'$\bf
            _ax.set_ylabel('Prediction Performance', fontsize=30)
            _ax.set_xlabel(' ')
        else:
            _ax.set_xlabel(' ')
            _ax.set_ylabel(' ')
            # _ax.set_yticklabels(['' for _ in _ax.get_yticks()])

        # if row_index == 0:
        #     _ax.set_xticks([])

        f = sns.lineplot(data=_main_data[(_main_data['channel'] == channel) & (_main_data['crop_instance'].isin(hue_order))], 
                        ax=_ax, x='timepoint', y='value', hue='crop_instance', #errorbar='se', 
                        hue_order=hue_order,
                        linewidth=linewidth,
                        palette=colormaps,
                        )
        
        f.legend().set_visible(False)
        _ax.set_title(channel, fontweight='bold')
        oads_ch_index = [i for i, x in enumerate(new_channel_names) if x == channel][0]
        
        _ax.fill_between(t, lower_bound_per_electrode[:, oads_ch_index], upper_bound_per_electrode[:, oads_ch_index], color='gray', alpha=0.3, edgecolor='white')
        _ax.vlines(0.0, linestyles='dashed', ymin=0.0, ymax=0.8, colors='gray', alpha=0.5)


    ###################### STATS
    pval_colors = [colormaps[x] for x in hue_order]
    condition_stats = get_fdr_correct_pvals(_main_data, channels=do_channels, alpha=0.01, crop_instances=hue_order, colors=pval_colors)

    for ax_index, channel in enumerate(do_channels):
        for timepoint in _main_data[_main_data.timepoint > 0.0].timepoint.unique():
            for cond_index, condition in enumerate(hue_order):
                if condition_stats[channel][condition][timepoint] < 0.01:
                    lineplots[ax_index].plot(timepoint, -0.08-cond_index*0.02, '*', color=pval_colors[cond_index])
    ######################

    import warnings

    #### Full vs Center
    all_tvalues = {}
    tvalues = {}
    for timepoint_index in timepoints:
        for channel in tqdm.tqdm(channels):
            ttest_data = _main_data[(_main_data['timepoint_index'] == timepoint_index) & (_main_data['channel'] == channel)]

            ttest_data_full = ttest_data[(ttest_data['crop_instance'] == 'feature-full')].groupby('subject').value.mean().to_list() #[:22]
            ttest_data_center = ttest_data[(ttest_data['crop_instance'] == 'center') & (ttest_data['fraction'] == 0.05)].groupby('subject').value.mean().to_list() #[:22]

            diff = np.array(ttest_data_full) - np.array(ttest_data_center)
            res = ttest_1samp(diff, popmean=0)

            tvalues[(timepoint_index, channel)] = res
    all_tvalues['full_vs_center'] = tvalues


    #### Full vs. Periphery
    tvalues = {}
    for timepoint_index in timepoints:
        for channel in tqdm.tqdm(channels):
            ttest_data = _main_data[(_main_data['timepoint_index'] == timepoint_index) & (_main_data['channel'] == channel)]

            ttest_data_full = ttest_data[(ttest_data['crop_instance'] == 'feature-full')].groupby('subject').value.mean().to_list() #[:22]
            ttest_data_periphery = ttest_data[(ttest_data['crop_instance'] == 'periphery') & (ttest_data['fraction'] == 0.05)].groupby('subject').value.mean().to_list() #[:22]

            diff = np.array(ttest_data_full) - np.array(ttest_data_periphery)
            res = ttest_1samp(diff, popmean=0)

            tvalues[(timepoint_index, channel)] = res

    all_tvalues['full_vs_periphery'] = tvalues


    # #########################################
    get_feature_axes(np.array(feature_plots), set_title=True, axis_off=False)
    #########################################

    #### GCS vs Full
    tvalues = {}
    for timepoint_index in timepoints:
        for channel in tqdm.tqdm(channels):
            ttest_data = _main_data[(_main_data['timepoint_index'] == timepoint_index) & (_main_data['channel'] == channel)]

            ttest_data_gcs = ttest_data[(ttest_data['crop_instance'] == 'gcs-full')].groupby('subject').value.mean().to_list() #[:22]
            ttest_data_full = ttest_data[(ttest_data['crop_instance'] == 'feature-full')].groupby('subject').value.mean().to_list() #[:22]

            diff = np.array(ttest_data_gcs) - np.array(ttest_data_full)
            res = ttest_1samp(diff, popmean=0)

            tvalues[(timepoint_index, channel)] = res

    all_tvalues['gcs_vs_full'] = tvalues

    #### GCS vs. Center-Crop
    tvalues = {}
    for timepoint_index in timepoints:
        for channel in tqdm.tqdm(channels):
            ttest_data = _main_data[(_main_data['timepoint_index'] == timepoint_index) & (_main_data['channel'] == channel)]

            ttest_data_gcs = ttest_data[(ttest_data['crop_instance'] == 'gcs-full')].groupby('subject').value.mean().to_list()
            ttest_data_center = ttest_data[(ttest_data['crop_instance'] == 'center') & (ttest_data['fraction'] == 0.05)].groupby('subject').value.mean().to_list()

            diff = np.array(ttest_data_gcs) - np.array(ttest_data_center)
            res = ttest_1samp(diff, popmean=0)

            tvalues[(timepoint_index, channel)] = res

    ####################################################################### NEW WAY OF ASSIGNINING COLORMAP FOR TOPOPLOTS
    all_tvalues['gcs_vs_center'] = tvalues
    _x = []
    for key, values in all_tvalues.items():
        _x.extend([x.statistic for x in values.values()])
    vlim = max(np.abs(min(_x)), np.abs(max(_x)))
    vlim = (-vlim, vlim)

    channel_mask = np.array([False for x in channels])

    corrected_pvals = fdr_correct_per_channel_timepoint(tvalues=all_tvalues['full_vs_center'], timepoints=timepoints, channels=channels)
    # topoplot_axes = [ax[4, 1+i] for i in range(len(timepoints))]
    topoplot_axes = axsnest2[0, :]
    topoplot_axes[1].annotate('Full > Center', xy=(0.5, 1.0), fontsize=28, annotation_clip=False, xycoords='axes fraction', ha='center')
    blue_patch = mpatches.Rectangle(color='blue', xy=(0.16,0.12), width=0.02, height=0.02, clip_on=False)
    red_patch = mpatches.Rectangle(color='red', xy=(-0.18,0.12), width=0.02, height=0.02, clip_on=False)
    topoplot_axes[1].add_artist(blue_patch)
    topoplot_axes[1].add_artist(red_patch)
    plot_tvalue_topoplots(axes=topoplot_axes, tvalues=all_tvalues['full_vs_center'], corrected_pvals=corrected_pvals, timepoints=timepoints, channels=channels, vlim=vlim, t=t, alpha=0.05, legend_label='', names=channel_names, axis_title_pad=30, title_fontsize=28, mask=channel_mask, mask_params=mask_params)



    corrected_pvals = fdr_correct_per_channel_timepoint(tvalues=all_tvalues['full_vs_periphery'], timepoints=timepoints, channels=channels)
    # topoplot_axes = [ax[5, 1+i] for i in range(len(timepoints))]
    topoplot_axes = axsnest2[1, :]
    # gs = ax[5, -1].get_gridspec()
    topoplot_axes[1].annotate('Full > Periphery', xy=(0.5, 1.0), fontsize=28, annotation_clip=False, xycoords='axes fraction', ha='center')
    blue_patch = mpatches.Rectangle(color='blue', xy=(0.16,0.12), width=0.02, height=0.02, clip_on=False)
    red_patch = mpatches.Rectangle(color='red', xy=(-0.18,0.12), width=0.02, height=0.02, clip_on=False)
    topoplot_axes[1].add_artist(blue_patch)
    topoplot_axes[1].add_artist(red_patch)
    plot_tvalue_topoplots(axes=topoplot_axes, tvalues=all_tvalues['full_vs_periphery'], corrected_pvals=corrected_pvals, timepoints=timepoints, channels=channels, vlim=vlim, t=t, alpha=0.05, names=channel_names, title=False, mask=channel_mask, mask_params=mask_params)



    corrected_pvals = fdr_correct_per_channel_timepoint(tvalues=all_tvalues['gcs_vs_full'], timepoints=timepoints, channels=channels)
    # topoplot_axes = [ax[4, 4+i] for i in range(len(timepoints))]
    topoplot_axes = axsnest3[0, :]
    topoplot_axes[1].annotate('GCS > Full', xy=(0.5, 1.0), fontsize=28, annotation_clip=False, xycoords='axes fraction', ha='center')
    blue_patch = mpatches.Rectangle(color='blue', xy=(0.16,0.12), width=0.02, height=0.02, clip_on=False)
    red_patch = mpatches.Rectangle(color='red', xy=(-0.18,0.12), width=0.02, height=0.02, clip_on=False)
    topoplot_axes[1].add_artist(blue_patch)
    topoplot_axes[1].add_artist(red_patch)
    plot_tvalue_topoplots(axes=topoplot_axes, tvalues=all_tvalues['gcs_vs_full'], corrected_pvals=corrected_pvals, timepoints=timepoints, channels=channels, vlim=vlim, t=t, alpha=0.05, names=channel_names, axis_title_pad=30, title_fontsize=28, mask=channel_mask, mask_params=mask_params)


    corrected_pvals = fdr_correct_per_channel_timepoint(tvalues=all_tvalues['gcs_vs_center'], timepoints=timepoints, channels=channels)
    # topoplot_axes = [ax[5, 4+i] for i in range(len(timepoints))]
    topoplot_axes = axsnest3[1, :]
    # gs = ax[5, -1].get_gridspec()
    # cbar_ax = fig.add_subplot(gs[4:, -1])
    cbar_ax = axsnest4
    topoplot_axes[1].annotate('GCS > Center', xy=(0.5, 1.0), fontsize=28, annotation_clip=False, xycoords='axes fraction', ha='center')
    blue_patch = mpatches.Rectangle(color='blue', xy=(0.16,0.12), width=0.02, height=0.02, clip_on=False)
    red_patch = mpatches.Rectangle(color='red', xy=(-0.18,0.12), width=0.02, height=0.02, clip_on=False)
    topoplot_axes[1].add_artist(blue_patch)
    topoplot_axes[1].add_artist(red_patch)
    plot_tvalue_topoplots(axes=topoplot_axes, tvalues=all_tvalues['gcs_vs_center'], corrected_pvals=corrected_pvals, timepoints=timepoints, 
                            legend_label='t-value fdr-corrected\n(significant with alpha = {alpha})',
                            channels=channels, vlim=vlim, t=t, alpha=0.05, cbar_ax=cbar_ax, names=channel_names, title=False, mask=channel_mask, mask_params=mask_params)


    #################################
    for timepoint in timepoints:
        # for __ax in all_axes:
        # for __ax in lineplots:
        lineplots[0].vlines(t[timepoint], linestyles='dashed', ymin=-0.05, ymax=0.9, colors='r', alpha=0.8, zorder=-10)

    # for _ax in ax.flatten():
    #     _ax.set_xticks([])
    #     _ax.set_yticks([])



    if True:
        handles, lables = [], []
        # for _ax in all_axes:
        for _ax in lineplots:
            _handles, _labels = _ax.get_legend_handles_labels()
            
            for handle, label in zip(_handles, _labels):
                if label == 'feature-full':
                    label = 'Full'
                elif 'center' in label:
                    # label = label.replace('center-crop-', 'Center-Crop\n')
                    label = 'Center 0.5%'
                elif 'periphery' in label:
                    # label = label.replace('random-crop-', 'Random-Crop\n')
                    label = 'Periphery 99.5%'
                # elif 'gcs-tested' in label:
                #     label = 'GCS'
                elif 'gcs' in label:
                    label = 'GCS'


                if label not in lables:
                    handles.append(handle)
                    lables.append(label)

        lables.append('Noise Ceiling')
        gray_patch = mpatches.Patch(edgecolor='lightgray', facecolor='lightgray')
        handles.append(gray_patch)
        # all_axes =ax.flatten()
        legend = fig.legend(handles,lables, title='', loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(handles), frameon=False, prop={'size': 30}) # all_axes[0]
        for legobj in legend.legend_handles:
            legobj.set_linewidth(5.0)
        # all_axes[0].set_zorder(100)
        lineplots[0].set_zorder(100)
        # all_axes[1].get_shared_x_axes().join(all_axes[0], *all_axes)
        # all_axes[1].get_shared_y_axes().join(all_axes[0], *all_axes)

        # fig.set_dpi(300.)
        # fig.set_size_inches(6,4)
        sns.despine()

    # for _ax in mask_axes:
    for _ax in feature_plots:
        _ax.spines[['top', 'right']].set_visible(True)
    # # plt.subplots_adjust(left=0.25)

    # feature_plots[0].text(0.0, 1.0, 'a)', color='black', size=40, weight='bold', clip_on=False, zorder=15, transform=fig.transFigure)
    # feature_plots[0].text(0.15, 1.0, 'b)', color='black', size=40, weight='bold', clip_on=False, zorder=15, transform=fig.transFigure)
    # feature_plots[0].text(0.15, 0.33, 'c)', color='black', size=40, weight='bold', clip_on=False, zorder=15, transform=fig.transFigure)
    # feature_plots[0].text(0.15, 0.15, 'd)', color='black', size=40, weight='bold', clip_on=False, zorder=15, transform=fig.transFigure)
    # feature_plots[0].text(0.6, 0.33, 'e)', color='black', size=40, weight='bold', clip_on=False, zorder=15, transform=fig.transFigure)
    # feature_plots[0].text(0.6, 0.15, 'f)', color='black', size=40, weight='bold', clip_on=False, zorder=15, transform=fig.transFigure)

    # fig.savefig(os.path.join(figure_dir, f'main_fig_imagenet_lineplots.png'), dpi=300., bbox_inches='tight')
    fig.savefig(os.path.join(figure_dir, f'figure2.png'), dpi=300., bbox_inches='tight')

    # plt.show()