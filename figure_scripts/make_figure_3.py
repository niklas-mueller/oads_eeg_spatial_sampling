import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import matplotlib as mpl

from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import fdrcorrection
from mne.viz import plot_topomap

from figure_utils import *

if __name__ == '__main__':

    home_path = os.path.expanduser('~')
    result_dir = '/home/nmuller/projects/fmg_storage/oads_experiment_analysis/'

    # figure_dir = '/home/nmuller/projects/fmg_storage/tux20_oads_eeg_paper_figures'
    figure_dir = '/home/nmuller/projects/oads_eeg_spatial_sampling/figures'
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


    # Load data


    # folder = 'encoding_alexnet_imagenet_share-pca_partial-corr_feature-cropping-AutoReject'
    folder = 'encoding_alexnet_feature-croppingAutoReject'
    # df_imagenet_partial = pd.read_parquet(os.path.join(result_dir, f'correct_size_new_fit/encoding_alexnet_imagenet_share-pca_partial-corr_feature-cropping-AutoReject',
    #                     f'all_encoding_model_data_alexnet_imagenet_partial-corr_feature_cropping-AutoReject.parquet'))
    df_imagenet_partial = pd.read_parquet(os.path.join(result_dir, folder, f'all_encoding_model_data_alexnet_imagenet_partial-corr_feature_cropping-AutoReject.parquet'))



    # ACTUAL FIGURE CODE
    partial_all_fig_data = df_imagenet_partial[
        (df_imagenet_partial['metric'] == 'partial_corr') & 
        # (df_imagenet_partial['channel'] == 'Iz') & 
        # (df_imagenet_partial['channel'].isin(['Iz', 'Pz'])) & 
        (df_imagenet_partial['layer'] == 'across-layers') & 
        (df_imagenet_partial['image_representation'] == 'rgb') & 
        
        (((df_imagenet_partial['given'].str.contains('periphery_circ')) &
        (df_imagenet_partial['condition'].str.contains('center_circ')) | 
        (df_imagenet_partial['given'].str.contains('center_circ')) &
        (df_imagenet_partial['condition'].str.contains('periphery_circ'))) |

        ((df_imagenet_partial['given'].str.contains('feature')) &
        (df_imagenet_partial['condition'].str.contains('gcs')) | 
        (df_imagenet_partial['given'].str.contains('gcs')) &
        (df_imagenet_partial['condition'].str.contains('feature'))) | 

        ((df_imagenet_partial['given'].str.contains('gcs')) &
        (df_imagenet_partial['condition'].str.contains('center_circ')) | 
        (df_imagenet_partial['given'].str.contains('center_circ')) &
        (df_imagenet_partial['condition'].str.contains('gcs'))) | 

        ((df_imagenet_partial['given'].str.contains('gcs')) &
        (df_imagenet_partial['condition'].str.contains('periphery_circ')) | 
        (df_imagenet_partial['given'].str.contains('periphery_circ')) &
        (df_imagenet_partial['condition'].str.contains('gcs')))    
        )
        ].copy()

    colormaps = {
        'feature-full': tab10.colors[5],
        'gcs-full': tab10.colors[0],
        'center': tab10.colors[1],
        'periphery': tab10.colors[7],
    }

    def rename(x):
        if 'feature' in x:
            return 'feature'
        elif 'gcs' in x:
            return x
        else:
            return x.replace('fraction_', '').split('_circ')[0] #.replace('_circ_', '')


    partial_all_fig_data['description'] = partial_all_fig_data.apply(lambda x: rename(str(x.condition)) + ' | ' + rename(str(x.given)), axis=1)
    partial_all_fig_data['fraction'] = partial_all_fig_data.apply(lambda x: f"{float(x.given.split('_')[-1]):.1%}" if 'fraction' in str(x.given) else (f"{float(x.condition.split('_')[-1]):.1%}" if 'fraction' in str(x.condition) else '0.5%'), axis=1)

    partial_all_fig_data['row'] = partial_all_fig_data.apply(lambda x: 'center | periphery' if 'center' in x.description and 'periphery' in x.description 
                                                            else ('gcs | feature' if 'gcs' in x.description and 'feature' in x.description 
                                                            else ('center | gcs' if 'center' in x.description and 'gcs' in x.description 
                                                            else ('periphery | gcs' if 'periphery' in x.description and 'gcs' in x.description else ""))), axis=1)


    _data = partial_all_fig_data[partial_all_fig_data['fraction'].isin(['0.5%'])].copy()
    _data.channel = _data.channel.cat.remove_unused_categories()
    _data.reset_index(level=0, inplace=True)

    all_stats = {}

    for channel in ['Iz', 'Pz']:
        all_stats[channel] = {}
        for timepoint in _data.timepoint.unique():
            all_stats[channel][timepoint] = {}
            
            for hue in ['center | periphery', 'periphery | center', 'feature | gcs', 'gcs | feature', 'center | gcs', 'gcs | center', 'periphery | gcs', 'gcs | periphery']:
                vals = _data[(_data['channel'] == channel) & (_data['timepoint'] == timepoint) & (_data['description'] == hue)].value.values
                stats = ttest_1samp(vals, popmean=0)
                all_stats[channel][timepoint][hue] = stats
                
    all_pvals = []
    for channel in ['Iz', 'Pz']:
        for timepoint in _data.timepoint.unique():
            for hue in ['center | periphery', 'periphery | center', 'feature | gcs', 'gcs | feature', 'center | gcs', 'gcs | center', 'periphery | gcs', 'gcs | periphery']:
                all_pvals.append((channel, timepoint, hue, all_stats[channel][timepoint][hue].pvalue))
                
    corrected_pvals = fdrcorrection([x[-1] for x in all_pvals], alpha=0.01)

    all_pvals = {channel: {timepoint: {hue: None for hue in ['center | periphery', 'periphery | center', 'feature | gcs', 'gcs | feature', 'center | gcs', 'gcs | center', 'periphery | gcs', 'gcs | periphery']} for timepoint in _data.timepoint.unique()} for channel in ['Iz', 'Pz']}
    counter = 0
    for channel in ['Iz', 'Pz']:
        for timepoint in _data.timepoint.unique():
            for hue in ['center | periphery', 'periphery | center', 'feature | gcs', 'gcs | feature', 'center | gcs', 'gcs | center', 'periphery | gcs', 'gcs | periphery']:
                all_pvals[channel][timepoint][hue.replace('gcs', 'GCS').replace('center', 'Center').replace('periphery', 'Periphery').replace('feature', 'Full')] = corrected_pvals[1][counter]
                counter = counter + 1



    _data = partial_all_fig_data[partial_all_fig_data['fraction'].isin(['0.5%'])].copy()
    _data = _data[_data['channel'].isin(['Iz', 'Pz'])] # 'TP7', 'TP8'
    _data.channel = _data.channel.cat.remove_unused_categories()
    _data.reset_index(level=0, inplace=True)

    f = sns.relplot(kind='line', data=_data, x='timepoint', y='value', hue='description', col='row', row='channel',  # , '5.0%', '20.0%'
                    linewidth=6, 
                    col_order=['center | periphery', 'center | gcs', 'periphery | gcs', 'gcs | feature'],
                    hue_order = [
                        'center | periphery',
                        'periphery | center',
                        'center | gcs',
                        'gcs | center',
                        'periphery | gcs',
                        'gcs | periphery',
                        'feature | gcs',
                        'gcs | feature',
                    ],
                    palette={
                        'center | periphery': colormaps['center'], 
                        'periphery | center': tab20c.colors[0], # colormaps['periphery'],
                        'feature | gcs': colormaps['feature-full'],
                        'gcs | feature': 'black',
                        'center | gcs': tab20c.colors[4],
                        'gcs | center': 'black', #colormaps['gcs-full'],
                        'periphery | gcs': tab20c.colors[1],
                        'gcs | periphery': 'black' # tab20c.colors[9],
                        })
    f.set_titles('')

    f.set_axis_labels('Time (s)', 'Partial Correlation (r)', fontsize=22)


    handles, lables = [], []
    for _ax in f.axes.flatten():
        _ax.vlines(0.0, linestyles='dashed', ymin=0.0, ymax=0.5, colors='gray', alpha=0.5)
        _ax.hlines(0.0, linestyles='dashed', xmin=-0.1, xmax=0.4, colors='gray', alpha=0.5)

        _handles, _labels = _ax.get_legend_handles_labels()
        for handle, label in zip(_handles, _labels):
            label = label.replace('gcs', 'GCS').replace('feature', 'Full').replace('periphery', 'Periphery').replace('center', 'Center')
            if label not in lables:
                handles.append(handle)
                lables.append(label)

    for ax_index in range(len(f.axes[0].flatten())):
        legend = f.axes.flatten()[ax_index].legend([handles[2*ax_index], handles[2*ax_index+1]], [lables[2*ax_index], lables[2*ax_index+1]], title='', loc='upper left', bbox_to_anchor=(0.05, 1.3), ncol=1, frameon=False, prop={'size': 28})

        for legobj in legend.legendHandles:
            legobj.set_linewidth(5.0)
    f._legend.remove()


    f.axes.flatten()[0].text(-0.1, 1.1, 'b)', transform=f.axes.flatten()[0].transAxes, 
                size=34, weight='bold', clip_on=False)
    f.axes.flatten()[1].text(-0.1, 1.1, 'c)', transform=f.axes.flatten()[1].transAxes, 
                size=34, weight='bold', clip_on=False)
    f.axes.flatten()[2].text(-0.1, 1.1, 'd)', transform=f.axes.flatten()[2].transAxes, 
                size=34, weight='bold', clip_on=False)
    f.axes.flatten()[3].text(-0.1, 1.1, 'e)', transform=f.axes.flatten()[3].transAxes, 
                size=34, weight='bold', clip_on=False)

    _ax0 = f.axes.flatten()[0].inset_axes([-0.8, 0.2, 0.6, 0.6])
    _ax0.set_aspect('equal', anchor="C")
    _ax0.axis('off')
    _ax1 = f.axes.flatten()[4].inset_axes([-0.8, 0.2, 0.6, 0.6])
    _ax1.set_aspect('equal', anchor="C")
    _ax1.axis('off')

    inset_axes = [_ax0, _ax1]
    for ax_index, ch in enumerate(['Iz', 'Pz']):
        info, tvalue_channel_names = make_custom_info(channels)
        mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k',
                linewidth=0, markersize=10)
        channel_mask = np.array([x in [ch] for x in channels])

        _ax = plot_topomap(data=np.zeros((len(channels), )), pos=info, axes=inset_axes[ax_index], mask_params=mask_params, mask=channel_mask, sensors=False, show=False)
        inset_axes[ax_index].set_title(ch, fontweight='bold', fontsize=22)

    #########################################
    palette={
        'Center | Periphery': colormaps['center'], 
        'Periphery | Center': tab20c.colors[0], 
        'Full | GCS': colormaps['feature-full'],
        'GCS | Full': 'black',
        'Center | GCS': tab20c.colors[4],
        'GCS | Center': 'black',
        'Periphery | GCS': tab20c.colors[1],
        'GCS | Periphery': 'black' 
        }
        
    cond_names = {}
    for _axes, ch in [(f.fig.axes[:-4], 'Iz'), (f.fig.axes[4:], 'Pz')]:
        for ax_index, _ax in enumerate(_axes):
        
            for cond_index in range(2):
                if ch == 'Pz':
                    cond_name = cond_names[(ax_index, cond_index)]
                else:
                    cond_name = _ax.get_legend().get_texts()[cond_index].get_text()
                    cond_names[(ax_index, cond_index)] = cond_name
                

                for timepoint in _data.timepoint.unique():
                    pval = all_pvals[ch][timepoint][cond_name]
                    if pval < 0.01:
                        _ax.plot(timepoint, -0.12-cond_index*0.02, '*', color=palette[cond_name])
    #########################################

    f.fig.subplots_adjust(wspace=0.2)
    f.fig.set_size_inches(35, 12)

    # f.fig.savefig(os.path.join(figure_dir, f'imagenet_center_vs_periphery_vs_full_partial_simple.png'), dpi=300., bbox_inches='tight')
    f.fig.savefig(os.path.join(figure_dir, f'figure3.png'), dpi=300., bbox_inches='tight')
    plt.show()