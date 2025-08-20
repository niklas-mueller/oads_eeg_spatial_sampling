import sys
sys.path.append('..')
from utils import record_activations
from oads_eeg_spatial_sampling.analysis.GDS import ToRetinalGanglionCellSampling
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mne import create_info
from mne.viz import plot_topomap
from mne.channels import make_dig_montage, make_standard_montage
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
import matplotlib as mpl
tab10 = mpl.colormaps.get_cmap('tab10')
tab20 = mpl.colormaps.get_cmap('tab20')
tab20c = mpl.colormaps.get_cmap('tab20c')
tab20b = mpl.colormaps.get_cmap('tab20b')


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
    mask = np.zeros((size[0], size[1]), dtype=int)
    y, x = np.ogrid[:size[0], :size[1]]
    mask[((x - size[1] // 2) ** 2 + (y - size[0] // 2) ** 2) < center_radius ** 2] = 1

    return mask

def get_rect_mask(shape, fraction):
    center_fraction = fraction
    _fraction = np.sqrt(center_fraction)

    mask_small = np.zeros(shape).astype(int)

    row_center = shape[0] / 2
    row_start = int(np.floor(row_center - row_center * _fraction))
    row_end = int(np.ceil(row_center + row_center * _fraction))
    # row_size = row_end - row_start
    # print(row_start, row_end)

    col_center = shape[1] / 2
    col_start = int(np.floor(col_center - col_center * _fraction))
    col_end = int(np.ceil(col_center + col_center * _fraction))
    # col_size = col_end - col_start
    # print(col_start, col_end)
    
    mask_small[row_start:row_end, col_start:col_end] = 1

    return mask_small

def make_custom_info(topoplot_channel_names, x_spacing = 0.015, y_spacing = 0.085, z_spacing = 0.0):
    info = create_info(ch_names=make_standard_montage('biosemi64').ch_names, sfreq=1024, ch_types='eeg')
    info.set_montage('biosemi64')

    position_dict = info.get_montage().get_positions()

    position_dict['ch_pos'].update({'I1': np.array([x_spacing, -y_spacing, z_spacing]), 'I2': np.array([-x_spacing, -y_spacing, z_spacing])})
    position_dict['ch_pos'].pop('F5')
    position_dict['ch_pos'].pop('F6')

    custom_montage = make_dig_montage(**position_dict)

    topoplot_channel_names = ['I1' if x == 'F5' else ('I2' if x == 'F6' else x) for x in topoplot_channel_names]# + ['I1', 'I2']
    # if 'I1' not in topoplot_channel_names:
    #     topoplot_channel_names.append('I1')
    # if 'I2' not in topoplot_channel_names:
    #     topoplot_channel_names.append('I2')

    info = create_info(ch_names=topoplot_channel_names, sfreq=1024, ch_types='eeg')
    info.set_montage(custom_montage)

    # fig = custom_montage.plot()

    return info, topoplot_channel_names


def fdr_correct_per_timepoint(tvalues, timepoints):
    _, corrected_pvals = fdrcorrection([tvalues[timepoint_index].pvalue for timepoint_index in timepoints])

    l = {timepoint_index: corrected_pvals[counter] for counter, timepoint_index in enumerate(timepoints)}

    return l

def fdr_correct_per_channel_timepoint(tvalues, timepoints, channels):
    _, corrected_pvals = fdrcorrection([tvalues[(timepoint_index, channel)].pvalue for timepoint_index in timepoints for channel in channels])
    l = {}
    counter = 0
    for timepoint_index in timepoints:
        _l = {}
        for channel in channels:
            _l[channel] = corrected_pvals[counter]
            counter += 1
        
        l[timepoint_index] = _l

    return l

def plot_tvalue_topoplots(axes, tvalues, corrected_pvals, timepoints, channels, vlim, t, alpha=0.05, cbar_ax=None, legend_label='t-value fdr-corrected (significant with alpha = {alpha})', names=True, title:bool=True, title_fontsize=24, axis_title_pad=6.0, mask=None, mask_params=None):
    global_min_sig_t_im = None
    global_min_sig_t = np.inf

    for ax_index, timepoint_index in enumerate(timepoints):
        tvalue_data = []
        tvalue_channel_names = []

        min_sig_t = np.inf
        for channel in channels:
            if channel in ['left', 'right', 'above', 'below']:
                continue
            _stat = tvalues[(timepoint_index, channel)]
            tvalue_data.append(_stat.statistic)
            # print(_stat.pvalue)
            if corrected_pvals[timepoint_index][channel] < alpha and np.abs(_stat.statistic) < np.abs(min_sig_t):
                min_sig_t = _stat.statistic
            tvalue_channel_names.append(channel)
        # print(min_sig_t)
        info, tvalue_channel_names = make_custom_info(tvalue_channel_names)

        p = [vlim[0], -np.abs(min_sig_t), np.abs(min_sig_t), vlim[1]]
        f = lambda x: np.interp(x, p, [0, 0.5, 0.5, 1])

        # cmap = 'seismic'
        cmap = LinearSegmentedColormap.from_list('map_white', 
                    list(zip(np.linspace(0,1), plt.cm.seismic(f(np.linspace(min(p), max(p)))))))
        im,_ = plot_topomap(tvalue_data, info, axes=axes[ax_index], names=tvalue_channel_names if type(names) is bool and names else (names if type(names) is list and ax_index == 0 else None), 
                            vlim=vlim, show=False, cmap=cmap, mask=mask, mask_params=mask_params) #, image_interp='nearest')
        if np.abs(min_sig_t) < np.abs(global_min_sig_t):
            global_min_sig_t_im = im

        if title:
            axes[ax_index].set_title(f'{int(t[timepoint_index]*1000):d} ms', pad=axis_title_pad, fontweight='bold', fontsize=title_fontsize)

    # ax_x_start = 0.95
    # ax_x_width = 0.04
    # ax_y_start = 0.1
    # ax_y_height = 0.8

    # # cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    if cbar_ax is not None and global_min_sig_t_im is not None:
        cbar = plt.colorbar(global_min_sig_t_im, cax=cbar_ax)
        cbar.set_label(legend_label.format(alpha=alpha))

    # fig.suptitle('T-Test: RGB > COC?')

    # plt.show()
    # return axes


def get_fdr_correct_pvals(_data, channels, alpha, colors, crop_instances=['feature-full', 'center', 'periphery', 'gcs-full']):
    # channel = 'Iz'
    _tval_data = _data[(_data['crop_instance'].isin(crop_instances))].copy()
    _tval_data.channel = _tval_data.channel.cat.remove_unused_categories()
    _tval_data = _tval_data.groupby(['timepoint', 'subject', 'crop_instance', 'channel']).value.mean().reset_index()
    
    
    timepoints = _tval_data[_tval_data.timepoint > 0.0].timepoint.unique()
    # crop_instances = _tval_data.crop_instance.unique()

    condition_stats = {}
    all_pvals = []

    for channel in channels:
        condition_stats[channel] = {}
        for crop_instance in crop_instances:
            condition_stats[channel][crop_instance] = {}
            for timepoint in timepoints:
                vals = _tval_data[((_tval_data['crop_instance'] == crop_instance) & (_tval_data['timepoint'] == timepoint) & (_tval_data.channel == channel))].value.values
                if len(vals) < 5:
                    # print(f'Not enough data for {channel} {crop_instance} {timepoint}')
                    res = None
                    all_pvals.append((crop_instance, timepoint, channel, 1.0))
                else:
                    res = ttest_1samp(vals, popmean=0, alternative='greater')
                    condition_stats[channel][crop_instance][timepoint] = res
                    all_pvals.append((crop_instance, timepoint, channel, res.pvalue))
    
    
    fdr_corrected_pvals = fdrcorrection([x[-1] for x in all_pvals], alpha=alpha)
    
    for index, (crop_instance, timepoint, channel, _) in enumerate(all_pvals):
        condition_stats[channel][crop_instance][timepoint] = fdr_corrected_pvals[1][index]

    return condition_stats


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
    mask = np.zeros((size[0], size[1]), dtype=int)
    y, x = np.ogrid[:size[0], :size[1]]
    mask[((x - size[1] // 2) ** 2 + (y - size[0] // 2) ** 2) < center_radius ** 2] = 1

    return mask

def get_rect_mask(shape, fraction):
    center_fraction = fraction
    _fraction = np.sqrt(center_fraction)

    mask_small = np.zeros(shape).astype(int)

    row_center = shape[0] / 2
    row_start = int(np.floor(row_center - row_center * _fraction))
    row_end = int(np.ceil(row_center + row_center * _fraction))
    # row_size = row_end - row_start
    # print(row_start, row_end)

    col_center = shape[1] / 2
    col_start = int(np.floor(col_center - col_center * _fraction))
    col_end = int(np.ceil(col_center + col_center * _fraction))
    # col_size = col_end - col_start
    # print(col_start, col_end)
    
    mask_small[row_start:row_end, col_start:col_end] = 1

    return mask_small

def get_dnn_feature_examples():
    import rawpy
    from torchvision.models import alexnet, AlexNet_Weights

    arw_dir = '/home/nmuller/projects/data/oads/oads_arw/ARW'
    image_filenames = os.listdir(arw_dir)

    images = []
    for index in range(4):
        with open(os.path.join(arw_dir, image_filenames[index]), 'rb') as f:
            img = rawpy.imread(f)
            img = img.postprocess()
            img = Image.fromarray(img).reduce(4)

            images.append((img, image_filenames[index]))

    use_rgbedges = False
    use_cocedges = False

    output_channels = 21
    gpu_name = 'cuda:0'
    device = torch.device(gpu_name if torch.cuda.is_available() else 'cpu')

    return_nodes = {
        'features.2': 'layer1',
        'features.5': 'layer2',
        'features.12': 'layer3',
        # 'classifier.5': 'feature',
    }

    model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

    model = model.to(device)

    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)


    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)


    trainloader = [[transform(img).to(device).unsqueeze(0), None, torch.tensor([index]).to(device), image_name] for index, (img, image_name) in enumerate(images)]
    activations = record_activations(loader=trainloader, models=[('alexnet', feature_extractor)], device=device, flatten=False, layer_names=return_nodes.values())

    # feature = activations['alexnet_layer1'][0][17]

    return activations

def get_feature_axes(ax, set_title, feature=None, axis_off:bool=True, cmap='gray'):

    if feature is None:
        activations = get_dnn_feature_examples()
        image_name = list(activations['alexnet_layer1'].keys())[0]
        feature = activations['alexnet_layer1'][image_name][17]

    feature_shape = feature.shape
    out_size = (feature_shape[1], feature_shape[1])
    gcs_feature = ToRetinalGanglionCellSampling(image_shape=out_size + ((1,)), out_size=feature_shape[1], series=1, dtype=np.float32)
    
    fraction = np.sqrt(0.005)
    shape = feature.shape
    mask = np.zeros(shape).astype(bool)
    row_center = shape[0] / 2
    row_start = int(row_center - row_center * fraction)
    row_end = int(row_center + row_center * fraction)
    row_size = row_end - row_start

    col_center = shape[1] / 2
    col_start = int(col_center - col_center * fraction)
    col_end = int(col_center + col_center * fraction)
    col_size = col_end - col_start

    mask[row_start:row_end, col_start:col_end] = True
    whole_size = shape[0]*shape[1]



    # Full Scene
    ax[0].imshow(feature, cmap=cmap)
    brown_patch = mpatches.Rectangle((-1, -1), shape[1]+1, shape[0]+1, edgecolor=tab10.colors[5], facecolor='none', linewidth=8, clip_on=False, zorder=100)
    ax[0].add_patch(brown_patch)

    # Center
    ax[1].imshow(np.where(mask, feature, feature.max()), cmap=cmap)
    orange_patch = mpatches.Rectangle((col_start-2, row_start-2), col_size+3, row_size+3, edgecolor=tab10.colors[1], facecolor='none', linewidth=8, zorder=100)
    ax[1].add_patch(orange_patch)

    # Periphery
    mask = ~mask
    ax[2].imshow(np.where(mask, feature, feature.max()), cmap=cmap)
    
    blue_patch = mpatches.Rectangle((col_start-2, row_start-2), col_size+3, row_size+3, edgecolor=tab10.colors[0], facecolor='none', linewidth=8, zorder=100)
    ax[2].add_patch(blue_patch)
    blue_surround_patch = mpatches.Rectangle((-1, -1), shape[1]+1, shape[0]+1, edgecolor=tab10.colors[0], facecolor='none', linewidth=8, clip_on=False, zorder=100)
    ax[2].add_patch(blue_surround_patch)

    # # GCS
    ax[3].imshow(gcs_feature(feature), cmap=cmap)
    black_patch = mpatches.Rectangle((-1, -1), shape[1]+1, shape[1]+1, edgecolor='black', facecolor='none', linewidth=8, clip_on=False, zorder=100)
    ax[3].add_patch(black_patch)
    
    if axis_off:
        for _ax in ax.flatten():
            _ax.axis('off')
    else:
        for _ax in ax.flatten():
            _ax.set_xticks([])
            _ax.set_yticks([])

    if set_title:
        ax[0].set_title('Full', fontsize=40)
        ax[1].set_title('Center', fontsize=40)
        ax[2].set_title('Periphery', fontsize=40)
        ax[3].set_title('GCS', fontsize=40)
