import os
import pickle
import torch
import yaml
import numpy as np
from PIL import Image

import pandas as pd
from mne import read_epochs
from oads_access.oads_access import OADS_Access
from torchvision.models import alexnet, AlexNet_Weights
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from pytorch_utils.pytorch_utils import ToRetinalGanglionCellSampling
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import h5py
# from run_rsa import CustomDataset, CustomOADS
# from run_encoding_model import get_sub_data

target_filenames = ['0434262454981c92.tiff', '0c6c2c66e61e3133.tiff', '0e2e2f2931313d37.tiff', '1332a2d4c4c14322.tiff', '202024d9b333968c.tiff', '24247432670b3131.tiff', '2c2064d163c52c78.tiff', '42431879e191d3c3.tiff', '61662ece1f6e7870.tiff', '7890918716766325.tiff', '85cc4e533959b1e1.tiff', '8ece4062eee68292.tiff', '93939b9a101e87e0.tiff', '948cacbc94b42474.tiff', 'af8f3a3939313171.tiff', 'b23232b232332361.tiff', 'b371f8ecf4e0f0f0.tiff', 'bce2d2d2c393e3b2.tiff', 'c77131717179f173.tiff', 'c93b3b2b29b9b8f1.tiff', 'cc8c9290bcbcf4fc.tiff', 'dad8acb8b82e36b1.tiff', 'e2f8f8fcf8f8f8f8.tiff', 'e6d8d40438c0c8e2.tiff', 'e6e648c62ebacc38.tiff', 'ef3632c2c1476e78.tiff', 'f4ca83dc6731310d.tiff']


def get_sub_data(sub, cleaning='AutoReject'):
    eeg_dir = f'/home/nmuller/projects/fmg_storage/osf_eeg_data/{cleaning.lstrip("-")}'
    with open(os.path.join(eeg_dir, f'filenames_oads_eeg_rsvp_sub-{str(sub).zfill(2)}.pkl'), 'rb') as f:
        filenames = pickle.load(f)

    with open(os.path.join(eeg_dir, f'is_test_oads_eeg_rsvp_sub-{str(sub).zfill(2)}.pkl'), 'rb') as f:
        is_test = pickle.load(f)

    data = np.load(os.path.join(eeg_dir, f'oads_eeg_rsvp_sub-{str(sub).zfill(2)}.npy'))
    
    train_filenames = [filenames[i] for i in range(len(filenames)) if not is_test[i]]
    test_filenames = [filenames[i] for i in range(len(filenames)) if is_test[i]]

    train_ids = [x.split('.')[0] for x in train_filenames if f"{x.split('.')[0]}.tiff" not in target_filenames]
    test_ids = [x.split('.')[0] for x in test_filenames if f"{x.split('.')[0]}.tiff" not in target_filenames]

    print(len(train_ids), len(test_ids))

    train_data = np.array([data[i] for i in range(len(data)) if not is_test[i]])
    test_data = np.array([data[i] for i in range(len(data)) if is_test[i]])

    _, n_channels, n_timepoints = train_data.shape
    sample_rate = 1024
    t = [i/sample_rate - 0.1 for i in range(n_timepoints)]

    return train_data, test_data, train_ids, test_ids, t
    

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, imgs_paths, device, transform):

        self.image_dir = image_dir
        self.transform = transform
        self.imgs_paths = imgs_paths
        self.device = device


    def __len__(self):
        return len(self.imgs_paths)
    
    def __getitem__(self, idx):
        img_path = self.imgs_paths[idx]

        img = Image.open(os.path.join(self.image_dir, f'{img_path}.tiff'))

        img = self.transform(img).to(self.device)

        return img

def main(layer, filename_addon=''):
    
    # model_type = 'alexnet_oads'
    model_type = 'alexnet_imagenet'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if model_type == 'alexnet_imagenet':
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    elif model_type == 'alexnet_oads':
        model = alexnet()
        model.classifier[6] = torch.nn.Linear(4096, 21)
        home_path = os.path.expanduser('~')
        model_path = os.path.join(f'{home_path}/projects/fmg_storage/trained_models/oads_results/alexnet/rgb/2023-06-08-115421/best_model_08-06-23-182404.pth')
        state_dict = torch.load(model_path, map_location=device)
        try:
            model.load_state_dict(state_dict)
        except:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(state_dict)
            model = model.module

        # OADS Crops (400,400) mean, std
        mean = [0.3410, 0.3123, 0.2787]
        std = [0.2362, 0.2252, 0.2162]

    else:
        raise ValueError('Model type not recognized')


    model = model.eval()


    width = 2155
    height = 1440
    ap = height / width
    # size = width, height
    size = height, width

    # if reduce_size:
    new_width = 400 # 600
    new_height = int(new_width * ap)
    # size = new_width, new_height
    size = new_height, new_width

    return_nodes = {
        'features.2': 'layer1',
        'features.5': 'layer2',
        'features.12': 'layer3', # 7
    }
    layers=['layer1', 'layer2', 'layer3']

    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
    feature_extractor = feature_extractor.to(device)

    
    save_predictions = False

    transform_list = [
        transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    transform = transforms.Compose(transform_list)


    gcs_foveas = [5, 10, 20, 25, 30, 2, 7, 12, 15, 18, 22, 28, 32, 35, 40]

    res_dir = os.path.join(f'/home/nmuller/projects/fmg_storage/oads_experiment_analysis/correct_size_new_fit/hyperparameter_optimization_new', model_type)
    os.makedirs(res_dir, exist_ok=True)


    all_activations = {}
    pre_df = None
    gcs = {layer: {} for layer in layers}

    run_subjects = range(5, 36) # 36

    for sub in tqdm(run_subjects, desc='Subs', total=len(run_subjects)):
        rows = []
        cols = ['subject', 'gcs', 'layer', 'channel', 'timepoint', 'metric', 'value']
        
        save_filename = f'sub-{str(sub).zfill(2)}_hyperparameter_optimization_new{filename_addon}.parquet'

        # Get the data
        train_data, test_data, train_filenames, test_filenames, t = get_sub_data(sub)


        ###############################################################
        train_indices = [i for i in range(len(train_filenames)) if train_filenames[i] not in all_activations]
        test_indices = [i for i in range(len(test_filenames)) if test_filenames[i] not in all_activations]
        
        sub_train_filenames = [train_filenames[i] for i in train_indices]
        sub_test_filenames = [test_filenames[i] for i in test_indices]
        ###############################################################

        eeg_dir = f'/home/nmuller/projects/data/oads_eeg/sub_{sub}/sub_{sub}-OC&CSD-AutoReject-epo.fif'
        epochs = read_epochs(fname=eeg_dir, preload=False)
        channel_names = epochs.ch_names
        visual_channel_names = ['O1', 'O2', 'Oz', 'Iz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO7', 'POz', 'PO4', 'PO8', 'I1', 'I2']
        visual_channel_indices = [channel_names.index(ch) for ch in visual_channel_names if ch in channel_names]

        n_channels, n_timepoints = test_data.shape[1:]

        

        train_dataset = ImageDataset(
            image_dir='/home/nmuller/projects/data/oads/oads_arw/tiff',
            imgs_paths=sub_train_filenames,
            device=device,
            transform=transform
        )

        test_dataset = ImageDataset(
            image_dir='/home/nmuller/projects/data/oads/oads_arw/tiff',
            imgs_paths=sub_test_filenames,
            device=device,
            transform=transform
        )

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=36, num_workers=8, shuffle=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=36, num_workers=8, shuffle=False)

        activations = {
            'train': {},
            'test': {},
        }

        for split, _dataloader in zip(['train', 'test'], [train_dataloader, test_dataloader]):
            for images in tqdm(_dataloader, total=len(_dataloader), desc=f'{split} dataloader'):
                # for condition, img in images.items():
                #     if condition not in activations[split]:
                #         activations[split][condition] = {layer: [] for layer in layers}

                images = images.to(device)
                with torch.no_grad():
                    img_act = feature_extractor(images)

                img_act = {k: v.detach().cpu() for k, v in img_act.items()}

                for layer in layers:
                    shape = None

                    for b in range(img_act[layer].shape[0]):
                        if shape is None:
                            shape = img_act[layer][b][0].shape
                            _max = max(shape)
                            out_size = (_max, _max)

                        for gcs_fovea in gcs_foveas:
                            if gcs_fovea not in activations[split]:
                                activations[split][gcs_fovea] = {layer: [] for layer in layers}

                            if gcs_fovea not in gcs[layer]:
                                gcs[layer][gcs_fovea] = ToRetinalGanglionCellSampling(fov=gcs_fovea, image_shape=out_size + ((1,)), out_size=out_size[0], series=1, dtype=np.float32)  # type: ignore

                            gcs_output = []
                            for feature in img_act[layer][b]:
                                gcs_feature = gcs[layer][gcs_fovea](feature.numpy())
                                gcs_output.append(gcs_feature)
                            gcs_output = np.array(gcs_output).flatten()

                            activations[split][gcs_fovea][layer].append(gcs_output)

                        # activations[split][condition][layer].append(img_act[layer][b].numpy().flatten())

        print(f'Done extraction activations')

        for i, filename in enumerate(sub_train_filenames):
            # all_activations[filename] = {cond: {layer: activations['train'][cond][layer][i] for layer in layers} for cond in activations['train'].keys()}
            all_activations[filename] = {gcs_fovea: np.hstack([activations['train'][gcs_fovea][layer][i].flatten() for layer in layers]) for gcs_fovea in activations['train'].keys()}
        for i, filename in enumerate(sub_test_filenames):
            # all_activations[filename] = {cond: {layer: activations['test'][cond][layer][i] for layer in layers} for cond in activations['test'].keys()}
            all_activations[filename] = {gcs_fovea: np.hstack([activations['test'][gcs_fovea][layer][i].flatten() for layer in layers]) for gcs_fovea in activations['test'].keys()}

        print(f'Done storing activations')

        activations = {
            split: {
                gcs_fovea: np.vstack([all_activations[filename][gcs_fovea] for filename in (train_filenames if split == 'train' else test_filenames)]) for gcs_fovea in gcs_foveas
            } for split in ['train', 'test']
        }

        print(f"Train activations: {activations['train'][20].shape}")

        print(f'Done constructing activations')

        for gcs_fovea in gcs_foveas:
            pca = PCA(n_components=100)
            design_matrix = pca.fit_transform(np.array(activations['train'][gcs_fovea]))
            test_design_matrix = pca.transform(np.array(activations['test'][gcs_fovea]))

            lin_reg = LinearRegression()
            lin_reg.fit(design_matrix, train_data.reshape(-1, n_channels * n_timepoints))

            pred = lin_reg.predict(test_design_matrix).reshape(-1, n_channels, n_timepoints)

            if save_predictions:
                with h5py.File(os.path.join(res_dir, f'{sub}_predictions.h5'), 'a') as f:
                    if f'{gcs_fovea}_across-layers' not in f:
                        f.create_dataset(f'{gcs_fovea}_across-layers', data=pred)
                # continue

            for channel in range(n_channels):
                for timepoint in range(n_timepoints):
                    _corr = np.corrcoef(test_data[:, channel, timepoint], pred[:, channel, timepoint])[0, 1]

                    rows.append([sub, gcs_fovea, 'across-layers', channel, t[timepoint], 'test_corr', _corr])

            if pre_df is not None:
                df = pd.concat([pre_df, pd.DataFrame(rows, columns=cols)], ignore_index=True)
            else:
                df = pd.DataFrame(rows, columns=cols)

            df.to_parquet(os.path.join(res_dir, save_filename))          
            # print(f'Saved results to {os.path.join(res_dir, save_filename)}')


if __name__ == '__main__':
    # tiff_filenames = os.listdir('/home/nmuller/projects/data/oads/oads_arw/tiff')

    # total_missing = []

    # for sub in range(5, 36):
    #     # Get the data
    #     train_data, test_data, train_filenames, test_filenames, t = get_sub_data(sub)

    #     train_indices = [i for i in range(len(train_filenames))]
    #     test_indices = [i for i in range(len(test_filenames))]
        
    #     sub_train_filenames = [train_filenames[i] for i in train_indices]
    #     sub_test_filenames = [test_filenames[i] for i in test_indices]

    #     counter = 0
    #     for filename in sub_train_filenames + sub_test_filenames:
    #         if filename not in tiff_filenames:
    #             # print(f'Sub {sub}: Missing file {filename}')
    #             counter += 1
    #             total_missing.append(filename)
    #     print(f'Sub {sub}: Missing {counter} files')

    # print(f'Total unique missing files: {len(set(total_missing))}')
    #     ###############################################################

    torch.multiprocessing.set_start_method('spawn')

    layer = 'layer1'

    main(layer)