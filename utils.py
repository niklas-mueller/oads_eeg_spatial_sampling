import os
import torch
import tqdm
import numpy as np
import multiprocessing
from torch.utils.data import Dataset
import rawpy
from PIL import Image

class CustomOADS():
    """
    CustomOADS class for loading and processing images from the OADS dataset.
    
    Args:
        basedir (str): The base directory of the dataset.
        n_processes (int): The number of processes to use for image loading.
    
    Attributes:
        basedir (str): The base directory of the dataset.
        image_dir (str): The directory containing the image files.
        n_processes (int): The number of processes to use for image loading.
        image_names (list): A list of image names in the dataset.
    """
    
    def __init__(self, basedir, n_processes):
        self.basedir = basedir
        self.image_dir = os.path.join(basedir, 'oads_arw', 'ARW')
        self.n_processes = n_processes

        self.image_names = os.listdir(self.image_dir)

    def load_image(self, image_name):
        """
        Load and process an image from the dataset.
        
        Args:
            image_name (str): The name of the image to load.
        
        Returns:
            tuple: A tuple containing the loaded image and its label.
        """
        with rawpy.imread(os.path.join(self.image_dir, f'{image_name}.ARW')) as raw:
            img = raw.postprocess()
            img = Image.fromarray(img)
        
        label = ''
        tup = (img, label)

        return tup

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class OADSImageDataset(Dataset):
    def __init__(self, oads_access, use_crops: bool, item_ids: list,
                 class_index_mapping: dict = None, transform=None, target_transform=None,
                 device='cuda:0', target: str = 'label', force_recompute:bool=False, preload_all:bool=False,
                 return_index:bool=False) -> None:
        super().__init__()

        self.oads_access = oads_access
        self.use_crops = use_crops
        self.item_ids = item_ids

        self.transform = transform
        self.target_transform = target_transform
        self.class_index_mapping = class_index_mapping
        self.device = device

        self.target = target
        self.force_recompute = force_recompute
        self.preload_all = preload_all

        self.return_index = return_index

        self.tupels = {}

        # if self.preload_all:
        #     print(f'Preloading {len(item_ids)} items.')
        #     with multiprocessing.Pool(oads_access.n_processes) as pool:
        #         # results = list(tqdm.tqdm(pool.map(self.iterate, [idx for idx in range(len(item_ids))]), total=len(item_ids)))
        #         results = list(tqdm.tqdm(pool.imap(self.iterate, [idx for idx in range(len(item_ids))]), total=len(item_ids)))

        #     # results = p_map(self.iterate, [idx for idx in range(len(item_ids))])
        #     for idx, tup in results:
        #         self.tupels[idx] = tup
            

    def iterate(self, idx):
        if self.use_crops:
            image_name, index = self.item_ids[idx]
            tup = self.oads_access.load_crop_from_image(
                image_name=image_name, index=index, force_recompute=self.force_recompute)
        else:
            image_name = self.item_ids[idx]
            tup = self.oads_access.load_image(image_name=image_name)

        if tup is None:
            return None, None
        
        # tup[0].load()
        return idx, tup

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        # if self.preload_all:
        #     tup = self.tupels[idx]
        # else:
        if self.use_crops:
            image_name, index = self.item_ids[idx]
            tup = self.oads_access.load_crop_from_image(
                image_name=image_name, index=index, force_recompute=self.force_recompute)
        else:
            image_name = self.item_ids[idx]
            tup = self.oads_access.load_image(image_name=image_name)

        if tup is None:
            return None

        img, label = tup
        del tup

        if img is None or label is None:
            return None

        if self.transform:
            img = self.transform(img)

        img = img.float()

        if self.target == 'label':
            label = label['classId']
            if self.class_index_mapping is not None:
                label = self.class_index_mapping[label]

        elif self.target == 'image':
            label = img

        else:
            label = np.array([])

        if self.target_transform:
            label = self.target_transform(label)
        if self.return_index:
            return (img, label, idx, image_name)
        
        return (img, label)

def record_activations(loader, models:tuple, device, verbose:bool=False, layer_names:list=[''], extract_pixel_layer:bool=False, flatten:bool=True, activations_cache:dict=None):
    if extract_pixel_layer and 'pixels' not in list(layer_names):
        layer_names = list(layer_names)
        layer_names.append('pixels')

    activations_per_model = {
        f"{model_name + ('_' + layer_name if len(layer_name) > 0 else '')}": {} for model_name, _ in models for layer_name in layer_names
    }

    with torch.no_grad():
        for i, (item) in tqdm.tqdm(enumerate(loader), total=len(loader)):
            x = item[0]
            # y = item[1]
            if len(item) > 2:
                z = item[2]
                image_names = item[3]

            x = x.to(device=device)

            for model_name, model in models:
                model.eval()
                batch_activations = model(x)

                # If this is a feature extractor
                if type(batch_activations) is dict:
                    for layer_index, (layer_name, layer_batch_activations) in enumerate(batch_activations.items()):
                        for idx, image_name, activations, img in zip(z, image_names, layer_batch_activations, x):

                            if layer_index == 0 and extract_pixel_layer:
                                img = img.flatten().cpu().detach().numpy()
                                img_key = (f'{model_name}_pixels', int(idx.cpu().detach().numpy()))
                                activations_per_model[img_key[0]][img_key[1]] = img

                            else:
                                activations = activations.cpu().detach().numpy().squeeze()
                                if flatten:
                                    activations = activations.flatten()
                                # activations_per_model[f'{model_name}_{layer_name}'][int(idx.cpu().detach().numpy())] = activations # 
                                activations_per_model[f'{model_name}_{layer_name}'][image_name] = activations # 
                
                # If this is a normal model module
                else:
                    for idx, image_name, activations in zip(z, image_names, batch_activations):
                        # activations_per_model[model_name][int(idx.cpu().detach().numpy())] = activations[:, 0, 0].cpu().detach().numpy()
                        activations_per_model[model_name][image_name] = activations[:, 0, 0].cpu().detach().numpy()

    return activations_per_model