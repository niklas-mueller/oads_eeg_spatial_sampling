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
    return torch.utils.data.dataloader.default_collate(batch) # type: ignore

class OADSImageDataset(Dataset):
    def __init__(self, oads_access, item_ids: list, transform=None, target_transform=None, device='cuda:0', return_index:bool=False) -> None:
        super().__init__()

        self.oads_access = oads_access
        self.item_ids = item_ids
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self.return_index = return_index
        self.tupels = {}

            

    def iterate(self, idx):
        image_name = self.item_ids[idx]
        tup = self.oads_access.load_image(image_name=image_name)

        if tup is None:
            return None, None
        
        return idx, tup

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
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

        label = np.array([]) # label['classId']

        if self.target_transform:
            label = self.target_transform(label)
        if self.return_index:
            return (img, label, idx, image_name)
        
        return (img, label)

def record_activations(loader, models:tuple, device, layer_names:list=[''], flatten:bool=True):
    activations_per_model = {
        f"{model_name + ('_' + layer_name if len(layer_name) > 0 else '')}": {} for model_name, _ in models for layer_name in layer_names
    }

    with torch.no_grad():
        for i, (item) in tqdm.tqdm(enumerate(loader), total=len(loader)):
            x = item[0]
            # y = item[1]
            if len(item) <= 2:
                raise ValueError("Expected at least 3 items in the batch (img, index, image_name), got less.")
            
            z = item[2]
            image_names = item[3]

            x = x.to(device=device)

            for model_name, model in models:
                model.eval()
                batch_activations = model(x)

                # If this is a feature extractor
                if type(batch_activations) is dict:
                    for layer_index, (layer_name, layer_batch_activations) in enumerate(batch_activations.items()):
                        for idx, image_name, activations in zip(z, image_names, layer_batch_activations):

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