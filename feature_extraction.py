import os

nproc = 5
os.environ["OMP_NUM_THREADS"] = str(nproc)
os.environ["OPENBLAS_NUM_THREADS"] = str(nproc)
os.environ["MKL_NUM_THREADS"] = str(nproc)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(nproc)
os.environ["NUMEXPR_NUM_THREADS"] = str(nproc)
from torchvision.models.feature_extraction import create_feature_extractor
# from oads_access.oads_access import OADS_Access, OADSImageDataset
from utils import OADSImageDataset, record_activations, collate_fn, CustomOADS
from torch.utils.data import DataLoader
# from pytorch_utils.pytorch_utils import collate_fn, record_activations


import pickle
import torch
from torchvision import transforms
from torchvision.models import alexnet, AlexNet_Weights


def extract_features(save_to_file:bool=True, subjects:list=list(range(5,36))):
    
    home_path = os.path.expanduser('~')
    basedir = f'{home_path}/projects/data/oads'
    eeg_dir = '/home/nmuller/projects/fmg_storage/osf_eeg_data'


    gpu_name = 'cuda:0'
    device = torch.device(gpu_name if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    torch.cuda.empty_cache()
    
    
    ############# ARGPARSE
    batch_size = 16 #512 # 512
    num_workers = nproc

    #######################


    oads = CustomOADS(basedir=f'{home_path}/projects/data/oads', n_processes=num_workers)


    width = 2155
    height = 1440

    size = (int(height), int(width))

    gcs = {}
    gcs_inverse = {}

    return_nodes = {
        'features.2': 'layer1',
        'features.5': 'layer2',
        'features.12': 'layer3',
        # 'classifier.5': 'feature',
    }

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

    model = model.to(device)
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

    transform_list = []
    transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(transform_list)

    #### TODO: Change so that once, all features are extracted into one big file, such that for each subject, the respective subselection can be loaded. 
    # Maybe an HDF5 file is the best solution for this.

    all_ids = []

    for sub in subjects:
        target_filenames = ['0434262454981c92.tiff', '0c6c2c66e61e3133.tiff', '0e2e2f2931313d37.tiff', '1332a2d4c4c14322.tiff', '202024d9b333968c.tiff', '24247432670b3131.tiff', '2c2064d163c52c78.tiff', '42431879e191d3c3.tiff', '61662ece1f6e7870.tiff', '7890918716766325.tiff', '85cc4e533959b1e1.tiff', '8ece4062eee68292.tiff', '93939b9a101e87e0.tiff', '948cacbc94b42474.tiff', 'af8f3a3939313171.tiff', 'b23232b232332361.tiff', 'b371f8ecf4e0f0f0.tiff', 'bce2d2d2c393e3b2.tiff', 'c77131717179f173.tiff', 'c93b3b2b29b9b8f1.tiff', 'cc8c9290bcbcf4fc.tiff', 'dad8acb8b82e36b1.tiff', 'e2f8f8fcf8f8f8f8.tiff', 'e6d8d40438c0c8e2.tiff', 'e6e648c62ebacc38.tiff', 'ef3632c2c1476e78.tiff', 'f4ca83dc6731310d.tiff']
        
        with open(os.path.join(eeg_dir, f'filenames_oads_eeg_rsvp_sub-{str(sub).zfill(2)}.pkl'), 'rb') as f:
            filenames = pickle.load(f)

        with open(os.path.join(eeg_dir, f'is_test_oads_eeg_rsvp_sub-{str(sub).zfill(2)}.pkl'), 'rb') as f:
            is_test = pickle.load(f)

        train_filenames = [filenames[i] for i in range(len(filenames)) if not is_test[i]]
        test_filenames = [filenames[i] for i in range(len(filenames)) if is_test[i]]

        train_ids = [x.split('.')[0] for x in train_filenames if f"{x.split('.')[0]}.tiff" not in target_filenames]
        test_ids = [x.split('.')[0] for x in test_filenames if f"{x.split('.')[0]}.tiff" not in target_filenames]

        # print(len(train_ids), len(test_ids))

        all_ids.extend(train_ids)
        all_ids.extend(test_ids)

    # all_ids = list(set(all_ids))[:50]

    print(len(all_ids))


    dataset = OADSImageDataset(oads_access=oads, item_ids=all_ids, use_crops=False, preload_all=False, target=None, return_index=True,
                                    class_index_mapping=None, transform=transform, device=device)
    # test_testdataset = OADSImageDataset(oads_access=oads, item_ids=test_ids, use_crops=False, preload_all=False, target=None, return_index=True,
    #                                 class_index_mapping=None, transform=transform, device=device)

    dataloader = DataLoader(dataset, collate_fn=collate_fn,
                            batch_size=batch_size, shuffle=False, num_workers=oads.n_processes)
    # test_testloader = DataLoader(test_testdataset, collate_fn=collate_fn,
    #                         batch_size=batch_size, shuffle=False, num_workers=oads.n_processes)


    activations = record_activations(loader=dataloader, models=[('alexnet', feature_extractor)], device=device, layer_names=return_nodes.values(), flatten=False)
    # test_activations = record_activations(loader=test_testloader, models=[(model_type, feature_extractor)], device=device, layer_names=return_nodes.values(), flatten=False)

    if save_to_file:
        ##### Save Activations
        os.makedirs('/home/nmuller/projects/fmg_storage/TEST_feature_extraction', exist_ok=True)
        with open('/home/nmuller/projects/fmg_storage/TEST_feature_extraction/activations.pkl', 'wb') as f:
            pickle.dump(activations, f)

    else:
        return activations

if __name__ == '__main__':
    main()