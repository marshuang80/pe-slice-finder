from matplotlib import transforms
import numpy as np
import pandas as pd
import torch
import os 
import sys
import utils
import time
import h5py

sys.path.append(os.getcwd())

from torch.utils.data import DataLoader, Dataset
from constants        import *
from typing           import Optional, Tuple
from torchvision      import transforms
from PIL              import Image


class RSNADataset(Dataset):
    """Dataset class for RSNA PE CT Dataset
    
    https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection
    """

    def __init__(
        self, 
        split: str = 'train',
        transforms: transforms.Compose = None, 
        normalize: bool = False,
    )-> None:
        
        self.hdf5_path = RSNA_TRAIN_HDF5
        self.csv_path = RSNA_TRAIN_CSV
            
        # read in pandas dataframe
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df[self.df[NEGATIVE_PE_SERIES_COL] == 0]
        self.df = self.df[self.df[SPLIT_COL] == split]
        print(self.df[TARGET_COL].value_counts())
       
        # class variables
        self.num_channels = 3 
        self.transforms = transforms
        self.normalize = normalize

    def __len__(self)-> int:
        """Get length of dataset"""
        return len(self.df)

    def __getitem__(self, idx:int)-> Tuple[torch.Tensor]:
        """Get the idx-th item in dataset. 
        
        Args: 
            idx (int): the index
        Returns: 
            Tuple (x, y)
        """
        
        # get image
        instance = self.df.iloc[idx]
        with h5py.File(self.hdf5_path, 'r') as hdf5_fh:
            image = hdf5_fh[instance[INSTANCE_COL]][:]

        # transform image
        if self.transforms is not None:
            image = Image.fromarray(np.uint8(image * 255), 'L')
            tensor = self.transforms(image)
        else:
            tensor = torch.tensor(image)
        tensor = tensor.expand(3, -1, -1)

        if self.normalize:
            tensor = transforms.functional.nomalize(
                tensor, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            )

        # get labels
        y = instance[TARGET_COL].astype(float)

        return tensor, y


def get_dataloader(
        split: str = 'train', 
        dataset_args: dict = {}, 
        dataloader_args: dict = {}, 
        weighted_sampling: Optional[bool] = False)-> torch.utils.data.DataLoader:
    """Loads dataloaders for the PE CT dataset.

    Args:
        split (str, optional): the split with which to filter the CSV.
        dataset_args (dict, optional): keyword arguments for dataset init.
        dataloader_args (dict): keyword arguments for dataloader init.
        weighted_sampling (bool, optional): 
    Returns:
        a pytorch dataloader
    """ 
    
    dataset = RSNADataset(split=split, **dataset_args)

     # weighted sampling
    if weighted_sampling and split == 'train': 
        sampler = utils.weighted_sampler(dataset)
        dataloader_args['shuffle'] = False
        dataloader = DataLoader(dataset, sampler=sampler, **dataloader_args)
    else:
        dataloader = DataLoader(dataset, **dataloader_args)

    return dataloader


if __name__ == "__main__":
    # for debugging purpose
    dataset_args = {
    }
    dataloader_args = {
        "batch_size": 1, 
        "num_workers": 8
    }
    training_loader = get_dataloader(
        split="train",
        dataset_args=dataset_args,
        dataloader_args=dataloader_args, 
    )

    for x, y, series_id in training_loader:
        print(x.shape)
        print(x.max())
        print(x.min())
        print(y)
        break