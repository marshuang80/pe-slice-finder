import pydicom 
import torch
import numpy as np
import argparse
import sys
import os
import torch
import time
from torchvision.transforms.transforms import RandomCrop
sys.path.append(os.getcwd())

from constants        import *
from typing           import Union, Iterable
from pathlib          import Path
from torchvision      import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler 


def convert_to_hu(raw_pixel:np.ndarray, intercept:int, slope:int)-> np.ndarray:
    """Convert raw pixel to Hounsfield Units

    Args: 
        raw_pixel (np.ndarray): raw pixel from DICOM
        intercept (int): rescale intercept
        slope (int): rescale slope

    Returns:
        rescaled ct scan as numpy arrray        
    """

    image = raw_pixel.astype(np.float64) * slope
    image = image.astype(np.int16)
    image += np.int16(intercept)
    
    return image


def normalize_and_window_hu(image:np.ndarray)-> np.ndarray:
    """Normalize and window HU after windowing and clipping

    HU_MIN = Level - Window / 2 
    HU_MAX = Level = Window / 2
    Optimal level for pulmonary embolism = 400
    Optimal window for pulmonary embolism = 1000

    Args:
        image (np.ndarray): image to be nomalized
    Returns: 
        normalized image
    """
    image = (image - HU_MIN) / (HU_MAX - HU_MIN)
    image[image > 1] = 1.
    image[image < 0] = 0.

    # TODO: figure out HU_MEAN
    #image = image - HU_MEAN
    return image


def read_dicom(path:Union[str,Path])->np.ndarray:
    """Read dicom file from path

    Args:
        path (str, Path): path to dicom file

    Returns: 
        rescaled ct scan as numpy array 
    """

    dcm = pydicom.read_file(path)

    # conver to hu
    raw_pixel = dcm.pixel_array
    intercept = dcm.RescaleIntercept
    slope = dcm.RescaleSlope
    image = convert_to_hu(raw_pixel, intercept, slope)

    # set area outside scanner to air
    #image[image <= -2000] = AIR_HU

    # normalize and window to optimal PE range
    image = normalize_and_window_hu(image) 

    return image


def weighted_sampler(dataset:DataLoader)-> WeightedRandomSampler:
    """Create a weighted sampler for pytorch dataloader to load positive and 
    negative samples with equal prevelance. 

    Args:
        dataset (Dataloader): pytorch dataloader
    
    Returns:
        pytorch weighted sampler
    """

    neg_class_count = (dataset.df[TARGET_COL] == 0).sum().item()
    pos_class_count = (dataset.df[TARGET_COL] == 1).sum().item()
    class_weight = [1/neg_class_count, 1/pos_class_count]
    weights = [class_weight[i] for i in dataset.df[TARGET_COL]]

    weights = torch.Tensor(weights).double()
    sampler = WeightedRandomSampler(
        weights, 
        num_samples=len(weights),
        replacement=True
    )
    return sampler


def get_transformations(args)-> transforms.Compose:
    """Create transformations for input images based on arguments"""

    transform = []
    if args.rotation_range != 0:
        transform.append(transforms.RandomRotation(args.rotation_range))
    if args.crop_shape is not None:
        transform.append(transforms.RandomCrop(args.crop_shape))
    if args.horizontal_flip:
        transform.append(transforms.RandomHorizontalFlip(p=0.5))
    if args.vertical_flip:
        transform.append(transforms.RandomVerticalFlip(p=0.5))
    transform.append(transforms.ToTensor())

    transform = transforms.Compose(transform)

    return transform

def str2bool(v):
    """convert input argument to bool"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def float_or_none(v):
    """convert input argument to bool"""
    if v is None:
        return v
    if v.lower() == 'none':
        return None
    else:
        return float(v)
