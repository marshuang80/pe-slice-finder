import os
import sys
import torch
import argparse
sys.path.append(os.getcwd())

from constants import *
from utils     import str2bool, float_or_none


class BaseArgParser:
    '''Base training argument parser
    Shared with CheXpert and Contrastive loss Training
    '''

    def __init__(self):
        self.parser = argparse.ArgumentParser(description = "Base Arguments")
        
        # logger args
        self.parser.add_argument("--trial_name", type=str, default='trial_1.0')
        self.parser.add_argument("--save_dir", type=str, default=LOG_DIR)
        self.parser.add_argument("--experiment_name", type=str, default='debug')
        
        # training
        self.parser.add_argument('--lr', type=float, default=0.001)
        self.parser.add_argument("--batch_size", type=int, default=128)
        self.parser.add_argument("--num_workers", type=int, default=8)
        self.parser.add_argument('--optimizer', type=str, default='AdamW')
        self.parser.add_argument('--loss_fn', type=str, default='BCE')
        self.parser.add_argument('--class_weight', type=float_or_none, default=None)
        self.parser.add_argument("--weighted_sampling", type=str2bool, default=False)

        # augmentations 
        self.parser.add_argument("--resize_shape", type=int, default=None)
        self.parser.add_argument("--crop_shape", type=int, default=None)
        self.parser.add_argument("--horizontal_flip", type=str2bool, default=False)
        self.parser.add_argument("--vertical_flip", type=str2bool, default=False)
        self.parser.add_argument("--rotation_range", type=int, default=20)
        self.parser.add_argument("--normalize", type=str2bool, default=False)

    def get_parser(self):
        return self.parser