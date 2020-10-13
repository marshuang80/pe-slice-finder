import os
import sys
sys.path.append(os.getcwd())

import h5py
import pickle
import argparse
import pandas as pd

from constants import *
from tqdm import tqdm
from collections import defaultdict


def main(args):
    # create hdf5 file
    hdf5_fh = h5py.File(args.hdf5_file, 'a')
    slice_labels = pickle.load(open(args.pickle_file, 'rb')) 
    results = defaultdict(list)

    for series in hdf5_fh.keys():
        # skip if no labelss
        if series not in slice_labels.keys():
            continue
        for slice_idx in range(hdf5_fh[series].shape[0]):
            label = 1 if slice_idx in slice_labels[series] else 0
            results['series'].append(series)
            results['slice_idx'].append(slice_idx)
            results['label'].append(label)

    # save as csv 
    df = pd.DataFrame.from_dict(results)
    df.to_csv('slice_labels.csv')
    
    # clean up
    hdf5_fh.close()


if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument('--hdf5_file', type=str, default='/data4/PE_stanford/Stanford_data/data.hdf5')
    parser.add_argument('--pickle_file', type=str, default='/data4/PE_stanford/Stanford_data/slice_labels.pkl')
    args = parser.parse_args()

    main(args)
