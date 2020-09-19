import os
import sys
sys.path.append(os.getcwd())

import h5py
import argparse
import pandas as pd
import utils
import cv2

from constants import *
from tqdm import tqdm


def main(args):
    # create hdf5 file
    hdf5_fh = h5py.File(args.hdf5_file, 'a')

    df = pd.read_csv(args.csv_file)
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        path = args.data_dir / row[STUDY_COL] / row[SERIES_COL] / (row[INSTANCE_COL] + ".dcm")
        image = utils.read_dicom(path=path)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        hdf5_fh.create_dataset(row[INSTANCE_COL], data=image, dtype='float32', chunks=True)
    
    # clean up
    hdf5_fh.close()


if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=RSNA_TRAIN_DATA_DIR)
    parser.add_argument('--csv_file', type=str, default=RSNA_TRAIN_CSV)
    parser.add_argument('--hdf5_file', type=str, default=RSNA_TRAIN_HDF5)
    args = parser.parse_args()

    main(args)
