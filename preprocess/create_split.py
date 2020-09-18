import pandas as pd
import os
import sys
import argparse
import numpy as np
sys.path.append(os.getcwd())

from constants import *
from sklearn.model_selection import train_test_split


def main(args):
    # test col
    test_df = pd.read_csv(args.test_csv)
    test_df[SPLIT_COL] = "test"

    # train and s
    train_df = pd.read_csv(args.train_csv)
    msk = np.random.rand(len(train_df)) > args.val_ratio
    train_df.loc[msk, SPLIT_COL] = 'train'
    train_df.loc[~msk, SPLIT_COL] = 'valid'
    print(train_df[SPLIT_COL].value_counts())

    # save results
    test_df.to_csv(args.test_csv)
    train_df.to_csv(args.train_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', type=str, default=RSNA_TEST_CSV)
    parser.add_argument('--train_csv', type=str, default=RSNA_TRAIN_CSV)
    parser.add_argument('--val_ratio', type=float, default=0.3)
    args = parser.parse_args()

    main(args)