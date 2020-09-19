import pandas as pd
import os
import sys
import argparse
import numpy as np
sys.path.append(os.getcwd())

from constants import *
from sklearn.model_selection import train_test_split


def main(args):

    # get positive series only    
    df = pd.read_csv(args.train_csv)
    df[SPLIT_COL] = 'na'
    label_df = df[df[NEGATIVE_PE_SERIES_COL] == 0]
    instances = label_df[INSTANCE_COL].unique()

    # split between train and val+test
    eval_size = args.eval_ratio * 2
    train_instances, test_val_instances = train_test_split(
        instances, test_size=eval_size, random_state=42
    )

    # split between val and test
    test_size = 0.5
    val_instances, test_instances = train_test_split(
        test_val_instances, test_size=test_size, random_state=42
    )
    
    # assign splits
    train_rows = label_df[INSTANCE_COL].isin(train_instances)
    label_df.loc[train_rows, SPLIT_COL] = "train"
    val_rows = label_df[INSTANCE_COL].isin(val_instances)
    label_df.loc[val_rows, SPLIT_COL] = "valid"
    test_rows = label_df[INSTANCE_COL].isin(test_instances)
    label_df.loc[test_rows, SPLIT_COL] = "test"

    # assign split back to original df 
    df[df[NEGATIVE_PE_SERIES_COL] == 0] = label_df

    # save results
    df.to_csv(args.train_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default=RSNA_TRAIN_CSV)
    parser.add_argument('--eval_ratio', type=float, default=0.1)
    args = parser.parse_args()

    main(args)