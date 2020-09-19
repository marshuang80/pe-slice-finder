
from pathlib import Path
from torchvision import models as models_2d


# Directories
HOME_DIR = Path.home()
RSNA_DATA_DIR = Path("/data4/rsna")
if not RSNA_DATA_DIR.is_dir():
    raise Exception("Please modify PROJECT_DATA_DIR in constants to a valid directory")

# Project cvs files 
RSNA_TRAIN_CSV = RSNA_DATA_DIR / "train.csv"
RSNA_TEST_CSV =  RSNA_DATA_DIR / "test.csv"

# Project image folders
RSNA_TRAIN_DATA_DIR = RSNA_DATA_DIR / "train"
RSNA_TEST_DATA_DIR = RSNA_DATA_DIR / "test"

# HDF5
RSNA_TRAIN_HDF5 = RSNA_DATA_DIR / "train.hdf5"
RSNA_TEST_HDF5 =  RSNA_DATA_DIR / "test.hdf5"

# Log dir
LOG_DIR = RSNA_DATA_DIR / "log"

# Dataframe Columns
SPLIT_COL = "Split"
STUDY_COL = 'StudyInstanceUID'
SERIES_COL = 'SeriesInstanceUID'
INSTANCE_COL = 'SOPInstanceUID'
TARGET_COL = 'pe_present_on_image'
NEGATIVE_PE_SERIES_COL = 'negative_exam_for_pe'

# HOUNSFIELD UNIT
AIR_HU = -1000
WATER_HU = 0
HU_MAX = 900 
HU_MIN = -100

# 2D CNN models
MODELS_2D = {
    'densenet121': [models_2d.densenet121, 1024],
    'densenet161': [models_2d.densenet161, 2208],
    'densenet169': [models_2d.densenet169, 1664],
    'densenet201': [models_2d.densenet201, 1920],
    'resnet18': [models_2d.resnet18, 512],
    'resnet34': [models_2d.resnet34, 512],
    'resnet50': [models_2d.resnet50, 2048],
    'resnext50': [models_2d.resnext50_32x4d, 2048],
    'resnext100': [models_2d.resnext101_32x8d, 2048]
}
