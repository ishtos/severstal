# TODO: change

import os
ope = os.path.exists
import numpy as np
import socket
import warnings
warnings.filterwarnings('ignore')

sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
hostname = socket.gethostname()
print('run on %s' % hostname)

RESULT_DIR = "../output/result"
DATA_DIR = "../input"
PRETRAINED_DIR = "../input/pretrained"
PI  = np.pi
INF = np.inf
EPS = 1e-12
ID = 'ImageId'
SPLIT = 'class_count'
TARGET = 'EncodedPixels'
IMG_SIZE = (256, 1600)
CROP_ID = 'CropImageId'
MASK_AREA = 'MaskArea'
DATASET = 'dataset'

STEEL = 'Steel'
NIH_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', STEEL 
]
