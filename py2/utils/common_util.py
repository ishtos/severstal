import os
osp = os.path
ope = os.path.exists
opj = os.path.join
import numpy as np

from utils.mask_functions import run_length_encode, run_length_decode

def make_split_label(x):
    if x['class_count'] == 0:
        return 0
    if x['class_count'] <= 2:
        return 5
    if str(x['1']) != 'nan':
        return 1
    if str(x['2']) != 'nan':
        return 2
    if str(x['3']) != 'nan':
        return 3
    if str(x['4']) != 'nan':
        return 4


def make_label(x):
    if x == 0:
        return np.array([0, 0, 0, 0, 0])
    elif x == 1:
        return np.array([1, 0, 0, 0, 0])
    elif x == 2:
        return np.array([0, 1, 0, 0, 0])
    elif x == 3:
        return np.array([0, 0, 1, 0, 0])
    elif x == 4:
        return np.array([0, 0, 0, 1, 0])
    elif x == 5:
        return np.array([0, 0, 0, 0, 1])