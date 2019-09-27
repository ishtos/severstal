import os
osp = os.path
ope = os.path.exists
opj = os.path.join

from utils.mask_functions import run_length_encode, run_length_decode

def make_split_label(x):
    if x['class_count'] != 1:
        return 0
    if str(x['1']) != 'nan':
        return 1
    if str(x['2']) != 'nan':
        return 2
    if str(x['3']) != 'nan':
        return 3
    if str(x['4']) != 'nan':
        return 4