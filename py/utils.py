import os
import random
import numpy as np

import torch
import torch.nn as nn
import albumentations as A

from models import *

def seed_everything(seed):
    """
    Set seed
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# TODO: bug fix
def rle2mask(rle, shape):
    '''
    rle: run-length as string formated
    shape: (width, height)
    is_null: whether rle is null or not

    return: mask image
    '''
    width, height = shape
    
    mask = np.zeros(width * height).astype(np.uint8)
    
    if str(rle) != 'nan':
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]
        
        for start, length in zip(starts, lengths):
            mask[int(start):int(start+length)] = 1
    return np.flipud(np.rot90(mask.reshape(height, width), k=1))

def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background

    return: run-length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def run_length_decode(rle, height=256, width=1600, fill_value=1):
    mask = np.zeros((height,width), np.float32)
    if rle != '':
        mask=mask.reshape(-1)
        r = [int(r) for r in rle.split(' ')]
        r = np.array(r).reshape(-1, 2)
        for start,length in r:
            start = start-1 
            mask[start:(start + length)] = fill_value
        mask = mask.reshape(width, height).T
    return mask


def run_length_encode(mask):
    m = mask.T.flatten()
    if m.sum() == 0:
        rle=''
    else:
        m   = np.concatenate([[0], m, [0]])
        run = np.where(m[1:] != m[:-1])[0] + 1
        run[1::2] -= run[::2]
        rle = ' '.join(str(r) for r in run)
    return rle
    
def build_mask(series):
    mask = np.zeros((256, 1600, 4))
    for i in range(4):
        mask[:,:,i] = run_length_decode(series[f'{i+1}'], (256, 1600))
    return mask

def get_transforms():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms = A.Compose([A.Normalize(mean, std)])
    return transforms

def get_model(network, n_classes):
    if network == 'Res34Unetv3':
        model = Res34Unetv3(n_classes)
        return model 
    elif network == 'Res34Unetv4':
        model = Res34Unetv4(n_classes)
        return model 
    elif network == 'Res34Unetv5':
        model = Res34Unetv5(n_classes)
        return model 
    else:
        raise ValueError(f'Unknown network {network}')

def get_loss(loss):
    if loss == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f'Unknown loss {loss}')

def do_kaggle_metric(y_true, y_pred, smooth=1.0):
    y_true_f = np.ravel(y_true)
    y_pred_f = np.ravel(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)