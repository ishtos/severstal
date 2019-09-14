import os
import random
import numpy as np

import torch

from models.models_zoo import *

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

def rle2mask(rle, shape, is_null):
    '''
    rle: run-length as string formated
    shape: (width, height)
    is_null: whether rle is null or not

    return: mask image
    '''
    width, height = shape
    
    mask = np.zeros(width * height).astype(np.uint8)
    
    if not is_null:
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]
        
        for start, length in zip(starts, lengths):
            mask[int(start):int(start+length)] = 255
        
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
        raise ValueError('Unknown network ' + network)

    return model

def do_kaggle_metric(predict,truth, threshold=0.5):

    N = len(predict)
    predict = predict.reshape(N, -1)
    truth   = truth.reshape(N, -1)

    predict = predict > threshold
    truth   = truth > 0.5
    intersection = truth & predict
    union        = truth | predict
    iou = intersection.sum(1) / (union.sum(1) + 1e-8)

    #-------------------------------------------
    result = []
    precision = []
    is_empty_truth   = (truth.sum(1) == 0)
    is_empty_predict = (predict.sum(1) == 0)

    threshold = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    for t in threshold:
        p = iou >= t

        tp  = (~is_empty_truth)  & (~is_empty_predict) & (iou > t)
        fp  = (~is_empty_truth)  & (~is_empty_predict) & (iou <= t)
        fn  = (~is_empty_truth)  & ( is_empty_predict)
        fp_empty = ( is_empty_truth)  & (~is_empty_predict)
        tn_empty = ( is_empty_truth)  & ( is_empty_predict)

        p = (tp + tn_empty) / (tp + tn_empty + fp + fp_empty + fn)

        result.append( np.column_stack((tp,fp,fn,tn_empty,fp_empty)) )
        precision.append(p)

    result = np.array(result).transpose(1, 2, 0)
    precision = np.column_stack(precision)
    precision = precision.mean(1)

    return precision, result, threshold