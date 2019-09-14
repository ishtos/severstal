import os
import random
import numpy as np

import torch

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

def rle2mask(rle, img, is_null):
    """
    Create mask images
    """
    width, height, _ = img.shape
    
    mask = np.zeros(width * height).astype(np.uint8)
    
    if not is_null:
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]
        
        for start, length in zip(starts, lengths):
            mask[int(start):int(start+length)] = 255
        
    return np.flipud(np.rot90(mask.reshape(height, width), k=1))

def mask2rle(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)