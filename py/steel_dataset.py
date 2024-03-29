import os
import cv2
import numpy as np
import pandas as pd

from copy import deepcopy
from tqdm import tqdm

from torch.utils.data import Dataset

from utils import build_mask
from steel_transforms import *


class SteelDataset(Dataset):
    def __init__(self, df, mode='train', is_tta=False, fine_size=256, pad_left=0, pad_right=0, transforms=None):
        self.df = df
        self.mode = mode
        self.is_tta = is_tta
        self.fine_size = fine_size
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join('..', 'input', 'train_images', self.df.iloc[idx]['ImageId']))
        img = img.astype(np.float32) / 255.

        if self.mode == 'train':
            mask = build_mask(self.df.iloc[idx])
            mask = mask.astype(np.float32)
            if self.transforms is not None:
                transformed = self.transforms(image=img)
                img = transformed['image']
            img = img.transpose(2, 0, 1)
            mask = mask.transpose(2, 0, 1)
            return img, mask
        
        elif self.mode == 'valid':
            mask = build_mask(self.df.iloc[idx])
            mask = mask.astype(np.float32)
            if self.transforms is not None:
                transformed = self.transforms(image=img)
                img = transformed['image']
            img = img.transpose(2, 0, 1)
            mask = mask.transpose(2, 0, 1)
            return img, mask

        elif self.mode == 'test':
            return img


def trainImageFetch(image_ids):
    img_train = []
    mask_train = []

    for idx, img_id in tqdm(enumerate(image_ids)):
        img_path = os.path.join('..', 'input', 'train_images', f'{img_id}')
    
    return img_train
