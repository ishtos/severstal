
import os
import pickle
import cv2

import numpy as np
import pandas as pd

from tqdm import tqdm
from utils import rle2mask


# FIXME: cannot save because of file size
def main():
    train = pd.read_csv(os.path.join('..', 'input', 'preprocessed_train.csv'))

    mask_list = []
    for (_, row) in tqdm(train.iterrows()):
        mask = np.zeros((256, 1600, 4))
        for i in range(0, 4):
            mask[:,:, i] = rle2mask(row[f'{i+1}'], (256, 1600))
        with open(os.path.join('..', 'input', 'train_masks', row['ImageId'].replace('jpg', 'pkl')), 'wb') as f:
            pickle.dump(mask, f)
    
if __name__ == '__main__':
    main()