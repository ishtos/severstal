
import os
import cv2

import numpy as np
import pandas as pd

from tqdm import tqdm
from utils.functions import rle2mask

def main():
    train = pd.read_csv(os.path.join('..', 'input', 'train.csv'))

    for _, row in tqdm(train.iterrows()):
        fn = row['ImageId']
        img = cv2.imread(os.path.join('..', 'input', 'train_images', f'{fn}'))
        mask = rle2mask(row['EncodedPixels'], img, row['is_null'])
        cv2.imwrite(os.path.join('..', 'input', 'train_masks', f'f{fn}'), mask)

        break

if __name__ == '__main__':
    main()