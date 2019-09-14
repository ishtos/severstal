import os
import cv2
import numpy as np
import pandas as pd

from copy import deepcopy
from tqdm import tqdm

import torch.utils.data import Dataset

from steel_transforms import *

train_root = os.path.join('..', 'input', 'train_images')
test_root = os.path.join('..', 'input', 'test_images')

train_id = pd.read_csv(os.path.join('..', 'input', 'train.csv'))['ImageId'].values
test_id = 