import pandas as pd

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

from utils.mask_functions import *
from utils.augment_util import *
from utils.common_util import *
from datasets.tool import *
from config.config import *
import numpy as np

class SteelDataset(Dataset):
    def __init__(self, 
                 steel_df,
                 img_size=(256, 1600),
                 mask_size=(256, 1600),
                 transform=None,
                 return_label=False,
                 dataset=None):
        self.steel_df = steel_df
        self.img_size = img_size
        self.mask_size = mask_size
        self.return_label = return_label
        self.dataset = dataset
        self.label = steel_df['label']
       
        base_dir = DATA_DIR
        if dataset in ['train', 'val']:
            img_dir = opj(base_dir, 'train', 'images', 'images_256x1600')
        elif dataset in ['test']:
            img_dir = opj(base_dir, 'test', 'images', 'images_256x1600')
        else:
            raise ValueError(dataset)

        self.pos_flag = steel_df[SPLIT] != 0
        self.img_ids = self.steel_df[ID].values
        self.img_dir = img_dir
        self.num = len(self.img_ids)
        self.basic_img_ids = self.img_ids
        self.transform = transform

        print('image_dir: %s' % self.img_dir)
        print('image size: %s' % str(self.img_size))

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_fname = opj(self.img_dir, f'{img_id}')
        label = self.label[index]

        image = cv2.imread(img_fname)
        if image is None:
            print(img_fname)
            raise ValueError(img_fname)
        
        print(image)
        if self.return_label:
            if self.transform is not None:
                image = self.transform(image=image)

            image = image / 255.0
            image = image_to_tensor(image)
            return image, label, index
        else:
            if self.transform is not None:
                image = self.transform(image=image)[0]
            image = image / 255.0
            image = image_to_tensor(image)
            return image, index

    def __len__(self):
        return self.num


class BalanceClassSampler(Sampler):
    def __init__(self, dataset, length=None):
        self.dataset = dataset
        if length is None:
            length = len(self.dataset)
        self.length = int(length)

        half = self.length // 2 + 1
        self.pos_length = half
        self.neg_length = half
        print('pos num: %s, neg num: %s' % (self.pos_length, self.neg_length))

    def __iter__(self):
        pos_index = np.where(self.dataset.pos_flag)[0]
        neg_index = np.where(~self.dataset.pos_flag)[0]

        pos = np.random.choice(pos_index, self.pos_length, replace=True)
        neg = np.random.choice(neg_index, self.neg_length, replace=True)

        l = np.hstack([pos, neg]).T
        l = l.reshape(-1)
        np.random.shuffle(l)
        l = l[:self.length]
        return iter(l)

    def __len__(self):
        return self.length


class FourBalanceClassSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

        label = (self.dataset.steel_df['split_label'].values)
        label = label.reshape(-1, 5)
        label = np.hstack([label.sum(1, keepdims=True) == 0, label]).T

        self.neg_index = np.where(label[0])[0]
        self.pos1_index = np.where(label[1])[1]
        self.pos2_index = np.where(label[2])[0]
        self.pos3_index = np.where(label[3])[0]
        self.pos4_index = np.where(label[4])[0]
        self.pos5_index = np.where(label[5])[0]

        num_neg = len(self.neg_index)
        self.length = 4*num_neg + (num_neg // 2)

    def __iter__(self):
        neg = self.neg_index().copy()
        np.random.shuffle(neg)
        num_neg = len(self.neg_index)

        poa1 = np.random.choice(self.pos1_index, num_neg, replace=True)
        pos2 = np.random.choice(self.pos2_index, num_neg, replace=True)
        pos3 = np.random.choice(self.pos3_index, num_neg, replace=True)
        pos4 = np.random.choice(self.pos4_index, num_neg, replace=True)
        pos5 = np.random.choice(self.pos5_index, num_neg // 2, replace=True)

        l = np.stack([neg, pos1, pos2, pos3, pos4]).T
        l = l.reshape(-1)
        return iter(l)

    def __len__(self):
        return self.length

