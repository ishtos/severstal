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
                 return_label=True,
                 pseudo=None,
                 pseudo_ratio=0.,
                 crop_version=None,
                 dataset=None,
                 predict_pos=False):
        self.img_size = img_size
        self.mask_size = mask_size
        self.return_label = return_label
        self.crop_version = crop_version
        self.dataset = dataset

        # TODO: modify
        if self.dataset == 'chexpert':
            self.suffix = 'jpg'
        else:
            self.suffix = 'jpg'

        base_dir = DATA_DIR
        if dataset in ['train', 'val']:
            if crop_version is None:
                img_dir = opj(base_dir, 'train', 'images', 'images_256x1600')
        elif dataset in ['test']:
            if crop_version is None:
                img_dir = opj(base_dir, 'test', 'images', 'images_256x1600')
        else:
            raise ValueError(dataset)

        if predict_pos:
            steel_df = steel_df[steel_df[STEEL] == 1]
        steel_df = steel_df.fillna(0)
        steel_df['pseudo'] = False
        self.steel_df = steel_df

        if crop_version is None:
            self.img_ids = self.steel_df[ID].values
        else:
            self.img_ids = self.steel_df[CROP_ID].values

        self.img_dir = img_dir
        self.num = len(self.img_ids)

        self.basic_img_ids = self.img_ids
        self.transform = transform

        # self.pseudo_flag = self.steel_df['pseudo'].values
        # self.basic_pseudo_flag = self.pseudo_flag
        # self.pseudo = pseudo
        # self.pseudo_ratio = pseudo_ratio

        if (dataset == 'train') and return_label:
            if 'pos' in self.steel_df.columns:
                self.pos_flag = self.steel_df['pos']
                self.pos_steel_df = self.steel_df[self.pos_flag]
                self.neg_steel_df = self.steel_df[~self.pos_flag]
            else:
                self.pos_flag = self.steel_df[TARGET] != 0
                self.pos_steel_df = self.steel_df[self.pos_flag]
                self.neg_steel_df = self.steel_df[~self.pos_flag]
        else:
            self.pos_steel_df = None
            self.neg_steel_df = None

        # if pseudo is not None:
        #     self.pseudo_list = pseudo.split(',')
        #     self.pos_pseudo_df_list = []
        #     self.neg_pseudo_df_list = []
        #     for pseudo in self.pseudo_list:
        #         pseudo_df = pd.read_csv(opj(DATA_DIR, 'pseudo', '%s.csv' % pseudo))
        #         pseudo_df['pseudo'] = True
        #         if CROP_ID not in pseudo_df.columns:
        #             pseudo_df[CROP_ID] = pseudo_df[ID]
        #         if 'chexpert' in pseudo:
        #             pseudo_df['suffix'] = 'jpg'
        #         else:
        #             pseudo_df['suffix'] = 'png'
        #         pseudo_df['pseudo_name'] = pseudo
        #         pos_pseudo_df = pseudo_df[pseudo_df[TARGET] != '-1']
        #         neg_pseudo_df = pseudo_df[pseudo_df[TARGET] == '-1']

        #         self.pos_pseudo_df_list.append(pos_pseudo_df)
        #         self.neg_pseudo_df_list.append(neg_pseudo_df)
        #     self.resample_pseudo(first=True)

        print('image_dir: %s' % self.img_dir)
        print('image size: %s' % str(self.img_size))

    def resample_pseudo(self, first=False):
        steel_df_list = [self.pos_steel_df, self.neg_steel_df]
        if first:
            print('%s pos_num: %d neg_num: %d' % ('train', len(steel_df_list[-2]), len(steel_df_list[-1])))

        for idx, (pos_pseudo_df, neg_pseudo_df) in enumerate(zip(self.pos_pseudo_df_list, self.neg_pseudo_df_list)):
            pos_pseudo_size = int(min(len(self.pos_steel_df) * self.pseudo_ratio, len(pos_pseudo_df)))
            neg_pseudo_size = int(min(len(self.neg_steel_df) / len(self.pos_steel_df) * pos_pseudo_size, len(neg_pseudo_df)))

            pos_pseudo_size = min(len(pos_pseudo_df), pos_pseudo_size)
            neg_pseudo_size = min(len(neg_pseudo_df), neg_pseudo_size)
            steel_df_list.append(pos_pseudo_df.iloc[np.random.choice(len(pos_pseudo_df), size=pos_pseudo_size, replace=False)])
            steel_df_list.append(neg_pseudo_df.iloc[np.random.choice(len(neg_pseudo_df), size=neg_pseudo_size, replace=False)])

            if first:
                if self.crop_version is None:
                    print('pseudo dir: %s' % opj(DATA_DIR, pos_pseudo_df.iloc[0][DATASET],
                                                 'images', 'images_%d' % self.img_size))
                else:
                    print('pseudo dir: %s' % opj(DATA_DIR, pos_pseudo_df.iloc[0][DATASET],
                                                 'images', 'images_%s_%d' % (self.crop_version, self.img_size)))
                print('%s pos_num: %d neg_num: %d' % (self.pseudo_list[idx], len(steel_df_list[-2]), len(steel_df_list[-1])))

        steel_df = pd.concat(steel_df_list)
        if self.crop_version is None:
            self.img_ids = steel_df[ID].values
        else:
            self.img_ids = steel_df[CROP_ID].values
        self.pseudo_flag = steel_df['pseudo'].values
        self.dataset = steel_df[DATASET].values
        self.pos_flag = steel_df[TARGET] != 0
        self.pseudo_suffix = steel_df['suffix'].values
        self.pseudo_name = steel_df['pseudo_name'].values
        self.num = len(self.img_ids)

    def resample(self):
        pass

    def __getitem__(self, index):
        # is_pseudo = self.pseudo_flag[index]
        img_id = self.img_ids[index]

        # if is_pseudo:
        #     dataset = self.dataset[index]
        #     pseudo_suffix = self.pseudo_suffix[index]
        #     if self.crop_version is None:
        #         img_fname = opj(DATA_DIR, dataset, 'images', 'images_1024',
        #                         '%s.%s' % (img_id, pseudo_suffix))
        #     else:
        #         img_fname = opj(DATA_DIR, dataset, 'images', 'images_%s' % self.crop_version,
        #                         '%s.%s' % (img_id, pseudo_suffix))
        # else:
        #     img_fname = opj(self.img_dir, '%s.%s' % (img_id, self.suffix))
        img_fname = opj(self.img_dir, '%s.%s' % (img_id, self.suffix))

        image = cv2.imread(img_fname)
        if image is None:
            print(img_fname)
        if image.shape[0] != self.img_size[0] or image.shape[1] != self.img_size[1]:
            image = cv2.resize(image, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_LINEAR)

        if self.return_label:
            # if is_pseudo:
            #     dataset = self.dataset[index]
            #     pseudo_name = self.pseudo_name[index]
            #     if self.crop_version is None:
            #         mask_file = opj(DATA_DIR, dataset, 'images', 'masks_1024', '%s.png' % img_id)
            #     else:
            #         if pseudo_name is not None and '8740_pseudo' in pseudo_name:
            #             mask_file = opj(DATA_DIR, dataset, 'images', 'masks_8740_%s' % self.crop_version,
            #                             '%s.png' % img_id)
            #         else:
            #             mask_file = opj(DATA_DIR, dataset, 'images', 'masks_%s' % self.crop_version,
            #                             '%s.png' % img_id)
            # else:
            #     mask_file = opj(self.mask_dir, '%s.png' % img_id)
            
            # TODO: modify
            mask = build_mask(self.steel_df.iloc[index], self.mask_size[0], self.mask_size[1])
            mask = mask.astype(np.int8)

            if self.transform is not None:
                image = self.transform(image=img)
                image = image['image']

            image = image / 255.0
            # mask = mask / 255.0
            image = image_to_tensor(image)
            mask = label_to_tensor(mask)

            return image, mask, index
        else:
            if self.transform is not None:
                image = self.transform(image)
                image = image['image']
            image = image / 255.0
            image = image_to_tensor(image)
            return image, index

    def __len__(self):
        return self.num


# https://www.kaggle.com/c/siim-acr-STEEL-segmentation/discussion/97456
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