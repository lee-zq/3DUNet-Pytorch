from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import random
from torchvision.transforms import RandomCrop
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from .transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize
from .tools import load_file_name_list, padding_img, extract_ordered_overlap

class Dataset(dataset):
    def __init__(self, args, mode='train'):

        self.args = args
        self.mode = mode
        
        if self.mode=='train':
            self.filename_list = load_file_name_list(os.path.join(args.dataset_path, 'train_path_list.txt'))
        elif self.mode =='val':
            self.filename_list = load_file_name_list(os.path.join(args.dataset_path, 'val_path_list.txt'))
        else:
            raise TypeError('Dataset mode error!!! ')

        self.train_trans = Compose([
                RandomCrop(self.args.crop_slices),
                # RandomFlip_LR(prob=0.5),
                # RandomFlip_UD(prob=0.5),
                # RandomRotate()
            ])
        self.val_trans = Compose([Center_Crop(16),
                                # Resize(scale=0.5)
                                ]) 
    def __getitem__(self, index):

        ct = sitk.ReadImage(self.filename_list[index][0], sitk.sitkInt16)
        seg = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        ct_array = ct_array / self.args.norm_factor
        ct_array = ct_array.astype(np.float32)

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)

        if self.mode=='train':
            ct_array,seg_array = self.train_trans(ct_array, seg_array)     
        elif self.mode == 'val':
            ct_array, seg_array = self.val_trans(ct_array, seg_array)
        else:
            raise ValueError("Dataset Mode Error")
        # print(ct_array.shape, seg_array.shape)
        return ct_array, seg_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)

if __name__ == "__main__":
    sys.path.append('/ssd/lzq/3DUNet')
    from config import args
    train_ds = Dataset(args, mode='train')

    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=1)

    for i, (ct, seg) in enumerate(train_dl):
        print(i,ct.size(),seg.size())