
from utils.common import *
from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch,os
from torch.utils.data import Dataset, DataLoader


class Lits_DataSet(Dataset):
    def __init__(self, crop_size, batch_size,resize_scale, dataset_path,mode=None):
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.resize_scale=resize_scale
        self.dataset_path = dataset_path
        self.n_labels = 3

        if mode=='train':
            self.filename_list = load_file_name_list(os.path.join(dataset_path,'train_name_list.txt'))
        elif mode =='val':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'val_name_list.txt'))
        elif mode == 'test':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'test_name_list.txt'))
        else:
            raise TypeError('Dataset mode error!!! ')


    def __getitem__(self, index):
        data, target = self.get_train_batch_by_index(train_batch_size=self.batch_size,
                                                     crop_size=self.crop_size, index=index,
                                                     resize_scale=self.resize_scale)

        data = data.transpose(0, 4, 1, 2, 3)
        target = target.transpose(0, 4, 1, 2, 3)
        return torch.from_numpy(data), torch.from_numpy(target)

    def __len__(self):
        return len(self.filename_list)

    def get_train_batch_by_index(self, train_batch_size, crop_size, index,resize_scale=1):
        train_imgs = np.zeros([train_batch_size, crop_size[0], crop_size[1], crop_size[2], 1])
        train_labels = np.zeros([train_batch_size, crop_size[0], crop_size[1], crop_size[2], self.n_labels])
        img, label = self.get_np_data_3d(self.filename_list[index],resize_scale=resize_scale)
        for i in range(train_batch_size):
            sub_img, sub_label = random_crop_3d(img, label, crop_size)

            sub_img = sub_img[:, :, :, np.newaxis]
            sub_label_onehot = make_one_hot_3d(sub_label, self.n_labels)

            train_imgs[i] = sub_img
            train_labels[i] = sub_label_onehot

        return train_imgs, train_labels

    def get_np_data_3d(self, filename, resize_scale=1):
        data_np = sitk_read_row(self.dataset_path + '/data/' + filename,
                                resize_scale=resize_scale)
        data_np=norm_img(data_np)
        label_np = sitk_read_row(self.dataset_path + '/label/' + filename.replace('volume', 'segmentation'),
                                 resize_scale=resize_scale)
        return data_np, label_np

# 测试代码
def main():
    fixd_path  = r'E:\Files\pycharm\MIS\3DUnet\fixed'
    dataset = Lits_DataSet([16, 64, 64], 2,0.5,fixd_path,mode='train')  #batch size
    data_loader=DataLoader(dataset=dataset,shuffle=True,num_workers=2)
    for data, mask in data_loader:
        data=torch.squeeze(data,dim=0)
        mask=torch.squeeze(mask,dim=0)
        print(data.shape, mask.shape)
if __name__ == '__main__':
    main()
