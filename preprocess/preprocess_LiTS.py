import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage

class LITS_fix:
    def __init__(self, raw_dataset_path,fixed_dataset_path):
        self.raw_root_path = raw_dataset_path
        self.fixed_path = fixed_dataset_path

        if not os.path.exists(self.fixed_path):    # 创建保存目录
            os.makedirs(self.fixed_path+'data')
            os.makedirs(self.fixed_path+'label')

        self.fix_data()                            # 对原始图像进行修剪并保存
        self.write_train_val_test_name_list()      # 创建索引txt文件

    def fix_data(self):
        upper = 200
        lower = -200
        expand_slice = 20  # 轴向外侧扩张的slice数量
        size = 48  # 取样的slice数量

        print('the raw dataset total numbers of samples is :',len(os.listdir(self.raw_root_path + 'data')) )
        for ct_file in os.listdir(self.raw_root_path + 'data/'):
            print(ct_file)
            # 将CT和金标准入读内存
            ct = sitk.ReadImage(os.path.join(self.raw_root_path + 'data/', ct_file), sitk.sitkInt16)
            ct_array = sitk.GetArrayFromImage(ct)

            seg = sitk.ReadImage(os.path.join(self.raw_root_path + 'label/', ct_file.replace('volume', 'segmentation')),
                                 sitk.sitkInt8)
            seg_array = sitk.GetArrayFromImage(seg)

            print(ct_array.shape, seg_array.shape)

            # 将灰度值在阈值之外的截断掉
            ct_array[ct_array > upper] = upper
            ct_array[ct_array < lower] = lower

            # 找到肝脏区域开始和结束的slice，并各向外扩张
            z = np.any(seg_array, axis=(1, 2))
            start_slice, end_slice = np.where(z)[0][[0, -1]]

            # 两个方向上各扩张个slice
            if start_slice - expand_slice < 0:
                start_slice = 0
            else:
                start_slice -= expand_slice

            if end_slice + expand_slice >= seg_array.shape[0]:
                end_slice = seg_array.shape[0] - 1
            else:
                end_slice += expand_slice

            print(str(start_slice) + '--' + str(end_slice))
            # 如果这时候剩下的slice数量不足size，直接放弃，这样的数据很少
            if end_slice - start_slice + 1 < size:
                print(ct_file, 'too little slice，give up the sample')
                continue

            ct_array = ct_array[start_slice:end_slice + 1, :, :]
            seg_array = sitk.GetArrayFromImage(seg)
            seg_array = seg_array[start_slice:end_slice + 1, :, :]

            new_ct = sitk.GetImageFromArray(ct_array)
            new_seg = sitk.GetImageFromArray(seg_array)

            sitk.WriteImage(new_ct, os.path.join(self.fixed_path + 'data/', ct_file))
            sitk.WriteImage(new_seg,
                            os.path.join(self.fixed_path + 'label/', ct_file.replace('volume', 'segmentation')))
    
    def write_train_val_test_name_list(self):
        data_name_list = os.listdir(self.fixed_path + "/" + "data")
        data_num = len(data_name_list)
        print('the fixed dataset total numbers of samples is :', data_num)
        random.shuffle(data_name_list)

        train_rate = 0.8
        val_rate = 0.2

        assert val_rate+train_rate == 1.0
        train_name_list = data_name_list[0:int(data_num*train_rate)]
        val_name_list = data_name_list[int(data_num*train_rate):int(data_num*(train_rate + val_rate))]

        self.write_name_list(train_name_list, "train_name_list.txt")
        self.write_name_list(val_name_list, "val_name_list.txt")

    def write_name_list(self, name_list, file_name):
        f = open(self.fixed_path + file_name, 'w')
        for i in range(len(name_list)):
            f.write(str(name_list[i]) + "\n")
        f.close()

def main():
    raw_dataset_path = 'F:/datasets/LiTS/train/'
    fixed_dataset_path = '../fixed_data/'
    
    LITS_fix(raw_dataset_path,fixed_dataset_path)

if __name__ == '__main__':
    main()
