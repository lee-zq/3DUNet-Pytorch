"""
在测试集目录中进行测试，给出性能评价指标和可视化结果
"""
from pickle import FALSE
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from scipy import ndimage
import config
from utils import logger, weights_init, metrics,common
from dataset.dataset_lits_test import Test_Datasets,to_one_hot_3d
import SimpleITK as sitk
import os
import numpy as np
from models.UNet import UNet3D
from models.ResUNet import ResUNet
from utils.common import load_file_name_list
from utils.metrics import DiceAverage
from collections import OrderedDict
import torch.nn as nn

def predict_one_img(model, img_dataset, args):
    dataloader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=0, shuffle=False)
    model.eval()
    test_dice = DiceAverage(args.n_labels)
    target = to_one_hot_3d(img_dataset.label, args.n_labels)
    
    with torch.no_grad():
        for data in tqdm(dataloader,total=len(dataloader)):
            data = data.to(device)
            output = model(data)
            # output = nn.functional.interpolate(output, scale_factor=(1//img_dataset.slice_down_scale,1//img_dataset.xy_down_scale,1//img_dataset.xy_down_scale), mode='trilinear', align_corners=False) # 空间分辨率恢复到原始size
            img_dataset.update_result(output.detach().cpu())

    pred = img_dataset.recompone_result()
    
    test_dice.update(pred, target)
    if args.n_labels==2:
        test_dice = OrderedDict({'Test dice0': test_dice.avg[0],'Test dice1': test_dice.avg[1]})
    else:
        test_dice = OrderedDict({'Test dice0': test_dice.avg[0],'Test dice1': test_dice.avg[1],'Test dice2': test_dice.avg[2]})

    pred_img = torch.argmax(pred,dim=1)
    pred_img = np.array(pred_img.numpy(),dtype='uint8')
    pred_img = sitk.GetImageFromArray(np.squeeze(pred_img,axis=0))

    return test_dice, pred_img

if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./experiments', args.save)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # model info
    # model = UNet3D(in_channels=1, filter_num_list=[16, 32, 48, 64, 96], class_num=args.n_labels).to(device)
    model = ResUNet(training=False).to(device)
    model = torch.nn.DataParallel(model, device_ids=[0,1])  # multi-GPU
    ckpt = torch.load('{}/best_model.pth'.format(save_path))
    model.load_state_dict(ckpt['net'])

    test_log = logger.Test_Logger(save_path,"test_log")
    # data info
    test_data_path = '/ssd/lzq/dataset/LiTS/test'
    result_save_path = '{}/result'.format(save_path)
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    
    datasets = Test_Datasets(test_data_path,args=args)
    for img_dataset,file_idx in datasets:
        test_dice,pred_img = predict_one_img(model, img_dataset, args)
        test_log.update(file_idx, test_dice)
        sitk.WriteImage(pred_img, os.path.join(result_save_path, 'result-'+file_idx+'.gz'))
