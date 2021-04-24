"""
在测试集目录中进行测试，给出性能评价指标和可视化结果
"""
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from scipy import ndimage
import config
from utils import logger, weights_init, metrics,common
from dataset.dataset_lits_test import Test_Datasets,to_one_hot_3d,Recompone_tool
import SimpleITK as sitk
import os
import numpy as np
from models.UNet import UNet3D
from utils.common import load_file_name_list
from utils.metrics import DiceAverage
from collections import OrderedDict

def test(model, img_dataset, n_labels):
    dataloader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=0, shuffle=False)
    model.eval()
    test_dice = DiceAverage(n_labels)
    save_tool = Recompone_tool(img_dataset.ori_shape, img_dataset.new_shape, n_labels, img_dataset.cut_param)
    target = torch.from_numpy(np.expand_dims(img_dataset.label_np,axis=0)).long()
    target = to_one_hot_3d(target, n_labels)
    with torch.no_grad():
        for data in tqdm(dataloader,total=len(dataloader)):
            data = data.unsqueeze(1)
            data = data.float().to(device)
            output = model(data)
            save_tool.add_result(output.detach().cpu())

    pred = save_tool.recompone_overlap()
    pred = torch.unsqueeze(pred,dim=0)

    test_dice.update(pred, target)
    if n_labels==2:
        test_dice = OrderedDict({'Test dice0': test_dice.avg[0],'Test dice1': test_dice.avg[1]})
    else:
        test_dice = OrderedDict({'Test dice0': test_dice.avg[0],'Test dice1': test_dice.avg[1],'Test dice2': test_dice.avg[2]})

    pred_img = torch.argmax(pred,dim=1)
    # save_tool.save(filename)
    return test_dice, pred_img

if __name__ == '__main__':
    args = config.args
    device = torch.device('cpu' if args.cpu else 'cuda')
    # model info
    model = UNet3D(in_channels=1, filter_num_list=[16, 32, 48, 64, 96], class_num=args.n_labels).to(device)
    ckpt = torch.load('./output/{}/best_model.pth'.format(args.save))
    model.load_state_dict(ckpt['net'])

    test_log = logger.Test_Logger('./output/{}'.format(args.save),"test_log")
    # data info
    test_data_path = '/ssd/lzq/dataset/LiTS/test'
    result_save_path = './output/{}/result'.format(args.save)
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    
    cut_param = {'patch_s': 32, 'patch_h': 128, 'patch_w': 128,
            'stride_s': 16, 'stride_h': 64, 'stride_w': 64}
    datasets = Test_Datasets(test_data_path,cut_param,n_labels=args.n_labels)
    for img_dataset,file_idx in datasets:
        test_dice,pred_img = test(model, img_dataset, args.n_labels)
        test_log.update(file_idx, test_dice)
        # pred_img=ndimage.zoom(pred_img,1/args.resize_scale,order=0) #rescale
        pred_img = sitk.GetImageFromArray(np.squeeze(np.array(pred_img.numpy(),dtype='uint8'),axis=0))
        sitk.WriteImage(pred_img, os.path.join(result_save_path, 'result-'+file_idx))
