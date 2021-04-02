"""
在测试集目录中进行测试，给出性能评价指标和可视化结果
"""
from dataset.dataset_lits import Lits_DataSet
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from scipy import ndimage
import config
from utils import logger, init_util, metrics,common
from dataset.test_dataset import test_Datasets,to_one_hot_3d,Recompone_tool
import SimpleITK as sitk
import os
import numpy as np
from models.UNet import UNet3D
from utils.common import load_file_name_list
from utils.metrics import DiceAverage

def test(model, dataset):
    dataloader = DataLoader(dataset=dataset, batch_size=4, num_workers=0, shuffle=False)
    model.eval()
    save_tool = Recompone_tool(dataset.ori_shape,dataset.new_shape,dataset.cut)
    target = torch.from_numpy(np.expand_dims(dataset.label_np,axis=0)).long()
    target = to_one_hot_3d(target)
    with torch.no_grad():
        for data in tqdm(dataloader,total=len(dataloader)):
            data = data.unsqueeze(1)
            data = data.float().to(device)
            output = model(data)
            save_tool.add_result(output.detach().cpu())

    pred = save_tool.recompone_overlap()
    pred = torch.unsqueeze(pred,dim=0)

    test_dice = DiceAverage.get_dices(pred, target)

    pred_img = torch.argmax(pred,dim=1)
    # save_tool.save(filename)
    return test_dice, pred_img

if __name__ == '__main__':
    args = config.args
    device = torch.device('cpu' if args.cpu else 'cuda')
    # model info
    model = UNet3D(in_channels=1, filter_num_list=[16, 32, 48, 64, 96], class_num=3).to(device)
    ckpt = torch.load('./output/{}/best_model.pth'.format(args.save))
    model.load_state_dict(ckpt['net'])

    # data info
    test_data_path = r'/ssd/lzq/dataset/LiTS/test'
    result_save_path = r'./output/{}/result'.format(args.save)
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    cut_param = {'patch_s': 32, 'patch_h': 128, 'patch_w': 128,
                 'stride_s': 24, 'stride_h': 96, 'stride_w': 96}
    datasets = test_Datasets(test_data_path,cut_param,resize_scale=1)
    for dataset,file_idx in datasets:
        test_dice,pred_img = test(model, dataset)
        print(test_dice)
        # pred_img=ndimage.zoom(pred_img,[1/args.resize_scale,1/args.resize_scale,1/args.resize_scale],order=0) #rescale
        pred_img = sitk.GetImageFromArray(np.squeeze(np.array(pred_img.numpy(),dtype='uint8'),axis=0))
        sitk.WriteImage(pred_img, os.path.join(result_save_path, 'result-'+file_idx))