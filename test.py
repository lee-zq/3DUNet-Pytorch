"""
在测试集目录中进行测试，给出性能评价指标和可视化结果
"""

from dataset.dataset_lits import Lits_DataSet
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import config
from utils import logger, init_util, metrics,common
from dataset.test_dataset import test_Datasets,to_one_hot_3d,Recompone_tool
import SimpleITK as sitk
import os
import numpy as np
from utils.common import load_file_name_list

def test(model, dataset, save_path, filename):
    dataloader = DataLoader(dataset=dataset, batch_size=4, num_workers=0, shuffle=False)
    model.eval()
    save_tool = Recompone_tool(save_path,filename,dataset.ori_shape,dataset.new_shape,dataset.cut)
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
    val_loss = metrics.DiceMeanLoss()(pred, target)
    val_dice0 = metrics.dice(pred, target, 0)
    val_dice1 = metrics.dice(pred, target, 1)
    val_dice2 = metrics.dice(pred, target, 2)

    pred_img = torch.argmax(pred,dim=1)
    img = sitk.GetImageFromArray(np.squeeze(np.array(pred_img.numpy(),dtype='uint8'),axis=0))
    sitk.WriteImage(img, os.path.join(save_path, filename))

    # save_tool.save(filename)
    print('\nAverage loss: {:.4f}\tdice0: {:.4f}\tdice1: {:.4f}\tdice2: {:.4f}\t\n'.format(
        val_loss, val_dice0, val_dice1, val_dice2))
    return val_loss, val_dice0, val_dice1, val_dice2

if __name__ == '__main__':
    args = config.args
    device = torch.device('cpu' if args.cpu else 'cuda')
    # model info
    # model = UNet(1, [32, 48, 64, 96, 128], 3, net_mode='3d',conv_block=RecombinationBlock).to(device)
    # model.load_state_dict(torch.load('./output/{}/state.pkl'.format(args.save)))
    model = torch.load('./output/{}/model.pth'.format(args.save))

    # data info
    test_data_path = './data/testdata/'
    result_save_path = './data/result/'
    cut_param = {'patch_s': 32,
                 'patch_h': 128,
                 'patch_w': 128,
                 'stride_s': 24,
                 'stride_h': 96,
                 'stride_w': 96}
    datasets = test_Datasets(test_data_path,cut_param)
    for dataset,file_idx in datasets:
        test(model, dataset,result_save_path,'result-'+file_idx)