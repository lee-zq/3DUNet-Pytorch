"""
此代码是在./fixed/test_name_list.txt目录中进行测试，只给出性能评价指标结果
"""

from dataset.dataset_lits import Lits_DataSet
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config
from models.Unet import UNet, RecombinationBlock
from utils import logger, init_util, metrics,common

def test(model, test_loader):
    print("Evaluation of Testset Starting...")
    model.eval()
    val_loss = 0
    val_dice0 = 0
    val_dice1 = 0
    val_dice2 = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.float(), target.float()
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = metrics.DiceMeanLoss()(output, target)
            dice0 = metrics.dice(output, target, 0)
            dice1 = metrics.dice(output, target, 1)
            dice2 = metrics.dice(output, target, 2)

            val_loss += float(loss)
            val_dice0 += float(dice0)
            val_dice1 += float(dice1)
            val_dice2 += float(dice2)

    val_loss /= len(test_loader)
    val_dice0 /= len(test_loader)
    val_dice1 /= len(test_loader)
    val_dice2 /= len(test_loader)

    print('\nTest set: Average loss: {:.6f}, dice0: {:.6f}\tdice1: {:.6f}\tdice2: {:.6f}\t\n'.format(
        val_loss, val_dice0, val_dice1, val_dice2))

if __name__ == '__main__':
    args = config.args
    device = torch.device('cpu' if args.cpu else 'cuda')
    # data info
    test_set = Lits_DataSet(args.crop_size, args.resize_scale, args.dataset_path, mode='test')
    test_loader = DataLoader(dataset=test_set,batch_size=args.batch_size,num_workers=1, shuffle=False)
    # model info
    # model = UNet(1, [32, 48, 64, 96, 128], 3, net_mode='3d',conv_block=RecombinationBlock).to(device)
    # model.load_state_dict(torch.load('./output/{}/state.pkl'.format(args.save)))
    model = torch.load('./output/{}/state.pkl'.format(args.save))
    test(model, test_loader)