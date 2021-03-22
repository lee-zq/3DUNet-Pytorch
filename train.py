from dataset.dataset_lits import Lits_DataSet
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config
from models.Unet import UNet, RecombinationBlock
from utils import logger, init_util, metrics,common
import os
import numpy as np
from collections import OrderedDict
# update 2021.03.22
def val(model, val_loader):
    model.eval()
    val_loss = 0
    val_dice0 = 0
    val_dice1 = 0
    val_dice2 = 0
    with torch.no_grad():
        for idx,(data, target) in tqdm(enumerate(val_loader),total=len(val_loader)):
            target = common.to_one_hot_3d(target.long())
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

    val_loss /= len(val_loader)
    val_dice0 /= len(val_loader)
    val_dice1 /= len(val_loader)
    val_dice2 /= len(val_loader)

    return OrderedDict({'Val Loss': val_loss, 'Val dice0': val_dice0,
                        'Val dice1': val_dice1,'Val dice2': val_dice2})


def train(model, train_loader, optimizer):
    print("=======Epoch:{}=======".format(epoch))
    model.train()
    train_loss = 0
    train_dice0 = 0
    train_dice1 = 0
    train_dice2 = 0
    for idx, (data, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
        target = common.to_one_hot_3d(target.long())
        data, target = data.float(), target.float()
        data, target = data.to(device), target.to(device)
        output = model(data)
        optimizer.zero_grad()

        # loss = nn.CrossEntropyLoss()(output,target)
        # loss=metrics.SoftDiceLoss()(output,target)
        # loss=nn.MSELoss()(output,target)
        # loss = metrics.DiceMeanLoss()(output, target)
        loss=metrics.WeightDiceLoss()(output,target)
        # loss=metrics.CrossEntropy()(output,target)
        loss.backward()
        optimizer.step()

        train_loss += float(loss)
        train_dice0 += float(metrics.dice(output, target, 0))
        train_dice1 += float(metrics.dice(output, target, 1))
        train_dice2 += float(metrics.dice(output, target, 2))
    train_loss /= len(train_loader)
    train_dice0 /= len(train_loader)
    train_dice1 /= len(train_loader)
    train_dice2 /= len(train_loader)

    return OrderedDict({'Train Loss': train_loss, 'Train dice0': train_dice0,
                       'Train dice1': train_dice1,'Train dice2': train_dice2})

if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./output', args.save)
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # data info
    train_set = Lits_DataSet(args.crop_size, args.resize_scale, args.dataset_path, mode='train')
    val_set = Lits_DataSet(args.crop_size, args.resize_scale, args.dataset_path, mode='val')
    train_loader = DataLoader(dataset=train_set,batch_size=args.batch_size,num_workers=1, shuffle=True)
    val_loader = DataLoader(dataset=val_set,batch_size=args.batch_size,num_workers=1, shuffle=True)
    # model info
    model = UNet(1, [16, 32, 48, 64, 96], 3, net_mode='3d',conv_block=RecombinationBlock).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    init_util.print_network(model)
    # model = nn.DataParallel(model, device_ids=[0,1])  # multi-GPU

    log = logger.Logger(save_path)

    best = [0,np.inf] # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器
    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)
        train_log = train(model, train_loader, optimizer)
        val_log = val(model, val_loader)
        log.update(epoch,train_log,val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['Val Loss'] < best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val Loss']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0],best[1]))
        # early stopping
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()