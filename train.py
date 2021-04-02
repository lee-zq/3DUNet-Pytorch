from dataset.dataset_lits import Lits_DataSet
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config
from models.UNet import UNet3D, RecombinationBlock
from utils import logger, init_util, metrics, common
import os
import numpy as np
from collections import OrderedDict

def val(model, val_loader, criterion):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage()
    with torch.no_grad():
        for idx,(data, target) in tqdm(enumerate(val_loader),total=len(val_loader)):
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target)
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss=criterion(output, target)
            
            val_loss.update(loss.item(),data.size(0))
            val_dice.update(output, target)

    return OrderedDict({'Val Loss': val_loss.avg, 'Val dice0': val_dice.avg[0],
                        'Val dice1': val_dice.avg[1],'Val dice2': val_dice.avg[2]})

def train(model, train_loader, optimizer, criterion):
    print("=======Epoch:{}=======lr:{}".format(epoch,optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage()

    for idx, (data, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
        data, target = data.float(), target.long()
        target = common.to_one_hot_3d(target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.item(),data.size(0))
        train_dice.update(output, target)

    return OrderedDict({'Train Loss': train_loss.avg, 'Train dice0': train_dice.avg[0],
                       'Train dice1': train_dice.avg[1],'Train dice2': train_dice.avg[2]})

if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./output', args.save)
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # data info
    train_set = Lits_DataSet(args.crop_size, args.resize_scale, args.dataset_path, mode='train')
    val_set = Lits_DataSet(args.crop_size, args.resize_scale, args.dataset_path, mode='val')
    # train_set = dataset_lits_faster.Lits_DataSet(args.crop_size, args.batch_size, args.resize_scale, args.dataset_path, mode='train')
    # val_set = dataset_lits_faster.Lits_DataSet(args.crop_size, args.batch_size, args.resize_scale, args.dataset_path, mode='val')
    train_loader = DataLoader(dataset=train_set,batch_size=args.batch_size,num_workers=args.n_threads, shuffle=True)
    val_loader = DataLoader(dataset=val_set,batch_size=1,num_workers=args.n_threads, shuffle=False)

    # model info
    model = UNet3D(in_channels=1, filter_num_list=[16, 32, 48, 64, 96], class_num=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    init_util.print_network(model)
    # model = nn.DataParallel(model, device_ids=[0,1])  # multi-GPU
    
    # loss=metrics.SoftDiceLoss()
    # loss = metrics.DiceMeanLoss()
    # loss=metrics.WeightDiceLoss()
    # loss=metrics.DiceMeanLoss()
    loss=metrics.DiceLoss(weight=np.array([0.2,0.3,0.5]))
    
    log = logger.Logger(save_path)

    best = [0,np.inf] # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器
    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)
        train_log = train(model, train_loader, optimizer, loss)
        val_log = val(model, val_loader, loss)
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