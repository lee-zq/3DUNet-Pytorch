from dataset import dataset_lits2
from torch.utils.data import DataLoader
import torch
import torch.optim as optim

import config
from models.Unet import UNet, RecombinationBlock
from utils import logger, init_util, metrics,common


def val(model, val_loader, epoch, logger):
    model.eval()
    val_loss = 0
    val_dice0 = 0
    val_dice1 = 0
    val_dice2 = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = torch.squeeze(data, dim=0)
            target = torch.squeeze(target, dim=0)
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

    logger.scalar_summary('val_loss', val_loss, epoch)
    logger.scalar_summary('val_dice0', val_dice0, epoch)
    logger.scalar_summary('val_dice1', val_dice1, epoch)
    logger.scalar_summary('val_dice2', val_dice2, epoch)
    print('\nVal set: Average loss: {:.6f}, dice0: {:.6f}\tdice1: {:.6f}\tdice2: {:.6f}\t\n'.format(
        val_loss, val_dice0, val_dice1, val_dice2))

def train(model, train_loader, optimizer, epoch, logger):
    model.train()
    train_loss = 0
    train_dice0 = 0
    train_dice1 = 0
    train_dice2 = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = torch.squeeze(data, dim=0)
        target = torch.squeeze(target, dim=0)
        data, target = data.float(), target.float()
        data, target = data.to(device), target.to(device)
        output = model(data)

        optimizer.zero_grad()

        # loss = nn.CrossEntropyLoss()(output,target)
        # loss=metrics.SoftDiceLoss()(output,target)
        # loss=nn.MSELoss()(output,target)
        loss = metrics.DiceMeanLoss()(output, target)
        # loss=metrics.WeightDiceLoss()(output,target)
        # loss=metrics.CrossEntropy()(output,target)
        loss.backward()
        optimizer.step()

        train_loss = loss
        train_dice0 = metrics.dice(output, target, 0)
        train_dice1 = metrics.dice(output, target, 1)
        train_dice2 = metrics.dice(output, target, 2)
        print(
            'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tdice0: {:.6f}\tdice1: {:.6f}\tdice2: {:.6f}\tT: {:.6f}\tP: {:.6f}\tTP: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item(),
                train_dice0, train_dice1, train_dice2,
                metrics.T(output, target), metrics.P(output, target), metrics.TP(output, target)))

    logger.scalar_summary('train_loss', float(train_loss), epoch)
    logger.scalar_summary('train_dice0', float(train_dice0), epoch)
    logger.scalar_summary('train_dice1', float(train_dice1), epoch)
    logger.scalar_summary('train_dice2', float(train_dice2), epoch)


if __name__ == '__main__':
    args = config.args
    device = torch.device('cpu' if args.cpu else 'cuda')
    # data info
    train_set = dataset_lits2.Lits_DataSet(args.crop_size, args.batch_size, args.resize_scale, args.dataset_path, mode='train')
    val_set = dataset_lits2.Lits_DataSet(args.crop_size, args.batch_size, args.resize_scale, args.dataset_path, mode='val')
    train_loader = DataLoader(dataset=train_set, shuffle=True)
    val_loader = DataLoader(dataset=val_set, shuffle=True)
    # model info
    model = UNet(1, [32, 48, 64, 96, 128], 3, net_mode='3d',conv_block=RecombinationBlock).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    init_util.print_network(model)
    # model = nn.DataParallel(model, device_ids=[0])   # multi-GPU

    logger = logger.Logger('./output/{}'.format(args.save))
    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)
        train(model, train_loader, optimizer, epoch, logger)
        val(model, val_loader, epoch, logger)
        torch.save(model, './output/{}/state.pkl'.format(args.save)) # 保存模型和参数
        # torch.save(model.state_dict(), PATH) 只保存参数