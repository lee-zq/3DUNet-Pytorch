from dataset.dataset_lits import Lits_DataSet
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
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

    logger.scalar_summary('val_loss', val_loss, epoch)
    logger.scalar_summary('val_dice0', val_dice0, epoch)
    logger.scalar_summary('val_dice1', val_dice1, epoch)
    logger.scalar_summary('val_dice2', val_dice2, epoch)
    print('Val performance: Average loss: {:.4f}\tdice0: {:.4f}\tdice1: {:.4f}\tdice2: {:.4f}\t\n'.format(
        val_loss, val_dice0, val_dice1, val_dice2))


def train(model, train_loader, optimizer, epoch, logger):
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
        loss = metrics.DiceMeanLoss()(output, target)
        # loss=metrics.WeightDiceLoss()(output,target)
        # loss=metrics.CrossEntropy()(output,target)
        loss.backward()
        optimizer.step()

        train_loss += loss
        train_dice0 += metrics.dice(output, target, 0)
        train_dice1 += metrics.dice(output, target, 1)
        train_dice2 += metrics.dice(output, target, 2)
    train_loss /= len(train_loader)
    train_dice0 /= len(train_loader)
    train_dice1 /= len(train_loader)
    train_dice2 /= len(train_loader)

    print('Train Epoch: {} \tLoss: {:.4f}\tdice0: {:.4f}\tdice1: {:.4f}\tdice2: {:.4f}'.format(
            epoch, train_loss, train_dice0, train_dice1, train_dice2))

    logger.scalar_summary('train_loss', float(train_loss), epoch)
    logger.scalar_summary('train_dice0', float(train_dice0), epoch)
    logger.scalar_summary('train_dice1', float(train_dice1), epoch)
    logger.scalar_summary('train_dice2', float(train_dice2), epoch)

if __name__ == '__main__':
    args = config.args
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

    logger = logger.Logger('./output/{}'.format(args.save))
    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)
        train(model, train_loader, optimizer, epoch, logger)
        val(model, val_loader, epoch, logger)
        torch.save(model, './output/{}/model.pth'.format(args.save))  # Save model with parameters
        # torch.save(model.state_dict(), './output/{}/model.pth'.format(args.save))  # Only save parameters