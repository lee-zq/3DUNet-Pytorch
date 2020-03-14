from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# 未测试
class SEBlock(nn.Module):
    def __init__(self,in_channels,out_channels,net_mode='2d'):
        super(SEBlock,self).__init__()

        if net_mode == '2d':
            self.gap=nn.AdaptiveAvgPool2d(1)
            conv=nn.Conv2d
        elif net_mode == '3d':
            self.gap=nn.AdaptiveAvgPool3d(1)
            conv=nn.Conv3d
        else:
            self.gap=None
            conv=None

        self.conv1=conv(in_channels,out_channels,1)
        self.conv2=conv(in_channels,out_channels,1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        inpu=x
        x=self.gap(x)
        x=self.conv1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.sigmoid(x)

        return inpu*x

# 未测试
class DenseBlock(nn.Module):
    def __init__(self,channels,conv_num,net_mode='2d'):
        super(DenseBlock,self).__init__()
        self.conv_num=conv_num
        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.relu=nn.ReLU()
        self.conv_list=[]
        self.bottle_conv_list=[]
        for i in conv_num:
            self.bottle_conv_list.append(conv(channels*(i+1),channels*4,1))
            self.conv_list.append(conv(channels*4,channels,3,padding=1))


    def forward(self,x):

        res_x=[]
        res_x.append(x)

        for i in self.conv_num:
            inputs=torch.cat(res_x,dim=1)
            x=self.bottle_conv_list[i](inputs)
            x=self.relu(x)
            x=self.conv_list[i](x)
            x=self.relu(x)
            res_x.append(x)

        return x



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, net_mode='2d'):
        super(ResBlock, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = None
            bn = None

        self.conv1 = conv(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = bn(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(out_channels, out_channels, 3, stride=stride, padding=1)
        self.bn2 = bn(out_channels)

        if in_channels!=out_channels:
            self.res_conv=conv(in_channels,out_channels,1,stride=stride)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            res=self.res_conv(x)
        else:
            res=x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        out = x + res
        out = self.relu(out)

        return out


class Up(nn.Module):
    def __init__(self, down_in_channels, in_channels, out_channels, conv_block, interpolation=True, net_mode='2d'):
        super(Up, self).__init__()

        if net_mode == '2d':
            inter_mode = 'bilinear'
            trans_conv = nn.ConvTranspose2d
        elif net_mode == '3d':
            inter_mode = 'trilinear'
            trans_conv = nn.ConvTranspose3d
        else:
            inter_mode = None
            trans_conv = None

        if interpolation == True:
            self.up = nn.Upsample(scale_factor=2, mode=inter_mode, align_corners=True)
        else:
            self.up = trans_conv(down_in_channels, down_in_channels, 2, stride=2)

        self.conv = RecombinationBlock(in_channels + down_in_channels, out_channels, net_mode=net_mode)

    def forward(self, down_x, x):
        up_x = self.up(down_x)

        x = torch.cat((up_x, x), dim=1)

        x = self.conv(x)

        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block, net_mode='2d'):
        super(Down, self).__init__()
        if net_mode == '2d':
            maxpool = nn.MaxPool2d
        elif net_mode == '3d':
            maxpool = nn.MaxPool3d
        else:
            maxpool = None

        self.conv = RecombinationBlock(in_channels, out_channels, net_mode=net_mode)

        self.down = maxpool(2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        out = self.down(x)

        return x, out


class SegSEBlock(nn.Module):
    def __init__(self, in_channels, rate=2, net_mode='2d'):
        super(SegSEBlock, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.in_channels = in_channels
        self.rate = rate
        self.dila_conv = conv(self.in_channels, self.in_channels // self.rate, 3, padding=2, dilation=self.rate)
        self.conv1 = conv(self.in_channels // self.rate, self.in_channels, 1)

    def forward(self, input):
        x = self.dila_conv(input)
        x = self.conv1(x)
        x = nn.Sigmoid()(x)

        return x


class RecombinationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normalization=True, kernel_size=3, net_mode='2d'):
        super(RecombinationBlock, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = None
            bn = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bach_normalization = batch_normalization
        self.kerenl_size = kernel_size
        self.rate = 2
        self.expan_channels = self.out_channels * self.rate

        self.expansion_conv = conv(self.in_channels, self.expan_channels, 1)
        self.skip_conv = conv(self.in_channels, self.out_channels, 1)
        self.zoom_conv = conv(self.out_channels * self.rate, self.out_channels, 1)

        self.bn = bn(self.expan_channels)
        self.norm_conv = conv(self.expan_channels, self.expan_channels, self.kerenl_size, padding=1)

        self.segse_block = SegSEBlock(self.expan_channels, net_mode=net_mode)

    def forward(self, input):
        x = self.expansion_conv(input)

        for i in range(1):
            if self.bach_normalization:
                x = self.bn(x)
            x = nn.ReLU6()(x)
            x = self.norm_conv(x)

        se_x = self.segse_block(x)

        x = x * se_x

        x = self.zoom_conv(x)

        skip_x = self.skip_conv(input)
        out = x + skip_x

        return out


class UNet(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, conv_block=RecombinationBlock, net_mode='2d'):
        super(UNet, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.inc = conv(in_channels, 16, 1)

        # down
        self.down1 = Down(16, filter_num_list[0], conv_block=conv_block, net_mode=net_mode)
        self.down2 = Down(filter_num_list[0], filter_num_list[1], conv_block=conv_block, net_mode=net_mode)
        self.down3 = Down(filter_num_list[1], filter_num_list[2], conv_block=conv_block, net_mode=net_mode)
        self.down4 = Down(filter_num_list[2], filter_num_list[3], conv_block=conv_block, net_mode=net_mode)

        self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)

        # up
        self.up1 = Up(filter_num_list[4], filter_num_list[3], filter_num_list[3], conv_block=conv_block,
                      net_mode=net_mode)
        self.up2 = Up(filter_num_list[3], filter_num_list[2], filter_num_list[2], conv_block=conv_block,
                      net_mode=net_mode)
        self.up3 = Up(filter_num_list[2], filter_num_list[1], filter_num_list[1], conv_block=conv_block,
                      net_mode=net_mode)
        self.up4 = Up(filter_num_list[1], filter_num_list[0], filter_num_list[0], conv_block=conv_block,
                      net_mode=net_mode)

        self.class_conv = conv(filter_num_list[0], class_num, 1)

    def forward(self, input):

        x = input

        x = self.inc(x)

        conv1, x = self.down1(x)

        conv2, x = self.down2(x)

        conv3, x = self.down3(x)

        conv4, x = self.down4(x)

        x = self.bridge(x)

        x = self.up1(x, conv4)

        x = self.up2(x, conv3)

        x = self.up3(x, conv2)

        x = self.up4(x, conv1)

        x = self.class_conv(x)

        x = nn.Softmax(1)(x)

        return x


'''
def main():
    torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    model = UNet(1,[32,48,64,96,128],3,net_mode='3d').to(device)
    x=torch.rand(4,1,64,96,96)
    x=x.to(device)
    model.forward(x)

if __name__ == '__main__':
    main()
'''
