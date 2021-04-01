from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from .nn.module import ResBlock

class Up(nn.Module):
    def __init__(self, down_in_channels, in_channels, out_channels, conv_block):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(down_in_channels, down_in_channels, 2, stride=2)

        self.conv = RecombinationBlock(in_channels + down_in_channels, out_channels)

    def forward(self, down_x, x):
        up_x = self.up(down_x)

        x = torch.cat((up_x, x), dim=1)

        x = self.conv(x)

        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block):
        super(Down, self).__init__()
        maxpool = nn.MaxPool3d
        self.conv = RecombinationBlock(in_channels, out_channels)

        self.down = maxpool(2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        out = self.down(x)

        return x, out


class SegSEBlock(nn.Module):
    def __init__(self, in_channels, rate=2):
        super(SegSEBlock, self).__init__()
        conv = nn.Conv3d
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
    def __init__(self, in_channels, out_channels, batch_normalization=True, kernel_size=3):
        super(RecombinationBlock, self).__init__()
        conv = nn.Conv3d
        bn = nn.BatchNorm3d

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

        self.segse_block = SegSEBlock(self.expan_channels)

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


class UNet3D(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, conv_block=ResBlock):
        super(UNet3D, self).__init__()
        conv = nn.Conv3d
        self.inc = conv(in_channels, 16, 1)

        # down
        self.down1 = Down(16, filter_num_list[0], conv_block=conv_block)
        self.down2 = Down(filter_num_list[0], filter_num_list[1], conv_block=conv_block)
        self.down3 = Down(filter_num_list[1], filter_num_list[2], conv_block=conv_block)
        self.down4 = Down(filter_num_list[2], filter_num_list[3], conv_block=conv_block)

        self.bridge = conv_block(filter_num_list[3], filter_num_list[4])

        # up
        self.up1 = Up(filter_num_list[4], filter_num_list[3], filter_num_list[3], conv_block=conv_block)
        self.up2 = Up(filter_num_list[3], filter_num_list[2], filter_num_list[2], conv_block=conv_block)
        self.up3 = Up(filter_num_list[2], filter_num_list[1], filter_num_list[1], conv_block=conv_block)
        self.up4 = Up(filter_num_list[1], filter_num_list[0], filter_num_list[0], conv_block=conv_block)

        self.class_conv = conv(filter_num_list[0], class_num, 1)

    def forward(self, x):

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
        x = F.softmax(x,dim=1)

        return x


'''
if __name__ == '__main__':
    torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(1,[32,48,64,96,128],3).to(device)
    x=torch.rand(4,1,64,96,96)
    x=x.to(device)
    model.forward(x)
'''
