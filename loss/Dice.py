"""

Dice loss
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        smooth = 1

        # dice系数的定义
        dice = 2 * (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                            target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        # 返回的是dice距离
        return torch.clamp((1 - dice).mean(), 0, 1)



class DiceLossV2(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        # pred = pred.squeeze(dim=1)

        smooth = 1

        # dice系数的定义
        dice0 = 2 * (pred[:,0] * target[:,0]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,0].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                            target[:,0].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        dice1 = 2 * (pred[:,1] * target[:,1]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,1].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                            target[:,1].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        dice = (dice1+dice0) / 2.0
        # 返回的是dice距离
        return torch.clamp((1 - dice).mean(), 0, 1)
