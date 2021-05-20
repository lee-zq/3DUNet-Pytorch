"""

加权交叉熵损失函数
统计了一下训练集下的正负样本的比例，接近20:1
"""

import torch
import torch.nn as nn


class WCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.FloatTensor([0.05, 1]).cuda()
        self.ce_loss = nn.CrossEntropyLoss(weight)

    def forward(self, pred, target):
        pred_ = torch.ones_like(pred) - pred
        pred = torch.cat((pred_, pred), dim=1)

        target = torch.long()

        return self.ce_loss(pred, target)
