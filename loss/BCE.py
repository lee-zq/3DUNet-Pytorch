"""

二值交叉熵损失函数
"""

import torch.nn as nn


class BCELoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        return self.bce_loss(pred, target)
