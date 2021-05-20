"""

Sensitivity Speciﬁcity loss
"""

import torch.nn as nn


class SSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        smooth = 1

        # jaccard系数的定义
        s1 = ((pred - target).pow(2) * target).sum(dim=1).sum(dim=1).sum(dim=1) / (smooth + target.sum(dim=1).sum(dim=1).sum(dim=1))

        s2 = ((pred - target).pow(2) * (1 - target)).sum(dim=1).sum(dim=1).sum(dim=1) / (smooth + (1 - target).sum(dim=1).sum(dim=1).sum(dim=1))

        # 返回的是jaccard距离
        return (0.05 * s1 + 0.95 * s2).mean()
