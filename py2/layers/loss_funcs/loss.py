import sys
sys.path.insert(0, '../..')

import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, BCELoss

from layers.loss_funcs import lovasz_losses as L


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, logits, targets, threshold):
        num = logits.size(0)

        logits = logits > threshold
        targets = targets > 0.5

        logits = logits.view(num, -1)
        targets = targets.view(num, -1)
        intersection = (logits * targets)

        score = 2. * (intersection.sum(1) + 1.).float() / (logits.sum(1) + targets.sum(1) + 2.).float()
        score[score >= 1] = 1
        score = 1 - score.sum() / num

        return score


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets, threshold=0.5):
        bce = self.bce(logits, targets)
        dice = self.dice(logits, targets, threshold)
        return 0.5 * bce + dice


class SymmetricLovaszLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SymmetricLovaszLoss, self).__init__()
    
    def forward(self, logits, targets):
        return ((L.lovasz_hinge(logits, targets, per_image=True)) + (L.lovasz_hinge(-logits, 1-targets, per_image=True))) / 2
