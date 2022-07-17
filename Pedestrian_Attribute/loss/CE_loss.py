from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.function import ratio2weight
from loguru import logger

class CEL_Sigmoid(nn.Module):

    def __init__(self, ratio=0.5,pos_weight=1, sample_weight=None, size_average=True):
        super(CEL_Sigmoid, self).__init__()
        self.ratio = ratio
        self.sample_weight = sample_weight
        self.size_average = size_average
        self.pos_weight = pos_weight
        # logger.warning('loss ratio for mask is 0')
    def forward(self, logits, targets, mask=None):
        batch_size = logits.shape[0]
        mask = torch.ones_like(logits) if mask == None else torch.where(mask==0, torch.ones_like(mask), self.ratio*torch.ones_like(mask))
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=self.pos_weight*torch.ones_like(logits))
        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        if self.sample_weight is not None:
            weight = ratio2weight(targets_mask, self.sample_weight)
            loss = (loss * weight.cuda() * mask.cuda())
            # loss = (loss * weight.cuda())
        else:
            print('fuck')
        loss = loss.sum() / (batch_size) if self.size_average else loss.sum()
        # loss = loss.sum() / (batch_size *35) if self.size_average else loss.sum()
        return loss
