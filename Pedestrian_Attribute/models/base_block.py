import math
from models.model import *
from models.model import LabelHead 
from models.transformer import *
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


class BaseClassifier(nn.Module):
    def __init__(self, nattr, args=None):
        super().__init__()
        self.logits = nn.Sequential(
            nn.Linear(2048, nattr),
            nn.BatchNorm1d(nattr)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.qhead = QueryHead(args)
        self.lhead = LabelHead(args)
        self.bn1 = nn.BatchNorm1d(args.num_att)
        self.bn2 = nn.BatchNorm1d(nattr)
        
        

    def fresh_params(self):
        return self.parameters()

    def forward(self, feature, pos, target_a, target_b, lam ):
        x1 = self.qhead(feature, pos)
        x1 = self.bn1(x1)
        
        x2, mask, attns = self.lhead(feature,x1, pos, target_a, target_b, lam)
        x2 = self.bn2(x2)
        return x1, x2, mask, attns
       


def initialize_weights(module):
    for m in module.children():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, _BatchNorm):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)


class FeatClassifier(nn.Module):

    def __init__(self, backbone, classifier):
        super(FeatClassifier, self).__init__()

        self.backbone = backbone
        self.classifier = classifier
        self.pos_embedding = PositionEmbeddingSine(2048//2, normalize=True, maxH=8, maxW=6)

    def fresh_params(self):
        params = self.classifier.fresh_params()
        return params

    def finetune_params(self):
        return self.backbone.parameters()

    def forward(self, x, target_a=None, target_b=None, lam=None):
        feat_map = self.backbone(x)
        pos = self.pos_embedding()
        logits = self.classifier(feat_map, pos, target_a, target_b, lam)
        return logits
