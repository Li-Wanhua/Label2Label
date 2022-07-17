# 改进计划：1.Q2L的head简化，NO querypos 2.新S2S的结构
from timm.models import create_model
import torch
import torch.nn as nn
import math
from resnet import *
from transformer import *
from utli import *
from arg import get_argparse
from loguru import logger


def create_net(args):
    if args.dataset == 'LFWA': num_classes = 40
    elif args.dataset == 'COCO': num_classes = 80
    if args.arch == 'fc':
        logger.info('fc model')
        model = create_model('resnet50', num_classes=num_classes, drop_path_rate=args.drop_path, pretrained=True,norm_layer=FrozenBatchNorm2d)
    else:
        logger.info('L2L model')
        model = L2L(args, num_classes)
    return model

class L2L(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.bn1d = args.bn1d
        self.visu = args.visu
        logger.info('back:%s nheads:%d decQ:%d decL:%d'%(args.back,args.nheads,args.decN,args.decNL))
        norm=FrozenBatchNorm2d
        self.backbone = create_model(args.back, num_classes=num_classes, drop_path_rate=args.drop_path, pretrained=True,norm_layer=norm)
        logger.info(norm)
        if args.back == 'resnet101':
            load_matched_state_dict(self.backbone, torch.load('/home/zhexuan_wh/.cache/torch/hub/checkpoints/resnet101-5d3b4d8f.pth',map_location='cpu'))
        self.pos_embedding = PositionEmbeddingSine(self.hidden_dim//2, normalize=True, maxH=round(args.height/32), maxW=round(args.width/32))
        # self.avg = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(self.hidden_dim, num_classes)
        self.qhead = QueryHead(args, num_classes)
        self.lhead = LabelHead(args, num_classes)
        self.bn2 = nn.BatchNorm1d(num_classes)
    def forward(self, x, target_a=None, target_b=None, lam=None):
        features = self.backbone.forward_features(x)
        pos = self.pos_embedding()
        out, r = self.qhead(features, pos)
        out1, mask = self.lhead(features, out, r, pos, target_a, target_b, lam)
        out1 = self.bn2(out1) if self.bn1d else out1
        return out, out1, mask, r


class QueryHead(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        self.nheads = args.nheads
        self.d_model = args.hidden_dim
        self.dim_ff = args.dim_forward
        self.drop_out = args.dropout
        decoder_layer = TransformerDecoderLayer(self.d_model, self.nheads, self.dim_ff,
                                     self.drop_out, args.act,normalize_before=not args.after_norm)
        decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, args.decN, decoder_norm, return_intermediate=False,visu=False)
        # Learnable Queries
        self.query_embed = nn.Embedding(num_classes, self.d_model)
        # TODO
        self.dropout_feas = nn.Dropout(args.dropout)
        self.classifier = GroupWiseLinear(num_classes, self.d_model)

    def forward(self, features, pos=None):
        B = features.shape[0]
        features = features.flatten(2).permute(2, 0, 1) # K B C
        pos = pos.flatten(2).permute(2, 0, 1).repeat(1, B, 1) if pos != None else None
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        features = self.decoder(query_embed, features, 
            memory_key_padding_mask=None, pos=pos, query_pos=None)
        out = self.dropout_feas(features).transpose(0,1) # [num_classes,B,C]
        x= self.classifier(out).squeeze()
        return x, out


class LabelHead(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        self.nheads = args.nheads
        self.d_model = args.hidden_dim
        self.dim_ff = args.dim_forward
        self.drop_out = args.dropout
        self.labeldp=args.labeldp
        self.num_classes = num_classes
        self.visu = False
        self.type = args.head
        self.gumbel = args.gumbel
        if self.type == 'enc':
            encoder_layer = TransformerEncoderLayer(self.d_model, self.nheads, self.dim_ff,
                                        args.ldropout, args.act, normalize_before=not args.after_norm)
            encoder_norm = nn.LayerNorm(self.d_model)
            self.encoder = TransformerEncoder(encoder_layer, args.decNL, encoder_norm)
        else:
            decoder_layer = TransformerDecoderLayer(self.d_model, self.nheads, self.dim_ff,
                                        args.ldropout, args.act, normalize_before=not args.after_norm)
            decoder_norm = nn.LayerNorm(self.d_model)
            self.decoder = TransformerDecoder(decoder_layer, args.decNL, decoder_norm, return_intermediate=False, visu=self.visu)
        self.label_embedding = nn.Embedding(2*num_classes, self.d_model)
        self.mask_embedding = nn.Embedding(num_classes, self.d_model)
        self.dropout_feas = nn.Dropout(args.dropout)
        # Attribute classifier
        self.classifier = GroupWiseLinear(num_classes, self.d_model)
    
        if self.type != 'norm' and self.type != 'enc':
            logger.warning('No mask For label head')
        if self.gumbel:
            logger.warning('Gumbel softmax was used')
    def forward(self, features, x, r, pos, target_a, target_b, lam):
        B = x.shape[0]
        if self.type != 'r2r':
            if len(x.shape)==1:
                x = x.unsqueeze(0)
            if self.gumbel:
                x = F.gumbel_softmax(torch.stack([x, 0*x], dim=2),hard=True)[:,:,0]
            else:
                x = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
            if lam is not None:
                if lam > 0.5:
                    mix_x = torch.where(x == target_a, x, target_b)
                else:
                    mix_x = torch.where(x == target_b, x, target_a)
                x = x.unsqueeze(2).repeat((1,1,self.d_model))
                x = x*self.label_embedding.weight[0:self.num_classes,:] + (1-x)*self.label_embedding.weight[self.num_classes:2*self.num_classes,:]
                mix_x = mix_x.unsqueeze(2).repeat((1,1,self.d_model))
                mix_x = mix_x*self.label_embedding.weight[0:self.num_classes,:] + (1-mix_x)*self.label_embedding.weight[self.num_classes:2*self.num_classes,:]
                x = lam * x + (1 - lam) * mix_x
            else:
                x = x.unsqueeze(2).repeat((1,1,self.d_model))
                x = x*self.label_embedding.weight[0:self.num_classes,:] + (1-x)*self.label_embedding.weight[self.num_classes:2*self.num_classes,:]
            
        else: x = r
        x = x.permute(1,0,2)
        mask = None
    
        if self.training and self.type == 'norm':
            mask_embed = self.mask_embedding.weight
            mask_embed = mask_embed.unsqueeze(1).repeat((1,B,1))
            mask = torch.rand((self.num_classes, B)).to(x.device)
            mask = torch.where(mask>self.labeldp, torch.ones_like(mask), torch.zeros_like(mask))
            mask_repeat = mask.unsqueeze(2).repeat((1,1,self.d_model))
            x = mask_repeat * x  + (1 - mask_repeat) * mask_embed
            mask = mask.permute(1,0)
        
        features = features.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1).repeat(1, B, 1) if pos != None else None
        if self.type == 'enc':
            x = self.encoder(x)
        else:
            x = self.decoder(x, features, 
                memory_key_padding_mask=None, pos=pos, query_pos=None)# get_query_pos(dim=self.d_model).unsqueeze(0).repeat([B,1,1]).permute([1,0,2]).to(pos.device))
        # x = self.decoder(x)
        if self.type != 'r2r': 
            out = self.dropout_feas(x).transpose(0,1) # [num_classes,B,C]
            x= self.classifier(out).squeeze()
        return x, mask


def get_query_pos(length=40, dim=2048):
    pos_1 = torch.tensor(list(range(length))).unsqueeze(1).repeat([1,dim])
    pos_2 = torch.tensor(list(range(dim))).unsqueeze(0).repeat([length,1])
    pos = torch.sin(pos_1/10000**(2*pos_2/dim))
    return pos

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, maxH=30, maxW=30):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.maxH = maxH
        self.maxW = maxW
        pe = self._gen_pos_buffer()
        self.register_buffer('pe', pe)

    def _gen_pos_buffer(self):
        _eyes = torch.ones((1, self.maxH, self.maxW))
        y_embed = _eyes.cumsum(1, dtype=torch.float32)
        x_embed = _eyes.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self):
        return self.pe.repeat((1,1,1,1))

class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.weight = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        for i in range(self.num_class):
            self.weight[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.weight * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

if __name__ == '__main__':
    from thop import profile
    with torch.no_grad():
        args = get_argparse()
        args.after_norm=False
        # args.head = 'com'
        model = L2L(args)
        model.eval()
        input = torch.randn([1,3,224,224])
        logger.info(model(x=input))
        # macs, params = profile(model, inputs=(input, ))
        # _, x, _, _ = model(torch.randn((4,3,250,250)))
        # logger.info(macs)
        # logger.info(params)