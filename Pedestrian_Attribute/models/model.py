
from timm.models import create_model
import timm.models.resnet
from timm.models.pit import Transformer
import torch
import torch.nn as nn
import math
from models.transformer import *




class QueryHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.nheads = args.nheads
        self.d_model = args.hidden_dim
        self.dim_ff = args.dim_forward
        self.drop_out = args.dropout
        self.num_attr = args.num_att
        encoder_layer = TransformerEncoderLayer(self.d_model, self.nheads, self.dim_ff,
                                     self.drop_out, 'gelu' ,normalize_before=not args.after_norm)
        encoder_norm = nn.LayerNorm(self.d_model)
        self.encoder = TransformerEncoder(encoder_layer, 1, encoder_norm)
        decoder_layer = TransformerDecoderLayer(self.d_model, self.nheads, self.dim_ff,
                                     self.drop_out, 'gelu',normalize_before=not args.after_norm)
        decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, args.decN, decoder_norm, return_intermediate=False)
        # Learnable Queries
        self.query_embed = nn.Embedding(self.num_attr, self.d_model)
        # TODO
        self.dropout_feas = nn.Dropout(self.drop_out)
        # Attribute classifier
        self.classifier = GroupWiseLinear(self.num_attr, self.d_model)

    def forward(self, features, pos=None):
        B,C,H,W = features.shape
        features = features.flatten(2).permute(2, 0, 1) # K B C
        pos = pos.flatten(2).permute(2, 0, 1).repeat(1, B, 1) if pos != None else None
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        query_embed = self.encoder(query_embed)
        features, _ = self.decoder(query_embed, features, 
            memory_key_padding_mask=None, pos=pos, query_pos=None)
        out = self.dropout_feas(features).transpose(0,1) # [self.num_attr,B,C]
        x= self.classifier(out).squeeze()
        return x
class LabelHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.nheads = args.nheads
        self.d_model = args.hidden_dim
        self.dim_ff = args.dim_forward
        self.num_attr = args.num_att
        self.drop_out = args.ldropout
        self.labeldp=args.labeldp
        decoder_layer = TransformerDecoderLayer(self.d_model, self.nheads, self.dim_ff,
                                     self.drop_out, 'gelu',normalize_before=not args.after_norm)
        decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, args.decNL, decoder_norm, return_intermediate=False)
        self.label_embedding = nn.Embedding(2*self.num_attr, self.d_model)
        self.mask_embedding = nn.Embedding(self.num_attr, self.d_model)
        # TODO 中间用大dropout？？？？
        # print("dropout dropout")
        self.dropout_feas = nn.Dropout(args.dropout)
        self.classifier = GroupWiseLinear(self.num_attr, self.d_model)

    def forward(self,features, x, pos, target_a, target_b, lam):
        B, _ = x.shape
        x = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
        if lam is not None:
            if lam > 0.5:
                mix_x = torch.where(x == target_a, x, target_b)
            else:
                mix_x = torch.where(x == target_b, x, target_a)
            x = x.unsqueeze(2).repeat((1,1,self.d_model))
            x = x*self.label_embedding.weight[0:self.num_attr,:] + (1-x)*self.label_embedding.weight[self.num_attr:2*self.num_attr,:]
            mix_x = mix_x.unsqueeze(2).repeat((1,1,self.d_model))
            mix_x = mix_x*self.label_embedding.weight[0:self.num_attr,:] + (1-mix_x)*self.label_embedding.weight[self.num_attr:2*self.num_attr,:]
            x = lam * x + (1 - lam) * mix_x
        else:
            x = x.unsqueeze(2).repeat((1,1,self.d_model))
            x = x*self.label_embedding.weight[0:self.num_attr,:] + (1-x)*self.label_embedding.weight[self.num_attr:2*self.num_attr,:]
        # x = x.unsqueeze(2).repeat((1,1,self.d_model))
        # k.     x = x*self.label_embedding.weight[0:self.num_attr,:] + (1-x)*self.label_embedding.weight[self.num_attr:2*self.num_attr,:]
        mask_embed = self.mask_embedding.weight
        mask_embed = mask_embed.unsqueeze(1).repeat((1,B,1))
        x = x.permute(1,0,2)
        mask = None
        if self.training:
            mask = torch.rand((self.num_attr, B)).to(x.device)
            mask = torch.where(mask>self.labeldp, torch.ones_like(mask), torch.zeros_like(mask))
            mask_repeat = mask.unsqueeze(2).repeat((1,1,self.d_model))
            x = mask_repeat * x + (1 - mask_repeat) * mask_embed
            mask = mask.permute(1,0)
        features = features.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1).repeat(1, B, 1) if pos != None else None
        features, attns = self.decoder(x, features, 
            memory_key_padding_mask=None, pos=pos, query_pos=None)
        out = self.dropout_feas(features).transpose(0,1) # [self.num_attr,B,C]
        x= self.classifier(out).squeeze()
        return x, mask, attns



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
    pass
    