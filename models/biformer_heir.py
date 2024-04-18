import math
from collections import OrderedDict
from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg

from ops.bra_legacy import BiLevelRoutingAttention

# from positional_encodings import PositionalEncodingPermute2D, Summer
# from siren_pytorch import SirenNet


def get_pe_layer(emb_dim, pe_dim=None, name='none'):
  if name == 'none':
    return nn.Identity()
  # ... (same as original implementation)

class Block(nn.Module):
  # ... (same as original implementation)

class BiFormer(nn.Module):
  def __init__(self, depth=[3, 4, 8, 3], in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
              head_dim=64, qk_scale=None, representation_size=None,
              drop_path_rate=0., drop_rate=0.,
              use_checkpoint_stages=[],
              ########
              n_win=[7, 5, 3, 3],  # Hierarchical window sizes
              kv_downsample_mode='ada_avgpool',
              kv_per_wins=[2, 2, -1, -1],
              topks=[8, 8, -1, -1],
              side_dwconv=5,
              layer_scale_init_value=-1,
              qk_dims=[None, None, None, None],
              param_routing=False, diff_routing=False, soft_routing=False,
              pre_norm=True,
              pe=None,
              pe_stages=[0],
              before_attn_dwconv=3,
              auto_pad=False,
              #-----------------------
              kv_downsample_kernels=[4, 2, 1, 1],  # Adjust based on n_win
              kv_downsample_ratios=[4, 2, 1, 1],  # -> kv_per_win can be simplified
              mlp_ratios=[4, 4, 4, 4],
              param_attention='qkvo',
              mlp_dwconv=False):
    """
    Args:
      depth (list): depth of each stage
      img_size (int, tuple): input image size
      in_chans (int): number of input channels
      num_classes (int): number of classes for classification head
      embed_dim (list): embedding dimension of each stage
      head_dim (int): head dimension
      mlp_ratio (int): ratio of mlp hidden dim to embedding dim
      qkv_bias (bool): enable bias for qkv if True
      qk_scale (float): override default qk scale of head_dim ** -0.5 if set
      representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
      drop_rate (float): dropout rate
      attn_drop_rate (float): attention dropout rate
      drop_path_rate (float): stochastic depth rate
      norm_layer (nn.Module): normalization layer
      conv_stem (bool): whether use overlapped patch stem
    """
    super().__init__()
    self.num_classes = num_classes
    self.num_features = self.embed_dim = embed_dim

    ############ downsample layers (patch embeddings) ######################
    self.downsample_layers = nn.ModuleList()
    # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv
    stem = nn.Sequential(
      nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
      nn.BatchNorm2d(embed_dim[0] // 2),
      nn.GELU(),
      nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),  # Adjust kernel size and stride for better feature extraction
      nn.BatchNorm2d(embed_dim[0])
    )
    if (pe is not None) and 0 in pe_stages:
      stem.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
    if use_checkpoint_stages:
      stem = checkpoint_wrapper(stem)
    self.downsample_layers.append(stem)

    for i in range(3):
      downsample_layer = nn.Sequential(
          nn.Conv2d(embed_dim[i], embed_dim[i+1], kernel_size=(5, 5) if i < 2 else (3, 3), stride=(2, 2) if i < 2 else (1, 1), padding=(2, 2) if i < 2 else (1, 1)),  # Adjust kernel size and stride for hierarchical extraction
          nn.BatchNorm2d(embed_dim[i+1])
      )
      if (pe is not None) and i+1 in pe_stages:
        downsample_layer.append(get_pe_layer(emb_dim=embed_dim[i+1], name=pe))
      if use_checkpoint_stages:
        downsample_layer = checkpoint_wrapper(downsample_layer)
      self.downsample_layers.append(downsample_layer)
    ##########################################################################

    self.stages = nn.ModuleList()
    nheads = [dim // head_dim for dim in qk_dims]
    dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
    cur = 0
    for i in range(4):
      stage = nn.Sequential(
          *[Block(dim=embed_dim[i], drop_path=dp_rates[cur + j],
                  layer_scale_init_value=layer_scale_init_value,
                  topk=topks[i],
                  num_heads=nheads[i],
                  n_win=n_win[i],  # Use hierarchical window sizes
                  qk_dim=qk_dims[i],
                  qk_scale=qk_scale,
                  kv_per_win=kv_per_wins[i],
                  kv_downsample_ratio=kv_downsample_ratios[i],
                  kv_downsample_kernel=kv_downsample_kernels[i],  # Adjust based on n_win
                  kv_downsample_mode=kv_downsample_mode,
                  param_attention=param_attention,
                  param_routing=param_routing,
                  diff_routing=diff_routing,
                  soft_routing=soft_routing,
                  mlp_ratio=mlp_ratios[i],
                  mlp_dwconv=mlp_dwconv,
                  side_dwconv=side_dwconv,
                  before_attn_dwconv=before_attn_dwconv,
                  pre_norm=pre_norm,
                  auto_pad=auto_pad) for j in range(depth[i])],
      )
      if i in use_checkpoint_stages:
        stage = checkpoint_wrapper(stage)
      self.stages.append(stage)
      cur += depth[i]

    ##########################################################################
    self.norm = nn.BatchNorm2d(embed_dim[-1])
    # Representation layer
    if representation_size:
      self.num_features = representation_size
      self.pre_logits = nn.Sequential(OrderedDict([
          ('fc', nn.Linear(embed_dim, representation_size)),
          ('act', nn.Tanh())
      ]))
    else:
      self.pre_logits = nn.Identity()

    # Classifier head
    self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      trunc_normal_(m.weight, std=.02)
      if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.bias, 0)
      nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight, 1.0)
      nn.init.constant_(m.bias, 0)

  def forward(self, x):
    x = x.flatten(2)
    x = x.transpose(1, 2)  # (B, L, C) -> (B, C, L)

    for stage in self.downsample_layers:
      x = stage(x)

    for stage in self.stages:
      x = stage(x)
    x = self.norm(x)  # B, C, L

    x = x.transpose(1, 2)  # (B, C, L) -> (B, L, C)
    x = rearrange(x, 'b l c -> b (l c)')  # (B, L, C) -> (B, (L*C), 1)
    x = self.pre_logits(x)
    x = torch.flatten(x, 1)  # (B, (L*C), 1) -> (B, L*C)

    if self.head is not nn.Identity():
      x = self.head(x)
    return x

@register_model
def create_biformer(pretrained=False, **kwargs):
  """
  Creates a BiFormer model

  Args:
    pretrained (bool): whether to load pretrained weights (not supported yet)
    **kwargs: other parameters

  Returns:
    nn.Module: a BiFormer model
  """
  model = BiFormer(**kwargs)
  if pretrained:
    raise NotImplementedError('pretrained is not supported yet')
  return model