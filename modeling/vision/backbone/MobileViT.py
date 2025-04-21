# --------------------------------------------------------
# MobileViT for Semantic Segmentation
# Based on: https://arxiv.org/abs/2110.02178
# Adapted for Detectron2 by OpenAI Assistant
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple

from detectron2.utils.file_io import PathManager
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from .build import register_backbone

class ConvBNAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1, act_layer=nn.SiLU):
        super().__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_chs)
        self.act = act_layer()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# class MobileViTBlock(nn.Module):
#     def __init__(self, dim, depth, kernel_size, patch_size):
#         super().__init__()
#         self.local_rep = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim),
#             nn.Conv2d(dim, dim, 1)
#         )

#         self.patch_size = patch_size
#         ph, pw = patch_size
#         self.d_model = dim * ph * pw  # Calculate the correct embedding dimension

#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=dim*2, batch_first=True),
#             num_layers=depth
#         )
#         self.fusion = nn.Conv2d(dim * 2, dim, 1)

#     def unfold(self, x):
#         B, C, H, W = x.shape
#         ph, pw = self.patch_size
#         new_H, new_W = math.ceil(H / ph) * ph, math.ceil(W / pw) * pw
#         x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
#         x_patches = x.unfold(2, ph, ph).unfold(3, pw, pw)
#         x_patches = x_patches.contiguous().view(B, C, -1, ph, pw).permute(0, 2, 1, 3, 4)
#         x_patches = x_patches.flatten(3)
#         return x_patches.reshape(B, -1, C * ph * pw)

#     def forward(self, x):
#         y = self.local_rep(x)
#         z = self.unfold(x)
#         z = self.transformer(z)
#         z = z.mean(dim=1).view(x.size(0), -1, 1, 1).expand_as(x)
#         return self.fusion(torch.cat((x, z), dim=1))

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, kernel_size, patch_size):
        super().__init__()
        self.local_rep = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim),
            nn.Conv2d(dim, dim, 1)
        )

        self.patch_size = patch_size
        ph, pw = patch_size
        self.linear_in = nn.Linear(dim * ph * pw, dim)  # Project into dim
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=dim * 2, batch_first=True),
            num_layers=depth
        )
        self.linear_out = nn.Linear(dim, dim * ph * pw)  # Project back
        self.fusion = nn.Conv2d(dim * 2, dim, 1)

    def unfold(self, x):
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        new_H, new_W = math.ceil(H / ph) * ph, math.ceil(W / pw) * pw
        x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
        x_patches = x.unfold(2, ph, ph).unfold(3, pw, pw)
        x_patches = x_patches.contiguous().view(B, C, -1, ph, pw).permute(0, 2, 1, 3, 4)
        x_patches = x_patches.flatten(3)
        return x_patches.reshape(B, -1, C * ph * pw), (new_H, new_W)

    def fold(self, patches, size, C):
        B, N, _ = patches.shape
        ph, pw = self.patch_size
        H, W = size
        H_patches, W_patches = H // ph, W // pw

        patches = patches.view(B, H_patches, W_patches, C, ph, pw)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        return patches.view(B, C, H, W)

    def forward(self, x):
        y = self.local_rep(x)
        z, (H, W) = self.unfold(x)
        z = self.linear_in(z)        # Project to d_model
        z = self.transformer(z)
        z = self.linear_out(z)       # Project back
        z = self.fold(z, (H, W), x.size(1))
        return self.fusion(torch.cat((x, z), dim=1))


class MobileViT(nn.Module):
    def __init__(self, dims, depths):
        super().__init__()
        self.stage1 = ConvBNAct(3, dims[0], 3, 2)
        self.stage2 = ConvBNAct(dims[0], dims[1], 3, 2)
        self.stage3 = MobileViTBlock(dims[1], depths[0], kernel_size=3, patch_size=(2, 2))
        self.stage4 = MobileViTBlock(dims[2], depths[1], kernel_size=3, patch_size=(2, 2))
        self.stage5 = MobileViTBlock(dims[3], depths[2], kernel_size=3, patch_size=(2, 2))
        self.conv_proj = nn.Conv2d(dims[1], dims[2], 1)
        self.conv_proj2 = nn.Conv2d(dims[2], dims[3], 1)

        self.dims = dims

    def forward(self, x):
        outputs = {}
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        outputs["res2"] = x
        x = self.conv_proj(x)
        x = self.stage4(x)
        outputs["res3"] = x
        x = self.conv_proj2(x)
        x = self.stage5(x)
        outputs["res4"] = x
        return outputs

class D2MobileViT(MobileViT, Backbone):
    def __init__(self, cfg, input_shape):
        dims = cfg['BACKBONE']['MOBILEVIT']['DIMS']
        depths = cfg['BACKBONE']['MOBILEVIT']['DEPTHS']

        super().__init__(dims, depths)

        self._out_features = cfg['BACKBONE']['MOBILEVIT']['OUT_FEATURES']
        self._out_feature_strides = {
            "res2": 8,
            "res3": 16,
            "res4": 32,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": dims[1],
            "res3": dims[2],
            "res4": dims[3],
            "res5": dims[3],
        }

    def forward(self, x):
        assert x.dim() == 4, f"MobileViT expects (N, C, H, W). Got {x.shape}"
        outputs = super().forward(x)
        outputs["res5"] = outputs["res4"]
        return {k: v for k, v in outputs.items() if k in self._out_features}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32

@register_backbone
def get_mobilevit_backbone(cfg):
    model = D2MobileViT(cfg['MODEL'], input_shape=None)

    return model
