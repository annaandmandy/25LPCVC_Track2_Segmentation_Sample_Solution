import torch
from timm.models.efficientformer import efficientformer_l1

def get_efficientformer_backbone(cfg):
    model = efficientformer_l1(pretrained=True)  # Load EfficientFormer-L1
    return model