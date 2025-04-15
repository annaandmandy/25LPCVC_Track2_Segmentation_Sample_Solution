import torch
import torch.nn as nn
from timm.models.efficientformer import efficientformer_l1, efficientformer_l3, efficientformer_l7
from detectron2.modeling.backbone import Backbone, BACKBONE_REGISTRY
from detectron2.modeling.backbone.build import ShapeSpec


class EfficientFormerBackbone(Backbone):
    def __init__(self, variant="l1", pretrained=True):
        super().__init__()
        if variant == "l1":
            self.model = efficientformer_l1(pretrained=pretrained, features_only=True)
        elif variant == "l3":
            self.model = efficientformer_l3(pretrained=pretrained, features_only=True)
        elif variant == "l7":
            self.model = efficientformer_l7(pretrained=pretrained, features_only=True)
        else:
            raise ValueError(f"EfficientFormer variant {variant} is not supported.")

        self._out_features = [f'stage{i+1}' for i in range(len(self.model.feature_info))]
        self._out_feature_channels = {
            name: self.model.feature_info[i]['num_chs']
            for i, name in enumerate(self._out_features)
        }
        self._out_feature_strides = {
            name: self.model.feature_info[i]['reduction']
            for i, name in enumerate(self._out_features)
        }

    def forward(self, x):
        feats = self.model(x)
        return {name: feat for name, feat in zip(self._out_features, feats)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_efficientformer_backbone(cfg, input_shape):
    variant = cfg.MODEL.BACKBONE.NAME.split("_")[-1]  # e.g., "l1"
    pretrained = cfg.MODEL.BACKBONE.LOAD_PRETRAINED
    return EfficientFormerBackbone(variant=variant, pretrained=pretrained)
