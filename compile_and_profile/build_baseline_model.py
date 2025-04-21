"""
Build and run inference with the XDecoder baseline model for semantic segmentation.
This module handles model construction, weight loading, and inference pipeline.
"""

import os
import sys
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn 
from torchvision import transforms
from transformers import CLIPTokenizer
from detectron2.modeling import ShapeSpec
from detectron2.structures import ImageList


from modeling.vision.backbone import build_backbone
from modeling.vision.encoder import build_encoder
from modeling.language import build_language_encoder
from modeling.interface import build_decoder
from utils.arguments import load_opt_from_config_files

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_baseline_model(image_input, text_input, output_path="./compile_and_profile", test_torch_model_local=True):
    """
    Build the XDecoder baseline model and optionally run inference.
    
    Args:
        image_input: Input image tensor (1, 3, 1024, 1024) with values in [0, 255]
        text_input: Text embedding tensor (2, 1, 77) containing [text_emb, text_attn_mask]
        output_path: Directory to save output visualizations
        test_torch_model_local: Whether to run inference test after building
    
    Returns:
        XDecoder model instance
    """
    # load configs and pretrained weights
    conf_file = "./configs/xdecoder/mobile_vit_lang.yaml"
    opt = load_opt_from_config_files([conf_file])

    pretrained_path = opt['RESUME_FROM']
    ckpt = torch.load(pretrained_path)

    # build backbone
    backbone = build_backbone(opt).to(device)
    
    # 載入權重
    if 'module' in ckpt:
        ckpt = ckpt['module']
        
    # 移除 'module.' 前綴
    ckpt = {key.replace('module.',''):ckpt[key] for key in ckpt.keys()}

    # 載入 backbone 權重
    backbone.load_state_dict({k.replace('backbone.', ''): v for k,v in ckpt.items() if 'backbone' in k}, strict=False)
    
    # build multi-scale feature extractor
    backbone_out_feats = [96, 192, 384, 768]
    backbone_out_strides = [4, 8, 16, 32]
    input_shape = {'res2': ShapeSpec(channels=backbone_out_feats[0], stride=backbone_out_strides[0]), 
                   'res3': ShapeSpec(channels=backbone_out_feats[1], stride=backbone_out_strides[1]), 
                   'res4': ShapeSpec(channels=backbone_out_feats[2], stride=backbone_out_strides[2]), 
                   'res5': ShapeSpec(channels=backbone_out_feats[3], stride=backbone_out_strides[3])}
    
    multi_scale_feature_extractor = build_encoder(opt, input_shape).to(device)
    
    # 載入 multi_scale_feature_extractor 權重
    pixel_decoder_weights = {k.replace('sem_seg_head.pixel_decoder.', ''): v 
                            for k,v in ckpt.items() 
                            if 'sem_seg_head.pixel_decoder' in k}
    multi_scale_feature_extractor.load_state_dict(pixel_decoder_weights, strict=False)
    
    # build lang_encoder
    lang_encoder = build_language_encoder(opt).to(device)
    
    # 載入 lang_encoder 權重
    lang_encoder.load_state_dict({k.replace('lang_encoder.', ''): v for k,v in ckpt.items() if 'lang_encoder' in k}, strict=False)

    with torch.no_grad():
        lang_encoder.eval()
        embeds = lang_encoder.get_text_embeddings(class_names=['background'], name='default', is_eval=True, prompt=True)
        # init text embeddings for pre-defined classes used for semantic-seg task; not used for ov-grounding actually.

    # build mask_decoder
    task_swtich = {'mask': True, 'caption': True, 'grounding': True, 'bbox': False, 'captioning': False, 'retrieval': False}
    extra = {'task_switch': task_swtich, 
             'predefined_categories_embeds': lang_encoder.default_text_embeddings.to(device), 
             'logit_scale': lang_encoder.logit_scale.to(device)
             }
    mask_decoder = build_decoder(opt, in_channels=512, lang_encoder=lang_encoder, mask_classification=True, extra=extra).to(device)
    mask_decoder.load_state_dict({k.replace('sem_seg_head.predictor.', ''): v for k,v in ckpt.items() if (('sem_seg_head.predictor' in k) and ('sem_seg_head.predictor.lang_encoder' not in k))}, strict=False)


    model = XDecoder(opt, backbone, multi_scale_feature_extractor, lang_encoder, mask_decoder)


    if test_torch_model_local:
        inference_torch_local(model, image_input, text_input, output_path)

    return model

def inference_torch_local(model, image_input, text_input, output_path='./compile_and_profile'):
    """
    Run inference with the torch model and save visualization.
    
    Args:
        model: XDecoder model instance
        image_input: Input image tensor (1, 3, 1024, 1024) with values in [0, 255]
        text_input: Text embedding tensor (2, 1, 77) containing [text_emb, text_attn_mask]
        output_path: Directory to save output mask visualization
        
    Returns:
        pred_mask: Binary prediction mask (1024, 1024)
    """
    os.makedirs(output_path, exist_ok=True)

    with torch.no_grad():
        pred_mask = model(image_input, text_input)

    pred_mask = pred_mask.detach().cpu().numpy() if device == 'cuda' else pred_mask.numpy()
    
    plt.matshow(pred_mask)
    plt.axis('off')
    plt.savefig(f'{output_path}/pred_mask_torch_local.jpg')


class XDecoder(nn.Module):
    """
    XDecoder model for visual-language tasks including semantic segmentation.
    
    The model consists of:
    - Vision backbone for feature extraction
    - Multi-scale feature extractor
    - Language encoder for text embedding
    - Mask decoder for segmentation prediction
    """
    
    def __init__(self, opt, backbone, feature_extractor, lang_encoder, mask_decoder):
        """
        Initialize XDecoder model components.
        
        Args:
            opt: Configuration options dictionary
            backbone: Vision backbone network
            feature_extractor: Multi-scale feature extractor
            lang_encoder: Language encoder for text embeddings
            mask_decoder: Decoder for mask prediction
        """
        super().__init__()
        self.opt = opt
        self.backbone = backbone
        self.feature_extractor = feature_extractor
        self.lang_encoder = lang_encoder
        self.mask_decoder = mask_decoder

        self.image_resolution = 1024
        self.temperature = lang_encoder.logit_scale
        self.num_queries = 101
        
        # NOTE: if try to apply normalization to the input image, have to ensure (mean & std) are repeated to the same size, otherwise QNN would output errors.
        self.pixel_mean = torch.tensor(self.opt['INPUT']['PIXEL_MEAN']).view(1, -1, 1, 1).repeat(1, 1, self.image_resolution, self.image_resolution).to(device)
        self.pixel_std = torch.tensor(self.opt['INPUT']['PIXEL_STD']).view(1, -1, 1, 1).repeat(1, 1, self.image_resolution, self.image_resolution).to(device)

    def pre_processing(self, image_input, text_input):
        """
        Preprocess inputs before model forward pass.
        
        Args:
            image_input: Raw input image tensor
            text_input: Raw input text tensor
            
        Returns:
            Tuple of processed (images, tokens)
        """
        # downsample image for faster inference
        down_sample_size = 256
        images = (image_input - self.pixel_mean) / self.pixel_std
        images = F.interpolate(images, size=down_sample_size, mode='bilinear', align_corners=False, antialias=False)

        tokens = {'input_ids': text_input[0], 'attention_mask': text_input[1]}
        return images, tokens
    
    def vl_similarity(self, image_feat, text_feat, temperature=1):
        """
        Calculate visual-language similarity scores.
        
        Args:
            image_feat: Image feature embeddings
            text_feat: Text feature embeddings
            temperature: Temperature parameter for scaling logits
            
        Returns:
            Similarity logits between image and text features
        """
        logits = torch.matmul(image_feat, text_feat.t())
        logits = temperature.exp().clamp(max=100) * logits
        return logits

    def post_processing(self, pred_gmasks, pred_gtexts, class_emb):
        """
        Post-process model outputs to generate final prediction mask.
        
        Args:
            pred_gmasks: Predicted grounding masks, shape=bs x nqueries x h x w
            pred_gtexts: Predicted text embeddings, shape=bs x nqueries x caption_dim(512)
            class_emb: Class embedding for matching, shape=n_text_prompt x class_emb_dim(512)
            
        Returns:
            Final binary prediction mask
        """
        pred_gmasks = pred_gmasks
        v_emb = pred_gtexts
        t_emb = class_emb

        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)            

        temperature = self.temperature
        out_prob = self.vl_similarity(v_emb, t_emb, temperature=temperature)
        
        matched_id = out_prob.max(1)[1] # choose the top-1 confident prediction
        mask_pred_results = pred_gmasks[:, matched_id,:,:] #torch.Size([1, 1, 64, 64])

        # upsampling to input image size, 1024
        image_shape = self.image_resolution # 1024
        mask_pred_results = F.interpolate(mask_pred_results, 
                                          size=image_shape, mode="bilinear", align_corners=False, 
                                          antialias=False)[0]
        mask_pred_results = mask_pred_results.squeeze() >= 0
        return mask_pred_results.float() # bool variables are not supported by QNN as output features


    def forward(self, image_input, text_input):
        """
        Forward pass of the XDecoder model.
        
        Args:
            image_input: Input image tensor (1, 3, 1024, 1024)
            text_input: Input text tensor (2, 1, 77)
            
        Returns:
            Binary prediction mask (1024, 1024)
        """
        images, tokens = self.pre_processing(image_input, text_input)

        visual_features = self.backbone(images) # return dict, {'res2', 'res3', 'res4', 'res5'}
        mask_features, _, multi_scale_features = self.feature_extractor(visual_features)

        text_embeddings = self.lang_encoder.get_text_token_embeddings(tokens, name='grounding', token=True, norm=False)
        grounding_tokens = text_embeddings['class_emb'].unsqueeze(1) # 1x1x512

        extra = {'grounding_tokens': grounding_tokens}
        output = self.mask_decoder(multi_scale_features, mask_features, mask=None, target_queries=None, target_vlp=None, 
                                   task='grounding_eval', extra=extra)
        
        pred_gmasks = output['pred_masks'][:, self.num_queries:]
        pred_gcaptions = output['pred_captions'][:, self.num_queries:]
        top1_mask_pred = self.post_processing(pred_gmasks, pred_gcaptions, text_embeddings['class_emb'][0])
        return top1_mask_pred


