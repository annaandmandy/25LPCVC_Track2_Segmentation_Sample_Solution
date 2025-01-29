import importlib
import torch
import os
import numpy
from torch.nn import functional as F
from transformers import CLIPTokenizer
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import ImageList

from utils.arguments import load_opt_from_config_files
from utils.model import align_and_update_state_dicts
from modeling.modules import sem_seg_postprocess
from modeling.BaseModel import BaseModel
from modeling import build_model
from modeling.vision.backbone import build_backbone, Backbone
from modeling.body import build_xdecoder_head
# from .modules import sem_seg_postprocess, SetCriterion, HungarianMatcher, bbox_postprocess
from modeling.language import build_language_encoder
from modeling.language.loss import vl_similarity
import qai_hub

# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()

def data_preprocess(img_path, text, max_token_num = 77):
    # Image original size -> resize_max_leng=1024 & padding to 1024x1024
    image = utils.read_image(img_path, format='RGB')
    transform = []
    transform.extend([T.ResizeShortestEdge(1024, max_size=1024),])    
    image, _ = T.apply_transform_gens(transform, image)
    pad_image = numpy.zeros((1024, 1024, 3), numpy.uint8)
    pad_image[:image.shape[0], :image.shape[1]] = image
    pad_image = torch.as_tensor(numpy.ascontiguousarray(pad_image.transpose(2, 0, 1))).cuda()

    # Build and apply CLIP Tokenizer
    tokenizer = None
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    pretrained_tokenizer = 'openai/clip-vit-base-patch32'
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)

    tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})

    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=max_token_num, return_tensors='pt')
    text_emb = tokens['input_ids'].cuda()
    text_attn_mask = tokens['attention_mask'].cuda()

    return torch.unsqueeze(pad_image, 0), text_emb, text_attn_mask

# Pretrained X-Decoder model
class XDecoder(torch.nn.Module):
    def __init__(self, cfg, pretrained_path):
        super(XDecoder, self).__init__()
        # Prepare config file and build model
        # Switcher for task {'bbox': False, 'mask': True, 'caption': True, 'captioning': False, 'retrieval': False, 'grounding': True}
        task_switch = {'bbox': False,
                       'mask': True,
                       'caption': True,
                       'captioning': False,
                       'retrieval': False,
                       'grounding': True}

        # build model
        extra = {'task_switch': task_switch}
        self.cfg = cfg
        self.backbone = build_backbone(self.cfg).cuda()
        self.lang_encoder = build_language_encoder(self.cfg).cuda()        
        self.sem_seg_head = build_xdecoder_head(self.cfg, self.backbone.output_shape(), self.lang_encoder, extra).cuda()
        self.sem_seg_head.predictor.lang_encoder.get_text_embeddings(['background'], is_eval=True)

        # Load pretrained weights
        checkpoint = torch.load(pretrained_path)

        # Load pretrained weights of backbone and semantic segmentation head
        state_dict_backbone = self.backbone.state_dict()
        for key in state_dict_backbone:
            state_dict_backbone[key] = checkpoint['backbone.' + key]
        self.backbone.load_state_dict(state_dict_backbone)

        state_dict_head = self.sem_seg_head.state_dict()
        for key in state_dict_head:
            state_dict_head[key] = checkpoint['sem_seg_head.' + key]
        self.sem_seg_head.load_state_dict(state_dict_head)

        print('Successfull loaded from {}'.format(pretrained_path))

        # Define preprocessing steps
        self.pixel_mean = torch.tensor([[[[123.675]], [[116.280]], [[103.530]]]]).cuda()
        self.pixel_std = torch.tensor([[[[58.395]], [[57.120]], [[57.375]]]]).cuda()
        self.size_divisibility = 32
        self.num_queries = 101
    
    def postprocess(self, outputs, class_emb):
        pred_gmasks = outputs['pred_masks'][0,self.num_queries:2*self.num_queries-1]
        v_emb = outputs['pred_captions'][0,self.num_queries:2*self.num_queries-1]
        t_emb = class_emb

        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)            

        temperature = self.sem_seg_head.predictor.lang_encoder.logit_scale
        out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
        
        matched_id = out_prob.max(0)[1]
        mask_pred_results = pred_gmasks[matched_id,:,:] #list: torch.Size([1, 128, 160])

        mask_pred_results = F.interpolate(mask_pred_results.unsqueeze(0), size=1024, mode="bilinear", align_corners=False, antialias=False).squeeze(0)

        return mask_pred_results

    def forward(self, image, text_input):
        # Image: resize 1024 -> 256, normalization
        # Text_emb: 1x77, text_attn_mask: 1x77
        image = F.interpolate(image, size=256, mode="bilinear", align_corners=False, antialias=False)

        img = (image - self.pixel_mean) / self.pixel_std
        # img = ImageList.from_tensors([img], self.size_divisibility)

        visual_features = self.backbone(img)
        tokens = {}
        text_emb = text_input[0]
        text_attn_mask = text_input[1]
        tokens['input_ids'] = text_emb
        tokens['attention_mask'] = text_attn_mask
        extra = {}
        txt_embedding = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(tokens, name='grounding', token=True, norm=False)
        extra['grounding_tokens'] = txt_embedding['class_emb'].unsqueeze(1)

        outputs = self.sem_seg_head(visual_features, extra=extra, task='grounding_eval')

        pred_mask = self.postprocess(outputs, txt_embedding['class_emb'])

        return pred_mask


if __name__ == "__main__":
    img_path = ''
    text = 'person'
    cfg_path = ['configs/xdecoder/focalt_unicl_lang_lpcvc25.yaml']
    pretrained_path = 'model_state_dict.pt'

    # image, text_emb, text_attn_mask = data_preprocess(img_path, text)
    cfg = load_opt_from_config_files(cfg_path)

    # Create the model
    model = XDecoder(cfg, pretrained_path)

    # Model example input
    img_tensor = torch.randn(1, 3, 1024, 1024).cuda()
    text_tensor = torch.randint(low=0, high=1000, size=(1, 77), dtype=torch.int64).cuda()
    txtmask_tensor = torch.randint(low=0, high=2, size=(1, 77), dtype=torch.int64).cuda()
    text_input = torch.cat([text_tensor, txtmask_tensor], dim = 0)

    example_input = (img_tensor, text_input)
    # example_input = (img_tensor, text_tensor, txtmask_tensor)
    # output = model(*example_input)

    with torch.no_grad():
        model.eval()
        torch.onnx.export(model, example_input, f"./xdecoder_lpcvc25.onnx", input_names=["image", "text_emb", "text_attn_mask"], output_names=["output"])

    # Compile model on a specific device
    compile_job = qai_hub.submit_compile_job(
        model=f"./xdecoder_lpcvc25.onnx",
        name="xdecoder_ovss",
        device=qai_hub.Device("Snapdragon X Elite CRD"),
        # by default, model is compiled to TFLite, and no device limits
    )
    compiled_model = compile_job.get_target_model().download(f"./xdecoder_lpcvc25.bin")

    image, text_emb, text_attn_mask = data_preprocess(img_path, text)
    input_array = (image, text_emb, text_attn_mask)

    # """Submit an inference job for the model."""
    inference_job = qai_hub.submit_inference_job(
        model=compiled_model,
        device=qai_hub.Device("Snapdragon X Elite CRD"),
        inputs=input_array,
        options="--max_profiler_iterations 1"
    )
    output_array = inference_job.download_output_data()
