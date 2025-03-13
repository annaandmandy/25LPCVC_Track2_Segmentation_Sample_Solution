# LPCVC 2025 Track 2 Baseline Instructions

:point_right: This document provides instructions for setting up and running the baseline model for LPCVC 2025 Track 2 - Open Vocabulary Segmentation with Text Prompts. The baseline is based on [X-Decoder](https://github.com/microsoft/X-Decoder).

## 1. Environment Setup

### Create and Configure Conda Environment
```sh
# Create and activate conda environment
conda create -n xdecoder python=3.11 anaconda
conda activate xdecoder
conda install pip
# [Different from XDecoder env setup, NO Mask2Former needed as there is no deformableAttention in the baseline model.

# Install required packages
pip install --upgrade -r ./LPCVC2025_BASELINE_REQUIREMENTS.txt # compared to original XDecoder requirements list, some libs are required older version, e.g., pillow==9.4.0, nltk==3.8.1, etc.

# Install additional tools

pip install git+https://github.com/MaureenZOU/detectron2-xyz.git # this is a modified version of detectron2 with some extra functions. 
# pip install git+https://github.com/facebookresearch/detectron2 # If you prefer to use official detectron2 lib, some of the XDecoder dataloader operations need to be replaced, e.g., len(getattr(self, self.key_dataset)) in datasets/build.py

pip install git+https://github.com/cocodataset/panopticapi.git # for coco dataset preparation

# Install MPI for multi-processing training (refer to: https://pypi.org/project/mpi4py/)
# sudo apt-get install libopenmpi-dev # install this if you don't have MPI environment on your system

conda install -c conda-forge mpi4py # noticed pip install mpi4py errors, then try conda install.
```

Setup Evaluation Tools
```sh
# captioning_evaluation tools (NOT needed for LPCVC text-promt segment task, but will cause XDecoder error if you don't have it.)
# Download and setup captioning evaluation tools
cd /PATH/TO/PROJECT/ROOT # Replace /PATH/TO/PROJECT/ROOT with your actual project path
mkdir /PATH/TO/PROJECT/ROOT/xdecoder_data && cd /PATH/TO/PROJECT/ROOT/xdecoder_data
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/coco_caption.zip
unzip coco_caption.zip
```

### Set Environment Variables
```sh
# Replace /PATH/TO/PROJECT/ROOT with your actual project path
DTAROOT=/PATH/TO/PROJECT/ROOT/xdecoder_data
export DETECTRON2_DATASETS=$DTAROOT
export DATASET=$DTAROOT
export DATASET2=$DTAROOT
export VLDATASET=$DTAROOT
export PATH=$PATH:$DTAROOT/coco_caption/jre1.8.0_321/bin
export PYTHONPATH=$PYTHONPATH:$DTAROOT/coco_caption
```

## 2. Dataset Preparation

### Download and Extract COCO Dataset
```sh
cd /PATH/TO/PROJECT/ROOT/xdecoder_data
mkdir coco && cd coco

# Download COCO 2017 dataset
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip

# Extract datasets
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
unzip panoptic_annotations_trainval2017.zip
unzip ./annotations/panoptic_train2017.zip -d ./
unzip ./annotations/panoptic_val2017.zip -d ./
```

### Download Additional Annotations
```sh
cd annotations
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/caption_class_similarity.pth
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/captions_train2017_filtrefgumdval_filtvlp.json
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/grounding_train2017_filtrefgumdval_filtvlp.json
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/panoptic_train2017_filtrefgumdval_filtvlp.json
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/refcocog_umd_val.json
wget https://github.com/peteanderson80/coco-caption/blob/master/annotations/captions_val2014.json

# Download LVIS annotations
cd /PATH/TO/PROJECT/ROOT/xdecoder_data
mkdir lvis && cd lvis
wget https://huggingface.co/xdecoder/SEEM/resolve/main/coco_train2017_filtrefgumdval_lvis.json
```

### Prepare Semantic Annotations from Panoptic Annotations
```sh
cd /PATH/TO/PROJECT/ROOT/xdecoder_data
# ref: [OneFormer](https://github.com/SHI-Labs/OneFormer/blob/main/datasets/README.md)
export DETECTRON2_DATASETS=/PATH/TO/PROJECT/ROOT/xdecoder_data
python prepare_coco_semantic_annos_from_panoptic_annos.py
```

### Evaluation Dataset - RefCOCOg
Following X-Decoder, refcocog ('refcocog_umd_val.json') is a good dataset used to evaluate the model for ov-segment with 
text-prompt.

### Expected Dataset Structure
```
.xdecoder_data/
├── coco/
│   ├── train2017/                    # Training images
│   ├── val2017/                      # Validation images
│   ├── panoptic_train2017/          # Panoptic training annotations
│   ├── panoptic_semseg_train2017/   # Generated semantic annotations
│   ├── panoptic_val2017/            # Panoptic validation annotations
│   ├── panoptic_semseg_val2017/     # Generated semantic annotations
│   └── annotations/                  # Various annotation files
|       ├── refcocog_umd_val.json # used for ov-seg evaluation
|       ├── captions_val2014.json
|       ├── caption_class_similarity.pth
|       ├── panoptic_train2017_filtrefgumdval_filtvlp.json
|       └── grounding_train2017_filtrefgumdval_filtvlp.json # used for ov-seg training
|       └── instances_{train,val}2017.json
|       └── panoptic_{train,val}2017.json
└── lvis/
    └── coco_train2017_filtrefgumdval_lvis.json
```


## 3. Model Setup

### Download Pre-trained Weights
```sh
mkdir ./pretrained_weights && cd ./pretrained_weights

# Download required model weights:
# 1. ImageNet21k initialized X-Decoder
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/focalt_in21k_yfcc_gcc_xdecoder_unicl.pt

# 2. X-Decoder pre-trained (Focal-T)
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/xdecoder_focalt_last.pt

# 3. LPCVC pre-trained baseline
# Download from Google Drive: https://drive.google.com/file/d/1pk1HVDvQuGEyGwB4fP6y35mLWqY5xqOq
```

**Note**: The LPCVC baseline model has modified layers compared to original X-Decoder, so loading all pretrained X-Decoder weights may not be possible.

## 4. Training

Run the training script:
```sh
sh command.sh
```

## Tips for Improvement

1. **Resolution vs Performance**: Higher resolution typically improves performance, but model finetuning often yields better results.

2. **Dataset Coverage**: The baseline model is trained on refCOCO, but test data includes more categories. Consider using additional datasets for training. Original X-Decoder already used a combination of different datasets, consider to involve them.

3. **Sample Data Usage**: The provided sample data are from the same source and annotated in the same strategy as the final test data. Thus consider including the sample data in post-training/finetuning for better performance.

4. **Alternative Architectures**: Consider trying other OV-Seg architectures like SEEM, OVSeg, CAT-Seg, EVF-SAM, Grounding SAM and efficient variants, etc. DO NOT limit yourself to the baseline!

5. **Architecture Optimization**: Consider trading resolution for more powerful architectures like ViT or hybrid models. Although complext architectures have higher latency, but accuracy v.s. efficiency is a trade-off. How about reduce the resolution but use advanced networks?

6. **Task-Specific Optimization**: The baseline (X-Decoder) is design to manage all general semantic and instance segmentation, visual-promtp segment anything, and text-promt open-vocabulary segmentation, etc. Focus on essential components for the LPCVC task rather than maintaining all X-Decoder capabilities.

### QNN Compatibility Notes
The following operations are **not** supported by QNN:
- GroupNorm
- DeformableAttention

Consider using QNN-supported alternatives (e.g., BatchNorm instead of GroupNorm).

## FAQ

### Environment Setup
- **Q: Is Mask2Former required?**  
  A: No, as DeformableAttention is not used in the baseline model.

- **Q: MPI installation issues?**  
  A: Use conda-forge channel for mpi4py installation instead of pip. Make sure you install `libopenmpi-dev`, and we noticed that using `conda install -c conda-forge mpi4py` is easier than using `pip` to install `mpi4py`, and sometimes `openmpi` is also needed to install. Please refer to the mpi4py official doc and the pip page (https://pypi.org/project/mpi4py/) and discussions on Reddit to look for similar 
  questions.

- **Q: Missing dataset files?**  
  A: Please follow the steps provided in this instructions or the original X-Decoder repo to download and put all 
  unzip data into the correct path. We know some data are not designed for the LPCVC 2025 Track2 task, but simply put 
  them there can resolve some bugs and let you start the training early and easily. MOREOVER, we do notice that the 
  XDecoder team put some pre-processed data files on their official Github repo (https://github.com/microsoft/X-Decoder/
  tree/v2.0) or HuggingFace repo (https://huggingface.co/xdecoder/X-Decoder/tree/main). If you find something is hard to 
  find on public sources, please go to check their Huggingface page and you probably will find them there.

-  **Q: Python packages installation errors?**  
  A: Errors we noticed:
    - AttributeError: module 'onnx' has no attribute 'load_from_string' 
      - [**Solution**] upgrade onnx, `pip install --upgrade onnx`
    - opencv: Could not load the Qt platform plugin "xcb" in "" even though it was found 
      - [**Solution**] uninstall pyqt5, `pip uninstall PyQt5`, may need to re-install opencv-python
    - The provided dependency requirements list is different from XDecoder? 
      - [**Answer**] Yes. Some libraries referred by XDecoder is out-dated and some operations are unsupported. Thus we figured out an alternative list, with most libs up-to-date, and some specific ones are installed the older version to match the operations in the code (e.g., pillow==9.4.0, nltk==3.8.1, etc.). Moreover, since XDecoder uses another modified Detectron2 lib, thus to use that one, these older version libs are required. If you prefer to use your own env or latest version of libs, some operations of the provided sample solution code base need to be replaced.
    


### Model Architecture
- **Q: How does the baseline differ from original X-Decoder?**  
  A: we did some mofifications to make it compatible by QNN libraries. Changes include:
  - Fixed-size [cls] embedding (1x512) instead of dynamic CLIP text embeddings
  - Replaced GroupNorm with QNN-compatible alternatives
  - and more.

### Latest Updates
- **Q: What's new in the latest baseline updates?**  
  A: The model now focuses specifically on text-prompt segmentation, removing general semantic segmentation outputs for better performance.
  In the latest updates (2025.03.11), we modified the baseline model following XDecoder evaluation strategy ONLY for open-vocabulary segmentation with text-prompt task (grounding segmentation) with text-prompt only. For details, the masks predictions generated by the queries for general semantic segmentation task are removed and only the queries fused both image and text knowledge are used to output the mask prediction. We noticed such strategy gives much better performance for the text-prompt segmentation task. Specifically, in `compile_and_profile/build_baseline_model. py` file, `XDecoder` class `forward()` function, the following lines are made:
    ```python
    # self.num_queries = 101
    pred_gmasks = output['pred_masks'][:, self.num_queries:]
    pred_gcaptions = output['pred_captions'][:, self.num_queries:]
    top1_mask_pred = self.post_processing(pred_gmasks, pred_gcaptions, text_embeddings['class_emb'][0])
    return top1_mask_pred
    ```
    where `self.num_queries=101` is the number of learnable queries in XDecoder, by default it's `101`. In the output mask prediction, the first 101 masks are generated by the queries learned for general segmentation tasks (e.g., semnatic and panoptic), the following masks are generated based on the fused image and text knowledge. 
  

- **Q: Must I use the baseline model?**  
  A: No, you can design your own solution. The key requirement is compatibility with the evaluation pipeline in `compile_and_profile/compile_profile_inference_aihub.py`.
