
## Baseline Intruction (compare)
:point_right: ***Please refer to [[XDecoder]](https://github.com/microsoft/X-Decoder) for model training and evaluation details.***

### 1. Environment Setup
```sh
# create conda virtual env
conda create -n xdecoder python=3.11 anaconda
conda activate xdecoder
conda install pip
# install required packages (XDecoder), [NO Mask2Former needed as there is no deformableAttention in the baseline model. ignored libs: mpi4py # difficult to install due to conflicts of versions ]
pip install -r ./LPCVC2025_BASELINE_REQUIREMENTS.txt

# install mpi4py and libopenmpi-dev for multi-procesing training (refer to: https://pypi.org/project/mpi4py/)
# NOTE: installing mpi and mpi4py may have some problems, please refer to the mpi4py and related online resources for details
sudo apt-get install libopenmpi-dev # for `mpirun` or `mpiexec`
conda install -c conda-forge mpi4py openmpi # for `from mpi4py import MPI

# isntall custom tools
pip install git+https://github.com/facebookresearch/detectron2 # (https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
pip install git+https://github.com/arogozhnikov/einops.git
pip install git+https://github.com/cocodataset/panopticapi.git # for coco dataset preparation

# other setup (quoted from XDecoder)
# captioning_evaluation tools (NOT needed for LPCVC text-promt segment task, but will cause XDecoder code error if you don't have it.)
# save coco_caption.zip to .xdecoder_data
cd /PATH/TO/DATASET/ROOT
mkdir .xdecoder_data
cd .xdecoder_data
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/coco_caption.zip
unzip coco_caption.zip

export DETECTRON2_DATASETS=/PATH/TO/DATASET/ROOT/xdecoder_data # /pth/to/xdecoder_data
export DATASET=/PATH/TO/DATASET/ROOT/xdecoder_data # /pth/to/xdecoder_data
export DATASET2=/PATH/TO/DATASET/ROOT/xdecoder_data # /pth/to/xdecoder_data
export VLDATASET=/PATH/TO/DATASET/ROOT/xdecoder_data # /pth/to/xdecoder_data
export PATH=$PATH:/PATH/TO/DATASET/ROOT/xdecoder_data/coco_caption/jre1.8.0_321/bin
export PYTHONPATH=$PYTHONPATH:/PATH/TO/DATASET/ROOT/xdecoder_data/coco_caption


# for example
# export DETECTRON2_DATASETS=/home/scott/Work/Work/Qualcomm/LPCVC_2025_Track2_OVSeg/xdecoder_data
# export DATASET=/home/scott/Work/Work/Qualcomm/LPCVC_2025_Track2_OVSeg/xdecoder_data
# export DATASET2=/home/scott/Work/Work/Qualcomm/LPCVC_2025_Track2_OVSeg/xdecoder_data
# export VLDATASET=/home/scott/Work/Work/Qualcomm/LPCVC_2025_Track2_OVSeg/xdecoder_data
# export PATH=$PATH:/home/scott/Work/Work/Qualcomm/LPCVC_2025_Track2_OVSeg/xdecoder_data/coco_caption/jre1.8.0_321/bin
# export PYTHONPATH=$PYTHONPATH:/home/scott/Work/Work/Qualcomm/LPCVC_2025_Track2_OVSeg/xdecoder_data/coco_caption

```
<!-- **OR** Find ourdocker image [[here]](). (built based official `pytorch/pytorch` image.)
```sh
docker pull /path/to/docker-image
docker run --rm -it -gpu --all -v /local/code/path:/docker/code/path image-name:image-tag /bin/bash
conda activate lpcvc_baseline # 


``` -->


### 2. Dataset Preparation
```sh
# create the xdecoder_data folder
cd /PATH/TO/DATASET/ROOT/xdecoder_data
mkdir coco
cd coco
# Prepare panoptic_train2017, panoptic_semseg_train2017 exactly the same as [Mask2Fomer](https://github.com/facebookresearch/Mask2Former/tree/main/datasets)

# download coco 2017 dataset images: https://cocodataset.org/#download
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip # extract instance train/val json files to ./annotations
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
unzip panoptic_annotations_trainval2017.zip # extract panoptic train/val json files to ./annotations 
unzip ./annotations/panoptic_train2017.zip -d ./ # extract panoptic train/val png annotations to ./panoptic_train2017
unzip ./annotations/panoptic_val2017.zip -d ./ # extract panoptic train/val png annotations to ./panoptic_val2017



# (SEEM & X-Decoder) Download additional logistic and custom annotation files to .xdecoder_data/coco/annotations
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/caption_class_similarity.pth -P ./annotations/
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/captions_train2017_filtrefgumdval_filtvlp.json -P ./annotations/
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/grounding_train2017_filtrefgumdval_filtvlp.json -P ./annotations/
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/panoptic_train2017_filtrefgumdval_filtvlp.json -P ./annotations/
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/refcocog_umd_val.json
wget https://github.com/peteanderson80/coco-caption/blob/master/annotations/captions_val2014.json -P ./annotations/

# (SEEM) Download LVIS annotations for mask preparation
wget https://huggingface.co/xdecoder/SEEM/resolve/main/coco_train2017_filtrefgumdval_lvis.json -P ./annotations/

# prepare coco panotpic annotation by function from Mask2Former [https://github.com/facebookresearch/Mask2Former/blob/main/datasets/prepare_coco_semantic_annos_from_panoptic_annos.py]
export DETECTRON2_DATASETS=/PATH/TO/DATASET/ROOT/xdecoder_data # /path/to/datasets/root
# **You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`. If left unset, the default is ./datasets relative to your current working directory.**

python prepare_coco_semantic_annos_from_panoptic_annos.py 
# help ref: [OneFormer](https://github.com/SHI-Labs/OneFormer/blob/main/datasets/README.md)
```


After dataset preparation, the dataset structure would be:
```
.xdecoder_data
└── coco/
    ├── train2017/ # images
    ├── val2017/ # images
    ├── panoptic_train2017/ # png annotations
    ├── panoptic_semseg_train2017/ # generated by the script following Mask2Former as mentioned above
    ├── panoptic_val2017/ # png annotations
    ├── panoptic_semseg_val2017/ # generated by the script following Mask2Former as mentioned above
    └── annotations/
        ├── refcocog_umd_val.json # used for ov-seg evaluation
        ├── captions_val2014.json
        ├── caption_class_similarity.pth
        ├── panoptic_train2017_filtrefgumdval_filtvlp.json
        └── grounding_train2017_filtrefgumdval_filtvlp.json # used for ov-seg training
        └── instances_{train,val}2017.json
        └── panoptic_{train,val}2017.json
└── lvis/
    └── coco_train2017_filtrefgumdval_lvis.json
```

#### Evaluation Dataset - RefCOCO (SEEM & X-Decoder)
During training, refcoco ('refcocog_umd_val.json') is a good dataset used to evaluate the model for ov-segment with text-prompt (following X-Decoder)


### 3. Baseline Model Details
- Architecture: Focal-T / ViT-b
- Training data: COCO
- Evaluation data: RefCOCOg
- Task: OV-Seg with text-prompt (Grounding)
- Download pre-trained weights
```sh
# download pre-trained weights, put them in folder 
mkdir ./pretrained_weights

# (1) imagenet21k init X-Decoder
filename: focalt_in21k_yfcc_gcc_xdecoder_unicl.pt
link: [[HuggingFace]](https://huggingface.co/xdecoder/X-Decoder/resolve/main/focalt_in21k_yfcc_gcc_xdecoder_unicl.pt?download=true)

# (2) x-decoder pretrained (Focal-T)
filename: xdecoder_focalt_last.pt
link: [[HuggingFace]](https://huggingface.co/xdecoder/X-Decoder/resolve/main/xdecoder_focalt_last.pt)
# (3) LPCVC pre-trained 
filename: model_state_dict.pt
link: [[Google Drive]](https://drive.google.com/file/d/1pk1HVDvQuGEyGwB4fP6y35mLWqY5xqOq)

```
**NOTE**: LPCVC modified some layers and operations of XDecoder, so you probably won't be able to load all pretrained Xdecoder weights.



### 4. Getting Started

- Training command: `sh command.sh`


### :bulb: Tips
- (1) Usually higher resolution gives better performance with the same baseline model. But finetune the model with more data will give more significant improvements. Performance v.s. Efficiency is a forever art.
- (2) The baseline model is based on XDecoder, which is trained on refcoco dataset. However, the test data contains much more categories than coco, thus involving more training data and categories will also benefit the final test performance.
- (3) The public sample data are from the same source and annotated in the same strategy as the final test data. Thus involving them into the post-training/finetuning stage would also gain extra scores.
- (4) DO NOT limit yourself to the baseline! Open-vocabulary segmentation with text-prompt (OV grounding segmentation) has gained a lot of attention recently. Try different architectures and dataset to copmare their difference. (e.g., X-Decoder, SEEM, OVSeg, CAT-Seg, EVF-SAM, Grounding SAM and all efficient variants, etc.) Try them out!
- (5) The baseline is based on Focal-T image backbone, which is a convolutional based network, ViT or hybrid architectures are usually more powerful. However, complex architecture would increase the latency, but how about "reducing the resolution a little bit but leveraging a more potential framework?"
- (6) The baseline (X-Decoder) is design to manage all general semantic and instance segmentation, visual-promtp segment anything, and text-promt open-vocabulary segmentation, etc. However, only for LPCVC 2025 Track-2 challenge, are all of them necessary? Sometimes less could gain more!
- \* **Known QNN UN-supported Operations/Layers**:
  - [GroupNorm, DeformableAttention]
  - Consider replace them when design the framework before training. Classic and common layers are better optimized by QNN lib (e.g., BatchNorm :white_check_mark:, GroupNomr :x:). Check QNN documentation for supported operations.


### :interrobang: FAQ
Here we list some errors we encounterred or many people asked about training and evaluating the baseline (X-Decoder) model for the LPCVC 2025 Track2 task.

(1) Environment set-up issues:
  - Q: Mask2Former build from source? 
    - A: No. DeformableAttention is not supported by QNN yet thus not used in the baseline model. So no need to build Mask2Former from source to enable the training and evaluation. Building it may result some errors.
  - Q: issues installing mpi and mpi4py?
    - A: We also encountered some issues. Make sure you install `libopenmpi-dev`, and we noticed that using `conda install -c conda-forge mpi4py openmpi` is easier than using `pip` to install `mpi4py`. Please refer to the mpi4py official doc and the pip page (https://pypi.org/project/mpi4py/) and discussions on Reddit to look for similar questions.
  - Q: Dataset missing or related tools/packages missing? 
    - A: Please follow the steps provided in this instructions or the original X-Decoder repo to download and put all unzip data into the correct path. We know some data are not designed for the LPCVC 2025 Track2 task, but simply put them there can resolve some bugs and let you start the training early and easily. MOREOVER, we do notice that the XDecoder team put some pre-processed data files on their official Github repo (https://github.com/microsoft/X-Decoder/tree/v2.0) or HuggingFace repo (https://huggingface.co/xdecoder/X-Decoder/tree/main). If you find something is hard to find on public sources, please go to check their Huggingface page and you probably will find them there.
(2) Model architecture
  - Q: The provided baseline model is different from origial X-Decoder (Focal-T)? 
    - A: Yes, we did some mofifications to make it compatible by QNN libraries. For example, we replaced the dynamic-shape text embedding output from CLIP text encoder by the fixed size [cls] embedding, shape=1x512. Besides, all `GroupNorm()` are replaced as it's not supported by QNN yet. 
