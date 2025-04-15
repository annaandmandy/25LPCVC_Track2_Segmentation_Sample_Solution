module load miniconda
conda activate xdecoder

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export WANDB_KEY="a9007bbf2af8533987785004be9fae23e52af2f5"
export QUALCOMM_AI_HUB_KEY="44165a068eebd71635bbb8e9198756c50e392b02"

# CUDA_VISIBLE_DEVICES=0,1 mpirun -n 2 python entry.py train \
#             --conf_files configs/xdecoder/focalt_unicl_lang_finetune.yaml \
#             --overrides \
#             FP16 True \
#             PORT 36874 \
#             COCO.INPUT.IMAGE_SIZE 1024 \
#             MODEL.DECODER.HIDDEN_DIM 512 \
#             MODEL.ENCODER.CONVS_DIM 512 \
#             MODEL.ENCODER.MASK_DIM 512 \
#             MODEL.DECODER.CAPTIONING.ENABLED False \
#             MODEL.DECODER.RETRIEVAL.ENABLED False \
#             MODEL.DECODER.GROUNDING.ENABLED True \
#             MODEL.DECODER.CAPTIONING_WEIGHT 8 \
#             MODEL.DECODER.RETRIEVAL_WEIGHT 8 \
#             MODEL.DECODER.TOP_CAPTIONING_LAYERS 3 \
#             MODEL.DECODER.TOP_RETRIEVAL_LAYERS 3 \
#             MODEL.DECODER.TOP_GROUNDING_LAYERS 6 \
#             MODEL.DECODER.GROUNDING.TEXT_WEIGHT 2.0 \
#             MODEL.DECODER.GROUNDING.CLASS_WEIGHT 0.5 \
#             COCO.TEST.BATCH_SIZE_TOTAL 128 \
#             COCO.TRAIN.BATCH_SIZE_TOTAL 128 \
#             COCO.TRAIN.BATCH_SIZE_PER_GPU 64 \
#             VLP.TEST.BATCH_SIZE_TOTAL 16 \
#             VLP.TRAIN.BATCH_SIZE_TOTAL 16 \
#             VLP.TRAIN.BATCH_SIZE_PER_GPU 8 \
#             VLP.DATALOADER.NUM_WORKERS 16 \
#             ADE20K.TEST.BATCH_SIZE_TOTAL 2 \
#             REF.TEST.BATCH_SIZE_TOTAL 2 \
#             SOLVER.LR_MULTIPLIER.lang_encoder 0.1 \
#             WEIGHT True \
#             RESUME_FROM xdecoder_data/pretrained_weights/focalt_in21k_yfcc_gcc_xdecoder_unicl.pt


# python entry.py evaluate \
#             --conf_files configs/xdecoder/focalt_unicl_lang_finetune.yaml \
#             --overrides \
#             COCO.INPUT.IMAGE_SIZE 512 \
#             MODEL.DECODER.CAPTIONING.ENABLED False \
#             MODEL.DECODER.RETRIEVAL.ENABLED False \
#             MODEL.DECODER.GROUNDING.ENABLED True \
#             COCO.TEST.BATCH_SIZE_TOTAL 1 \
#             COCO.TRAIN.BATCH_SIZE_TOTAL 1 \
#             COCO.TRAIN.BATCH_SIZE_PER_GPU 1 \
#             VLP.TEST.BATCH_SIZE_TOTAL 1 \
#             VLP.TRAIN.BATCH_SIZE_TOTAL 1 \
#             VLP.TRAIN.BATCH_SIZE_PER_GPU 1 \
#             MODEL.DECODER.HIDDEN_DIM 512 \
#             MODEL.ENCODER.CONVS_DIM 512 \
#             MODEL.ENCODER.MASK_DIM 512 \
#             ADE20K.TEST.BATCH_SIZE_TOTAL 1 \
#             FP16 True \
#             WEIGHT True \
#             RESUME_FROM /pth/to/xdecoder_data/pretrained/focalt_in21k_yfcc_gcc_xdecoder_unicl.pt

# works with baseline model and FFN layer replacement
python entry.py train \
    --conf_files configs/xdecoder/focalt_unicl_lang_finetune.yaml \
    --overrides \
    FP16 True \
    PORT 36874 \
    SOLVER.MAX_NUM_EPOCHS 2 \
    COCO.INPUT.IMAGE_SIZE 512 \
    MODEL.DECODER.HIDDEN_DIM 512 \
    MODEL.ENCODER.CONVS_DIM 512 \
    MODEL.ENCODER.MASK_DIM 512 \
    MODEL.DECODER.CAPTIONING.ENABLED False \
    MODEL.DECODER.RETRIEVAL.ENABLED False \
    MODEL.DECODER.GROUNDING.ENABLED True \
    MODEL.DECODER.CAPTIONING_WEIGHT 4 \
    MODEL.DECODER.RETRIEVAL_WEIGHT 4 \
    MODEL.DECODER.TOP_CAPTIONING_LAYERS 2 \
    MODEL.DECODER.TOP_RETRIEVAL_LAYERS 2 \
    MODEL.DECODER.TOP_GROUNDING_LAYERS 3 \
    MODEL.DECODER.GROUNDING.TEXT_WEIGHT 1.0 \
    MODEL.DECODER.GROUNDING.CLASS_WEIGHT 0.5 \
    COCO.TEST.BATCH_SIZE_TOTAL 8 \
    COCO.TRAIN.BATCH_SIZE_TOTAL 8 \
    COCO.TRAIN.BATCH_SIZE_PER_GPU 8 \
    VLP.TEST.BATCH_SIZE_TOTAL 8 \
    VLP.TRAIN.BATCH_SIZE_TOTAL 8 \
    VLP.TRAIN.BATCH_SIZE_PER_GPU 8 \
    VLP.DATALOADER.NUM_WORKERS 4 \
    ADE20K.TEST.BATCH_SIZE_TOTAL 1 \
    REF.TEST.BATCH_SIZE_TOTAL 1 \
    SOLVER.LR_MULTIPLIER.lang_encoder 0.1 \
    WEIGHT True \
    RESUME_FROM xdecoder_data/pretrained_weights/focalt_in21k_yfcc_gcc_xdecoder_unicl.pt

# python entry.py train \
#     --conf_files configs/xdecoder/mobile_vit_lang.yaml \
#     --overrides \
#     FP16 True \
#     PORT 36874 \
#     SOLVER.MAX_NUM_EPOCHS 2 \
#     COCO.INPUT.IMAGE_SIZE 512 \
#     MODEL.DECODER.HIDDEN_DIM 512 \
#     MODEL.ENCODER.CONVS_DIM 512 \
#     MODEL.ENCODER.MASK_DIM 512 \
#     MODEL.DECODER.CAPTIONING.ENABLED False \
#     MODEL.DECODER.RETRIEVAL.ENABLED False \
#     MODEL.DECODER.GROUNDING.ENABLED True \
#     MODEL.DECODER.CAPTIONING_WEIGHT 4 \
#     MODEL.DECODER.RETRIEVAL_WEIGHT 4 \
#     MODEL.DECODER.TOP_CAPTIONING_LAYERS 2 \
#     MODEL.DECODER.TOP_RETRIEVAL_LAYERS 2 \
#     MODEL.DECODER.TOP_GROUNDING_LAYERS 3 \
#     MODEL.DECODER.GROUNDING.TEXT_WEIGHT 1.0 \
#     MODEL.DECODER.GROUNDING.CLASS_WEIGHT 0.5 \
#     COCO.TEST.BATCH_SIZE_TOTAL 8 \
#     COCO.TRAIN.BATCH_SIZE_TOTAL 8 \
#     COCO.TRAIN.BATCH_SIZE_PER_GPU 8 \
#     VLP.TEST.BATCH_SIZE_TOTAL 8 \
#     VLP.TRAIN.BATCH_SIZE_TOTAL 8 \
#     VLP.TRAIN.BATCH_SIZE_PER_GPU 8 \
#     VLP.DATALOADER.NUM_WORKERS 4 \
#     ADE20K.TEST.BATCH_SIZE_TOTAL 1 \
#     REF.TEST.BATCH_SIZE_TOTAL 1 \
#     SOLVER.LR_MULTIPLIER.lang_encoder 0.1 \
#     MODEL.BACKBONE.PRETRAINED False \
#     WEIGHT True \
#     RESUME_FROM None 
