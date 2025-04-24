# DS542 Final Project
Project cloned from IEEE LPCVC - open vocabulary model

## Model Training and Compiling Instruction
### Train and evaluate
Use command.sh to train and evaluate the model
1. Change python entry.py train or evaluate to switch mode
2. Lower Batch size to prevent CUDA memory overflow
3. Change config file to the model that were been used
    - set Resume From to pretrained weight
### Compile
Use compile_AIHUB.sh to compile
- This script uses compile_profile_inference_aihub.py and build_baseline_model.py to build model, compile and sent it to Qualcomm AI Hub to test inference time on edge device(Snapdragon X Elite CRD).


