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
1. Change config file in build_baseline_model.py
2. Change compile_profile_inference_aihub.py submitted name to certain model name

Use python deploy.py to call out an interface to run the model and test.

## Optimize Model
1. DyT: Optimize the normalization layer applys Dynamic Tanh in attention blocks, FFN blocks,
 normalization layer of encoder decoder in xdecoder. 
2. SwiGLU: Integrate SwiGLU activation into the Feed-Forward Networks (FFN) by replacing the traditional two-layer MLP structure with a gated variant. Specifically, we will implement two parallel linear projections (linear_v and linear_g) and apply the SiLU activation with gating, aiming to improve model capacity, stability, and efficiency.
3. SwiGLU + DyT: adding c attention and 
4. Linear Attention: Replace standard Multihead Attention with Linear Attention (Performer-style) to enhance computational efficiency. We replaced the traditional softmax attention mechanism with a linear feature map approximation, significantly reducing the quadratic complexity of attention computation to linear time. This improves scalability to high-resolution inputs, accelerates training, and reduces GPU memory usage without sacrificing segmentation performance.
5. GAU: Integrate Gated Attention Units (GAU) in place of standard feed-forward networks to further refine model capacity and dynamic feature selection. This enhances expressiveness and stability, while keeping the model lightweight and well-suited for low-power vision applications.

These adjustment are made in modeling/body/decoder/modules.py and xdecoder.py files.

## Performance
Tho overall best model, which applies DyT, SwiGLU, Linear attention and gated attention has an increase of accuracy and decrease in GPU usage comparing to the baseline model, and it was able to mask the dog image in the example test case.


