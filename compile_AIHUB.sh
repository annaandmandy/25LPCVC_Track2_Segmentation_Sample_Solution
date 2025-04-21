module load miniconda
conda activate xdecoder

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export WANDB_KEY="a9007bbf2af8533987785004be9fae23e52af2f5"
export QUALCOMM_AI_HUB_KEY="44165a068eebd71635bbb8e9198756c50e392b02"

python compile_and_profile/compile_profile_inference_aihub.py