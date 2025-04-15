# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import sys
import torch
import logging
import wandb
import qai_hub as hub

from utils.arguments import load_opt_command

# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_wandb(args, job_dir, entity='lpcvc', project='xdecoder', job_name='tmp'):
    wandb_dir = os.path.join(job_dir, 'wandb')
    os.makedirs(wandb_dir, exist_ok=True)
    runid = None
    if os.path.exists(f"{wandb_dir}/runid.txt"):
        runid = open(f"{wandb_dir}/runid.txt").read()

    wandb.init(project='LPCVC_PROJECT',
            name=job_name,
            dir=wandb_dir,
            entity="huanghy1004-boston-university",
            resume="allow",
            id=runid,
            config={"hierarchical": True},)

    open(f"{wandb_dir}/runid.txt", 'w').write(wandb.run.id)
    wandb.config.update({k: args[k] for k in args if k not in wandb.config})

def main(args=None):
    '''
    [Main function for the entry point]
    1. Set environment variables for distributed training.
    2. Load the config file and set up the trainer.
    '''

    opt, cmdline_args = load_opt_command(args)
    command = cmdline_args.command

    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['base_path'] = absolute_user_dir

    # update_opt(opt, command)
    world_size = 1
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])

    if opt['TRAINER'] == 'xdecoder':
        from trainer import XDecoder_Trainer as Trainer
    else:
        assert False, "The trainer type: {} is not defined!".format(opt['TRAINER'])
    
    trainer = Trainer(opt)
    os.environ['TORCH_DISTRIBUTED_DEBUG']='DETAIL'

    if command == "train": 
        if opt['rank'] == 0 and opt['WANDB']:
            wandb.login(key=os.environ['WANDB_KEY'])
            init_wandb(opt, trainer.save_folder, job_name=trainer.save_folder)
        trainer.train()

        if opt['rank'] == 0:
            save_dir = opt.get('SAVE_DIR', './output')
            os.makedirs(save_dir, exist_ok=True)
            # Save each model in trainer.raw_models (if it exists)
            if hasattr(trainer, 'raw_models'):
                for model_name, model in trainer.raw_models.items():
                    save_path = os.path.join(save_dir, f"{model_name}_trained.pth")
                    torch.save(model.state_dict(), save_path)
                    logger.info(f"Saved {model_name} model to {save_path}")
            else:
                # Save the main model if raw_models is not used
                save_path = os.path.join(save_dir, "trained_model.pth")
                torch.save(trainer.model.state_dict(), save_path)
                logger.info(f"Saved model to {save_path}")
            
            # Compile and profile the model on Qualcomm AI Hub
            logger.info("Compiling and profiling the model on Qualcomm AI Hub...")
            try:
                # Get a representative input sample from your dataloader
                sample = next(iter(trainer.val_loader))  # or use trainer.train_loader
                
                # Prepare inputs for tracing
                if hasattr(trainer.model, 'prepare_inputs_for_tracing'):
                    example_inputs = trainer.model.prepare_inputs_for_tracing(sample)
                else:
                    # Fallback to a simple image input if no special preparation method exists
                    example_inputs = {'image': sample['image'][:1].to('cuda')}  # take first item in batch
                
                # Trace the model
                traced_model = torch.jit.trace(trainer.model, example_inputs=example_inputs)
                
                # Submit to Qualcomm AI Hub with more appropriate settings
                compile_job = hub.submit_compile_job(
                    model=traced_model,
                    device=hub.Device('Qualcomm Snapdragon X Elite CRD'),  # More generic target
                    input_specs={'image': {'shape': [1, 3, 512, 512], 'dtype': 'float32'}},
                    options="--target_runtime ai_engine"
                )
                
                # Wait for completion and handle results
                compile_job.wait()
                if compile_job.success:
                    profile_job = hub.submit_profile_job(
                        model=compile_job.get_target_model(),
                        device=hub.Device('Qualcomm Snapdragon X Elite CRD'),
                    )
                    profile_job.wait()
                    
                    # Save compiled model with better naming
                    model_path = os.path.join(save_dir, f"compiled_{opt['MODEL']['NAME']}.tflite")
                    compile_job.get_target_model().download(model_path)
                    logger.info(f"Successfully compiled model saved to {model_path}")
                else:
                    logger.error(f"Compilation failed: {compile_job.status.message}")
                    
            except Exception as e:
                logger.error(f"Qualcomm AI Hub integration failed: {str(e)}")
                    
    elif command == "evaluate":
        trainer.eval()
    else:
        raise ValueError(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
    sys.exit(0)
