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
import torch.distributed as dist
import torch.multiprocessing as mp
# import wandb

from utils.arguments import load_opt_command

# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# def init_wandb(args, job_dir, entity='xueyanz', project='xdecoder', job_name='tmp'):
#     wandb_dir = os.path.join(job_dir, 'wandb')
#     os.makedirs(wandb_dir, exist_ok=True)
#     runid = None
#     if os.path.exists(f"{wandb_dir}/runid.txt"):
#         runid = open(f"{wandb_dir}/runid.txt").read()

#     wandb.init(project=project,
#             name=job_name,
#             dir=wandb_dir,
#             entity=entity,
#             resume="allow",
#             id=runid,
#             config={"hierarchical": True},)

#     open(f"{wandb_dir}/runid.txt", 'w').write(wandb.run.id)
#     wandb.config.update({k: args[k] for k in args if k not in wandb.config})

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size, args=None):
    setup_distributed(rank, world_size)
    
    opt, cmdline_args = load_opt_command(args)
    command = cmdline_args.command
    
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['base_path'] = absolute_user_dir

    opt['rank'] = rank
    opt['local_rank'] = rank
    opt['world_size'] = world_size
    
    if opt['TRAINER'] == 'xdecoder':
        from trainer import XDecoder_Trainer as Trainer
    else:
        assert False, "The trainer type: {} is not defined!".format(opt['TRAINER'])
    
    trainer = Trainer(opt)
    os.environ['TORCH_DISTRIBUTED_DEBUG']='DETAIL'

    if command == "train":
        # if opt['rank'] == 0 and opt['WANDB']:
            # wandb.login(key=os.environ['WANDB_KEY'])
            # init_wandb(opt, trainer.save_folder, job_name=trainer.save_folder)
        trainer.train()
    elif command == "evaluate":
        trainer.eval()
    else:
        raise ValueError(f"Unknown command: {command}")
        
    cleanup()

def main(args=None):
    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        logger.error("No GPU available for distributed training")
        return
        
    world_size = n_gpus
    mp.spawn(main_worker,
             args=(world_size, args),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()
    sys.exit(0)
