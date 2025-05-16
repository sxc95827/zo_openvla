#!/bin/bash

# Set CUDA_VISIBLE_DEVICES to use GPUs 4, 5, 6, 7
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Run finetune_zo.py on 4 GPUs
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_zo.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "/media/SSD7/personal/Norman/openvla_libero_data/" \
  --dataset_name "libero_spatial_no_noops" \
  --run_root_dir "/media/SSD7/personal/Norman/openvla_libero_data/libero_spatial_no_noops/" \
  --adapter_tmp_dir "/media/SSD7/personal/Norman/openvla_libero_data/libero_spatial_no_noops/" \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --zo_eps 1e-3 \
  --perturbation_mode two_side \
  --q 1 \
  --gradient_sparsity 0.9 \
  --sparse_gradient_resample_steps 10 \
  --max_steps 100000 \
  --save_steps 5000 \
  --image_aug True \
#   --wandb_project "openvla-zo" \
#   --wandb_entity <YOUR_WANDB_ENTITY> \
  --run_id_note "zo-libero-spatial-no-noops-4gpu" 