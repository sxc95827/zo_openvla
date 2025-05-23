#!/bin/bash
export WANDB_MODE=disabled
# Set CUDA_VISIBLE_DEVICES to use GPUs 4, 5, 6, 7
export CUDA_VISIBLE_DEVICES=0

# Run finetune_zo.py on 4 GPUs
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_zo.py \
  --vla_path "/media/SSD7/personal/Norman/openvla_libero_data/libero_object_no_noops/openvla-7b/" \
  --data_root_dir "/media/SSD7/personal/Norman/openvla_libero_data/" \
  --dataset_name "libero_spatial_no_noops" \
  --run_root_dir "/media/SSD7/personal/Norman/openvla_libero_data/libero_spatial_no_noops/" \
  --adapter_tmp_dir "/media/SSD7/personal/Norman/openvla_libero_data/libero_spatial_no_noops/" \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --zo_eps 1e-3 \
  --perturbation_mode two_side \
  --q 1 \
  --gradient_sparsity 0.9 \
  --sparse_gradient_resample_steps 10 \
  --max_steps 20000 \
  --save_steps 2000 \
  --log_steps 10 \
  --detailed_logging True \
  --log_file "detailed_training_log.jsonl" \
  --save_best_checkpoint True \
  --checkpoint_metric "action_accuracy" \
  --checkpoint_metric_better "higher" \
  --min_steps_between_checkpoints 2000 \
  --max_checkpoints_to_keep 3 \
#  --image_aug True \
#   --wandb_project "openvla-zo" \
#   --wandb_entity <YOUR_WANDB_ENTITY> \
#  --run_id_note "zo-libero-spatial-no-noops-4gpu" 