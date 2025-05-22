"""
finetune_zo.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA) and zero-order optimization (ZO-SGD) for gradient estimation.

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune_zo.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune_zo.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
import re
import numpy as np
import json
import datetime
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Union, Any, Iterable

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

import pdb
import datetime


# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "offline"


@dataclass
class FinetuneZOConfig:
    # fmt: off
    vla_path: str = "/media/SSD7/personal/Norman/openvla_libero_data/libero_object_no_noops/openvla-7b/"                           # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("/media/SSD7/personal/Norman/openvla_libero_data/libero_object_no_noops/1.0.0/")        # Path to Open-X dataset directory
    dataset_name: str = "libero_object_no_noops"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)
    # Checkpoint Saving Strategy
    save_best_checkpoint: bool = False                              # Whether to save only when model improves
    checkpoint_metric: str = "loss"                                 # Metric to use for best checkpoint ["loss", "action_accuracy", "l1_loss"]
    checkpoint_metric_better: str = "lower"                         # Whether lower or higher metric is better ["lower", "higher"]
    min_steps_between_checkpoints: int = 1000                       # Minimum steps between consecutive checkpoints
    max_checkpoints_to_keep: int = 3                                # Maximum number of best checkpoints to keep
    
    # Logging Parameters
    log_steps: int = 10                                             # Log metrics every N steps
    detailed_logging: bool = True                                   # Enable detailed logging
    log_file: str = "training_log.jsonl"                            # Path to save detailed training log

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Zero-Order Optimization Parameters
    zo_eps: float = 1e-3                                            # Perturbation size for zero-order optimization
    perturbation_mode: str = "two_side"                             # Perturbation mode: "one_side" or "two_side"
    q: int = 1                                                      # Number of perturbations per step
    gradient_sparsity: Optional[float] = None                       # Gradient sparsity (None for no sparsity)
    sparse_gradient_group: str = "layer"                            # Group level for sparse gradient: "layer" or "global"
    sparse_gradient_resample_steps: int = 1                         # Steps between resampling sparse gradient masks

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on


# Utility function for fast random masking
def fast_random_mask_like(tensor, sparsity, generator=None):
    """Create a random mask with the given sparsity."""
    if sparsity <= 0:
        return torch.ones_like(tensor, dtype=torch.bool)
    if sparsity >= 1:
        return torch.zeros_like(tensor, dtype=torch.bool)
    
    # Generate random values
    if generator is not None:
        random_values = torch.rand(tensor.shape, generator=generator, device=tensor.device)
    else:
        random_values = torch.rand(tensor.shape, device=tensor.device)
    
    # Create mask based on threshold
    threshold = torch.quantile(random_values.flatten(), sparsity)
    return random_values > threshold


# Add logging function
def write_log(log_file, log_data):
    """Write log data to a JSONL file"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_data) + '\n')


# Function to compute gradient norm
def compute_gradient_norm(parameters):
    """Compute L2 norm of gradients"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2.0)
    return total_norm


# Function to compute parameter update norm
def compute_param_update_norm(optimizer):
    """Compute L2 norm of parameter updates"""
    total_norm = 0.0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


@draccus.wrap()
def finetune_zo(cfg: FinetuneZOConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}` using Zero-Order Optimization")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
        f"+zo-eps{cfg.zo_eps}"
        f"+q{cfg.q}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.gradient_sparsity is not None:
        exp_id += f"+sparse{cfg.gradient_sparsity}"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    exp_date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir, adapter_dir = cfg.run_root_dir / exp_id / exp_date_time, cfg.adapter_tmp_dir / exp_id / exp_date_time
    os.makedirs(run_dir, exist_ok=True)
    
    # Initialize best metric tracking
    best_metric_value = float('inf') if cfg.checkpoint_metric_better == "lower" else float('-inf')
    best_checkpoints = []  # List to store best checkpoints info: [(step, metric_value, checkpoint_path)]
    last_checkpoint_step = 0  # Step of last saved checkpoint
    
    # Set log file path
    log_file_path = os.path.join(run_dir, cfg.log_file)
    
    # Initialize log file
    if distributed_state.is_main_process:
        # Record training configuration
        config_log = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event": "training_start",
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()}
        }
        write_log(log_file_path, config_log)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we use SGD for zero-order optimization
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = SGD(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft-zo+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)
    
    # Add more metrics tracking
    recent_loss_diffs = deque(maxlen=cfg.grad_accumulation_steps)  # loss1 - loss2
    recent_projected_grads = deque(maxlen=cfg.grad_accumulation_steps)  # projected gradient values
    recent_param_norms = deque(maxlen=cfg.grad_accumulation_steps)  # parameter norms
    recent_grad_norms = deque(maxlen=cfg.grad_accumulation_steps)  # gradient norms
    recent_update_norms = deque(maxlen=cfg.grad_accumulation_steps)  # update norms

    # Initialize random generators for zero-order optimization
    zo_random_seed = np.random.randint(1000000000)
    sparse_grad_rng = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
    sparse_grad_random_seed = np.random.randint(1000000000)
    sparse_grad_rng.manual_seed(sparse_grad_random_seed)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        
        # Initialize named parameters to optimize
        named_parameters_to_optim = []
        for name, param in vla.named_parameters():
            if param.requires_grad:
                named_parameters_to_optim.append((name, param))
        
        for batch_idx, batch in enumerate(dataloader):
            # Resample sparse gradient mask if needed
            if batch_idx % cfg.sparse_gradient_resample_steps == 0:
                sparse_grad_random_seed = np.random.randint(1000000000)
                sparse_grad_rng.manual_seed(sparse_grad_random_seed)
            
            # Sample new random seed for zero-order optimization
            zo_random_seed = np.random.randint(1000000000)
            
            # Calculate initial parameter norm
            if cfg.detailed_logging:
                with torch.no_grad():
                    initial_param_norm = sum(p.norm().item() ** 2 for name, p in named_parameters_to_optim) ** 0.5
                    recent_param_norms.append(initial_param_norm)
            
            # Zero-order optimization step
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # First function evaluation
                with torch.no_grad():
                    vla.eval()  # Set model to evaluation mode
                    output1: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                    )
                    loss1 = output1.loss.detach()
                
                # Perturb parameters
                torch.manual_seed(zo_random_seed)
                for name, param in named_parameters_to_optim:
                    grad_sparsity = cfg.gradient_sparsity
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                    if grad_sparsity is not None:
                        z[fast_random_mask_like(z, grad_sparsity, generator=sparse_grad_rng)] = 0
                    param.data = param.data + z * cfg.zo_eps
                
                # Second function evaluation
                with torch.no_grad():
                    vla.eval()  # Set model to evaluation mode
                    output2: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                    )
                    loss2 = output2.loss.detach()
                    
                    # Record loss difference
                    if cfg.detailed_logging:
                        loss_diff = (loss1 - loss2).item()
                        recent_loss_diffs.append(loss_diff)
                    
                    # Clean up to free memory
                    del output2
                    torch.cuda.empty_cache()
                
                # Reset parameters to original state
                torch.manual_seed(zo_random_seed)
                for name, param in named_parameters_to_optim:
                    grad_sparsity = cfg.gradient_sparsity
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                    if grad_sparsity is not None:
                        z[fast_random_mask_like(z, grad_sparsity, generator=sparse_grad_rng)] = 0
                    param.data = param.data - z * cfg.zo_eps
                
                # Estimate gradient
                if cfg.perturbation_mode == "one_side":
                    projected_grad = ((loss1 - loss2) / cfg.zo_eps).item()
                else:  # two side perturbation
                    projected_grad = ((loss1 - loss2) / (2 * cfg.zo_eps)).item()
                
                # Record projected gradient value
                if cfg.detailed_logging:
                    recent_projected_grads.append(projected_grad)
                
                # Apply estimated gradient
                torch.manual_seed(zo_random_seed)
                for name, param in named_parameters_to_optim:
                    grad_sparsity = cfg.gradient_sparsity
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                    if grad_sparsity is not None:
                        z[fast_random_mask_like(z, grad_sparsity, generator=sparse_grad_rng)] = 0
                    
                    # Set gradient
                    param.grad = projected_grad * z / cfg.q
                
                # Calculate gradient norm
                if cfg.detailed_logging:
                    grad_norm = compute_gradient_norm(trainable_params)
                    recent_grad_norms.append(grad_norm.item())
                    
                # Optimizer step
                optimizer.step()
                
                # Calculate parameter update norm
                if cfg.detailed_logging:
                    update_norm = compute_param_update_norm(optimizer)
                    recent_update_norms.append(update_norm)
                
                optimizer.zero_grad()
                
                # Set model back to train mode
                vla.train()
                
                # Use loss1 for logging
                loss = loss1

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output1.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)
            
            # Get current performance metric value (to determine whether to save checkpoint)
            current_metric_value = None
            if cfg.checkpoint_metric == "loss":
                current_metric_value = smoothened_loss
            elif cfg.checkpoint_metric == "action_accuracy":
                current_metric_value = smoothened_action_accuracy
            elif cfg.checkpoint_metric == "l1_loss":
                current_metric_value = smoothened_l1_loss
            
            # Calculate average detailed metrics
            if cfg.detailed_logging and len(recent_loss_diffs) > 0:
                avg_loss_diff = sum(recent_loss_diffs) / len(recent_loss_diffs)
                avg_projected_grad = sum(recent_projected_grads) / len(recent_projected_grads)
                avg_param_norm = sum(recent_param_norms) / len(recent_param_norms)
                avg_grad_norm = sum(recent_grad_norms) / len(recent_grad_norms)
                avg_update_norm = sum(recent_update_norms) / len(recent_update_norms)
            else:
                avg_loss_diff = 0.0
                avg_projected_grad = 0.0
                avg_param_norm = 0.0
                avg_grad_norm = 0.0
                avg_update_norm = 0.0

            # Write metrics to log file (every cfg.log_steps steps)
            if distributed_state.is_main_process and gradient_step_idx % cfg.log_steps == 0:
                # Basic metrics
                log_data = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "step": gradient_step_idx,
                    "metrics": {
                        "loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                    }
                }
                
                # Detailed metrics
                if cfg.detailed_logging:
                    log_data["metrics"].update({
                        "loss_diff": avg_loss_diff,
                        "projected_grad": avg_projected_grad,
                        "param_norm": avg_param_norm,
                        "grad_norm": avg_grad_norm,
                        "update_norm": avg_update_norm,
                        "zo_eps": cfg.zo_eps,
                        "learning_rate": cfg.learning_rate,
                    })
                
                # Write to log file
                write_log(log_file_path, log_data)
                
                # Output to console
                print(f"Step {gradient_step_idx}: loss={smoothened_loss:.4f}, accuracy={smoothened_action_accuracy:.4f}, l1_loss={smoothened_l1_loss:.4f}")
                if cfg.detailed_logging:
                    print(f"  - Details: loss_diff={avg_loss_diff:.6f}, proj_grad={avg_projected_grad:.6f}, param_norm={avg_param_norm:.2f}, grad_norm={avg_grad_norm:.6f}, update_norm={avg_update_norm:.6f}")

            # Push Metrics to W&B (每10个梯度步骤)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                metrics_dict = {
                    "train_loss": smoothened_loss,
                    "action_accuracy": smoothened_action_accuracy,
                    "l1_loss": smoothened_l1_loss,
                }
                
                # Add detailed metrics to wandb
                if cfg.detailed_logging:
                    metrics_dict.update({
                        "loss_diff": avg_loss_diff,
                        "projected_grad": avg_projected_grad,
                        "param_norm": avg_param_norm,
                        "grad_norm": avg_grad_norm,
                        "update_norm": avg_update_norm,
                    })
                
                wandb.log(metrics_dict, step=gradient_step_idx)

            # Update progress bar
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                progress.update()

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            checkpoint_time = gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0
            should_save_checkpoint = checkpoint_time
            
            # Checkpoint saving strategy based on performance metrics
            is_best_checkpoint = False
            if cfg.save_best_checkpoint and current_metric_value is not None:
                min_steps_passed = gradient_step_idx - last_checkpoint_step >= cfg.min_steps_between_checkpoints
                
                # Check if this is a better metric value
                is_better = (cfg.checkpoint_metric_better == "lower" and current_metric_value < best_metric_value) or \
                            (cfg.checkpoint_metric_better == "higher" and current_metric_value > best_metric_value)
                
                if is_better and min_steps_passed:
                    should_save_checkpoint = True
                    is_best_checkpoint = True
                    best_metric_value = current_metric_value
            
            if should_save_checkpoint:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")
                    if is_best_checkpoint:
                        print(f"  -> New best {cfg.checkpoint_metric}: {current_metric_value:.6f}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir
                    
                    # Create specific directory for performance-based checkpoint
                    if cfg.save_best_checkpoint and is_best_checkpoint:
                        checkpoint_name = f"checkpoint-{gradient_step_idx}-{cfg.checkpoint_metric}{current_metric_value:.6f}"
                        save_dir = Path(str(save_dir.parent) + f"/{checkpoint_name}")
                        os.makedirs(save_dir, exist_ok=True)

                    # Save Processor & Weights
                    processor.save_pretrained(run_dir if not is_best_checkpoint else save_dir)
                    vla.module.save_pretrained(save_dir)
                    
                    # Record checkpoint saving
                    checkpoint_log = {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "event": "checkpoint_saved",
                        "step": gradient_step_idx,
                        "path": str(save_dir),
                        "is_best": is_best_checkpoint,
                    }
                    if current_metric_value is not None:
                        checkpoint_log[cfg.checkpoint_metric] = current_metric_value
                    write_log(log_file_path, checkpoint_log)
                    
                    # Update last checkpoint step
                    last_checkpoint_step = gradient_step_idx

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

                # Merge LoRA weights into model backbone for faster inference
                #   =>> Note that merging is slow and can be done post-hoc to speed up training
                if cfg.use_lora:
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=False
                    )
                    merged_vla = PeftModel.from_pretrained(base_vla, save_dir if is_best_checkpoint else adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    
                    if distributed_state.is_main_process:
                        if cfg.save_best_checkpoint and is_best_checkpoint:
                            # Save best checkpoint
                            merged_vla.save_pretrained(save_dir)
                            
                            # Add checkpoint info to list
                            best_checkpoints.append((gradient_step_idx, current_metric_value, str(save_dir)))
                            
                            # Sort checkpoint list by metric
                            if cfg.checkpoint_metric_better == "lower":
                                best_checkpoints.sort(key=lambda x: x[1])  # Sort by metric value ascending
                            else:
                                best_checkpoints.sort(key=lambda x: x[1], reverse=True)  # Sort by metric value descending
                            
                            # If number of checkpoints exceeds limit, remove worst checkpoint
                            if len(best_checkpoints) > cfg.max_checkpoints_to_keep:
                                worst_checkpoint = best_checkpoints.pop()  # Remove worst checkpoint at end of list
                                if os.path.exists(worst_checkpoint[2]):
                                    print(f"Removing worst checkpoint: {worst_checkpoint[2]} with {cfg.checkpoint_metric}={worst_checkpoint[1]:.6f}")
                                    import shutil
                                    shutil.rmtree(worst_checkpoint[2])
                            
                            # Record all currently retained best checkpoints
                            checkpoints_log = {
                                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "event": "best_checkpoints_update",
                                "checkpoints": [{"step": step, cfg.checkpoint_metric: value, "path": path} 
                                               for step, value, path in best_checkpoints]
                            }
                            write_log(log_file_path, checkpoints_log)
                            
                            print(f"Saved Best Model Checkpoint for Step {gradient_step_idx} at: {save_dir}")
                        elif cfg.save_latest_checkpoint_only and not cfg.save_best_checkpoint:
                            # Overwrite latest checkpoint
                            merged_vla.save_pretrained(run_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:
                            # Prepare to save checkpoint in new directory
                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            # Save dataset statistics to new directory
                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                            # Save processor and model weights to new directory
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                # Block on Main Process Checkpointing
                dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                # Record training completion
                if distributed_state.is_main_process:
                    end_log = {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "event": "training_complete",
                        "total_steps": gradient_step_idx,
                        "final_metrics": {
                            "loss": smoothened_loss,
                            "action_accuracy": smoothened_action_accuracy,
                            "l1_loss": smoothened_l1_loss,
                        }
                    }
                    write_log(log_file_path, end_log)
                break


if __name__ == "__main__":
    finetune_zo()
