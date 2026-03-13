# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Lumina-T2I using PyTorch FSDP.
"""
import argparse
from collections import OrderedDict
import contextlib
from copy import deepcopy
from datetime import datetime
import functools
from functools import partial
import json
import logging
import os
import random
import socket
from time import time
from PIL import Image
from diffusers.models import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import ItemProcessor, MyDataset, read_general2
from grad_norm import calculate_l2_grad_norm, get_model_parallel_dim_dict, scale_grad
from imgproc import generate_crop_size_list, var_center_crop
import models
from parallel import distributed_init, get_intra_node_process_group
from transport import create_transport, Sampler
from tqdm import tqdm 
from torchvision.transforms import ToPILImage

#############################################################################
#                            Data item Processor                            #
#############################################################################


class T2IItemProcessor(ItemProcessor):
    def __init__(self, transform, mask_transform=None):
        self.image_transform = transform
        self.mask_transform = mask_transform
    
    def process_item(self, data_item, training_mode=False):
        try:
            image_root = data_item.get("_image_root", None)
            if "caption" in data_item:
                # 修复：将此处的变量名严格定义为 image_path
                image_path = data_item.get("image") or data_item.get("image_path")
                if not image_path:
                    raise ValueError("Missing 'image_path' key.")
                
                full_image_path = read_general2(image_path, image_root)
                image = Image.open(full_image_path).convert("RGB")
                text = data_item.get("caption", "")
                
                mask_val = data_item.get("mask") or data_item.get("mask_path")
                if mask_val is not None:
                    full_mask_path = read_general2(mask_val, image_root)
                    mask = Image.open(full_mask_path).convert("L")
                else:
                    mask = Image.new("L", image.size, 0)
                
                image = self.image_transform(image)
                if self.mask_transform:
                    mask = self.mask_transform(mask)
                else:
                    mask = self.image_transform(mask.convert("RGB"))[0:1, :, :]
                return image, mask, text, image_path
            else:
                raise ValueError("Data item must contain 'caption' key.")
        except Exception as e:
            print(f"Error processing data_item: {e}")
            return None, None, "", ""

def dataloader_collate_fn(samples):
    samples = [s for s in samples if s[0] is not None]
    image = [x[0] for x in samples]
    mask = [x[1] for x in samples]
    caps = [x[2] for x in samples] 
    return image, mask, caps


def get_train_sampler(dataset, rank, world_size, global_batch_size, max_steps, resume_step, seed):
    sample_indices = torch.empty([max_steps * global_batch_size // world_size], dtype=torch.long)
    epoch_id, fill_ptr, offs = 0, 0, 0
    while fill_ptr < sample_indices.size(0):
        g = torch.Generator()
        g.manual_seed(seed + epoch_id)
        epoch_sample_indices = torch.randperm(len(dataset), generator=g)
        epoch_id += 1
        epoch_sample_indices = epoch_sample_indices[(rank + offs) % world_size :: world_size]
        offs = (offs + world_size - len(dataset) % world_size) % world_size
        epoch_sample_indices = epoch_sample_indices[: sample_indices.size(0) - fill_ptr]
        sample_indices[fill_ptr : fill_ptr + epoch_sample_indices.size(0)] = epoch_sample_indices
        fill_ptr += epoch_sample_indices.size(0)
    return sample_indices[resume_step * global_batch_size // world_size :].tolist()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def setup_lm_fsdp_sync(model: nn.Module) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in list(model.layers),
        ),
        process_group=get_intra_node_process_group(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=next(model.parameters()).dtype,
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        process_group=fs_init.get_data_parallel_group(),
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.precision],
            reduce_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.grad_precision or args.precision],
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def setup_mixed_precision(args):
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif args.precision in ["bf16", "fp16", "fp32"]:
        pass
    else:
        raise NotImplementedError(f"Unknown precision: {args.precision}")


def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            captions.append("")
    with torch.no_grad():
        text_inputs = tokenizer(
            captions, padding=True, pad_to_multiple_of=8, max_length=256, truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask
        prompt_embeds = text_encoder(
            input_ids=text_input_ids, attention_mask=prompt_masks, output_hidden_states=True
        ).hidden_states[-2]
    return prompt_embeds.cuda(), prompt_masks.cuda()


#############################################################################
#                                Training Loop                              #
#############################################################################


def initialize_progress_bar(total_steps):
    return tqdm(total=total_steps, desc="Training Progress", leave=True)


def update_progress_bar(pbar, step, loss=None, grad_norm=None, lr=None, max_steps=100):
    postfix = {"step": f"{step}/{max_steps}"}
    if loss is not None:
        postfix["loss"] = f"{loss:.4f}"
    if grad_norm is not None:
        postfix["grad_norm"] = f"{grad_norm:.4f}"
    if lr is not None:
        postfix["lr"] = f"{lr:.6f}"
    pbar.set_postfix(postfix)
    pbar.update()


def calculate_l2_grad_norm(model, model_parallel_dim_dict):
    """
    Calculate the L2 norm of the gradients.
    """
    grad_norm_sq = 0.0
    for n, p in model.named_parameters():
        if p.grad is not None:
            mp_dim = model_parallel_dim_dict.get(n, None)
            if mp_dim is not None:
                mp_size = fs_init.get_model_parallel_world_size()
                grad_norm_sq += torch.norm(p.grad, p=2).item() ** 2 * mp_size / min(mp_size, p.shape[mp_dim])
            else:
                grad_norm_sq += torch.norm(p.grad, p=2).item() ** 2
    return grad_norm_sq**0.5


def find_latest_checkpoint(checkpoint_path):
    """
    Find the latest checkpoint in the directory if path ends with 'latest'.
    """
    if not checkpoint_path or not checkpoint_path.endswith('latest'):
        return checkpoint_path
        
    parent_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(parent_dir):
        return checkpoint_path
    
    checkpoint_dirs = []
    for d in os.listdir(parent_dir):
        full_path = os.path.join(parent_dir, d)
        if os.path.isdir(full_path) and d.isdigit():
            checkpoint_dirs.append((int(d), full_path))
    
    if not checkpoint_dirs:
        return checkpoint_path
    
    latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: x[0], reverse=True)[0][1]
    return latest_checkpoint


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    distributed_init(args)

    dp_world_size = fs_init.get_data_parallel_world_size()
    dp_rank = fs_init.get_data_parallel_rank()
    mp_world_size = fs_init.get_model_parallel_world_size()
    mp_rank = fs_init.get_model_parallel_rank()

    assert args.global_batch_size % dp_world_size == 0, "Batch size must be divisible by data parallel world size."
    local_batch_size = args.global_batch_size // dp_world_size
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    setup_mixed_precision(args)

    os.makedirs(args.results_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + args.id
    time_based_dir = os.path.join(args.results_dir, current_time)
    os.makedirs(time_based_dir, exist_ok=True)

    checkpoint_dir = os.path.join(time_based_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    args_file_path = os.path.join(time_based_dir, "args.json")
    with open(args_file_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)

    if rank == 0:
        logger = create_logger(time_based_dir)
        logger.info(f"Experiment directory: {time_based_dir}")
        tb_logger = SummaryWriter(
            os.path.join(
                time_based_dir,
                "tensorboard",
                datetime.now().strftime("%Y%m%d_%H%M%S_") + socket.gethostname(),
            )
        )
    else:
        logger = create_logger(None)
        tb_logger = None
        
    logger.info("Training arguments: " + json.dumps(args.__dict__, indent=2))
    logger.info(f"Setting up language model: google/gemma-2b")

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    tokenizer.padding_side = "right"

    text_encoder = (
        AutoModelForCausalLM.from_pretrained(
            "google/gemma-2b",
            torch_dtype=torch.bfloat16,
        )
        .get_decoder()
    )
    text_encoder.eval()
    cap_feat_dim = text_encoder.config.hidden_size
        
    if args.model in ["NextDiT_2B_patch2", "DiT_Llama2_7B_patch2", "NextDiT_2B_GQA_patch2"]:
        model = models.__dict__[args.model](
            in_channels=16 if args.vae == "sd3" else 4,
            qk_norm=args.qk_norm,
            cap_feat_dim=cap_feat_dim,
        )
    else:
        raise ValueError(f"Model {args.model} is not supported!")
            
    logger.info(f"DiT Parameters: {model.parameter_count():,}")
    model_patch_size = model.patch_size
    model_parallel_dim_dict = get_model_parallel_dim_dict(model)

    if args.auto_resume and args.resume is None:
        try:
            existing_checkpoints = os.listdir(checkpoint_dir)
            if len(existing_checkpoints) > 0:
                existing_checkpoints.sort()
                args.resume = os.path.join(checkpoint_dir, existing_checkpoints[-1])
        except Exception:
            pass
        if args.resume is not None:
            logger.info(f"Auto resuming from: {args.resume}")

    if args.resume:
        latest_resume = find_latest_checkpoint(args.resume)
        if latest_resume != args.resume:
            logger.info(f"Found latest checkpoint for resume: {latest_resume}")
            args.resume = latest_resume
    
    if args.init_from:
        latest_init = find_latest_checkpoint(args.init_from)
        if latest_init != args.init_from:
            logger.info(f"Found latest checkpoint for init_from: {latest_init}")
            args.init_from = latest_init

    # 隐患修复 1：加载预训练基座 DiT 权重（防止模型随机初始化）
    if args.init_from:
        logger.info(f"Loading Base DiT weights from: {args.init_from}")
        base_ckpt_path = os.path.join(args.init_from, "consolidated.00-of-01.pth")
        if os.path.exists(base_ckpt_path):
            base_state_dict = torch.load(base_ckpt_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(base_state_dict, strict=False)
            logger.info(f"Base weights loaded. Missing keys (expected mask_embedder): {missing}")
        
    # 隐患修复 2：如果有断点续训，加载我们极简版的 adapter 权重
    if args.resume:
        logger.info(f"Loading Adapter weights from: {args.resume}")
        adapter_ckpt_path = os.path.join(args.resume, "adapter.pth")
        if os.path.exists(adapter_ckpt_path):
            adapter_state_dict = torch.load(adapter_ckpt_path, map_location="cpu")
            model.load_state_dict(adapter_state_dict, strict=False)

    # === 极简版初始化 ===
    model = model.to(device)
    
    if args.checkpointing:
        print("Applying gradient checkpointing")
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            # 将 NO_REENTRANT 改为 REENTRANT，这会以略微增加计算时间为代价，
            # 极度且严格地压榨中间激活显存。
            checkpoint_impl=CheckpointImpl.REENTRANT, 
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in list(model.layers),
        )
    logger.info(f"Model:\n{model}\n")

    transport = create_transport("Linear", "velocity", None, None, None, snr_type=args.snr_type)
    
    if args.vae == "sd3":
        logger.info("Using SD3 VAE")
        vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae").to(device)
    elif args.vae == "sdxl":
        logger.info("Using SDXL VAE")
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    else:
        vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{args.vae}"
            if args.local_diffusers_model_root is None
            else os.path.join(args.local_diffusers_model_root, f"stabilityai/sd-vae-ft-{args.vae}")
        ).to(device)

    for name, param in model.named_parameters():
        if "mask_embedder" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False    

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)

    if args.resume:
        logger.info(f"Resuming optimizer states from: {args.resume}")
        opt_path = os.path.join(args.resume, "optimizer.pth")
        if os.path.exists(opt_path):
            opt.load_state_dict(torch.load(opt_path, map_location="cpu"))
        for param_group in opt.param_groups:
            param_group["lr"] = args.lr
            param_group["weight_decay"] = args.wd

        with open(os.path.join(args.resume, "resume_step.txt")) as f:
            resume_step = int(f.read().strip())
    elif args.init_from:
        # 修复：加载预训练权重时，直接将步数设为 0，不强求存在 resume_step.txt
        resume_step = 0
    else:
        resume_step = 0

    logger.info(f"Resume step: {resume_step}")
    
    logger.info("Creating FIXED resolution data transform...")
    image_transform = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    # 针对 mask 需要使用最近邻插值，保证 0/1 二值属性不被破坏
    mask_transform = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ]
    )

    dataset = MyDataset(
        args.data_path,
        # 新增 mask_transform 传入
        item_processor=T2IItemProcessor(image_transform, mask_transform=mask_transform), 
        cache_on_disk=args.cache_data_on_disk,
    )
    num_samples = args.global_batch_size * args.max_steps
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(f"Total # samples to consume: {num_samples:,} ({num_samples / len(dataset):.2f} epochs)")
    
    sampler = get_train_sampler(
        dataset,
        dp_rank,
        dp_world_size,
        args.global_batch_size,
        args.max_steps,
        resume_step,
        args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dataloader_collate_fn,
        drop_last=True
    )

    model.train()

    log_steps = 0
    running_loss = 0
    running_grad_norm = 0
    start_time = time()
    
    pbar = initialize_progress_bar(total_steps=len(loader))

    logger.info(f"Training for {args.max_steps:,} steps...")
    for step, (x, masks, caps) in enumerate(loader, start=resume_step):
        # 强制将 List 堆叠为形状一致的张量 [B, C, H, W]
        x_tensor = torch.stack(x).to(device, non_blocking=True)
        masks_tensor = torch.stack(masks).to(device, non_blocking=True)
         
        with torch.no_grad():
            vae_scale = {"sdxl": 0.13025, "sd3": 1.5305, "ema": 0.18215, "mse": 0.18215}[args.vae]
            vae_shift = {"sdxl": 0.0, "sd3": 0.0609, "ema": 0.0, "mse": 0.0}[args.vae]
            if step == resume_step:
                logger.warning(f"VAE scale: {vae_scale}, VAE shift: {vae_shift}")
            
            # === 修改 VAE 编码输入 ===
            # 将列表解析替换为直接对批量张量编码
            x_latent = (vae.encode(x_tensor).latent_dist.sample() - vae_shift) * vae_scale
            # ========================

        with torch.no_grad():
            cap_feats, cap_mask = encode_prompt(caps, text_encoder, tokenizer, args.caption_dropout_prob)
            
            # 这里的 masks_tensor 直接使用，不需要重新 stack 了
            target_h, target_w = x_latent.shape[2], x_latent.shape[3] # VAE后的空间维度
            masks_latent = torch.nn.functional.interpolate(
                masks_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False
            )

        loss_item = 0.0
        opt.zero_grad()
        torch.cuda.empty_cache()
        for mb_idx in range((local_batch_size - 1) // args.micro_batch_size + 1):
            mb_st = mb_idx * args.micro_batch_size
            mb_ed = min((mb_idx + 1) * args.micro_batch_size, local_batch_size)
            last_mb = mb_ed == local_batch_size

            # === 详细注释：提取当前微批次的变量切片 ===
            x_mb = x_latent[mb_st:mb_ed] 
            cap_feats_mb = cap_feats[mb_st:mb_ed]
            cap_mask_mb = cap_mask[mb_st:mb_ed]
            masks_latent_mb = masks_latent[mb_st:mb_ed]
            
            # 组合模型前向传播所需的所有关键字参数 (Keyword Arguments)
            model_kwargs = dict(cap_feats=cap_feats_mb, cap_mask=cap_mask_mb, struct_mask=masks_latent_mb)
            
            # 使用混合精度 (Mixed Precision) 进行前向传播与损失计算
            with {
                "bf16": torch.amp.autocast('cuda', dtype=torch.bfloat16), # 已修复废弃警告
                "fp16": torch.amp.autocast('cuda', dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.precision]:
                loss_dict = transport.training_losses(model, x_mb, model_kwargs)

            loss = loss_dict["loss"].sum() / local_batch_size
            loss_item += loss.item()
            
            # 直接反向传播（去除 FSDP 的 no_sync 包装）
            loss.backward()

        # 使用 PyTorch 原生梯度裁剪（替代原有复杂函数）
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()

        if tb_logger is not None:
            tb_logger.add_scalar("train/loss", loss_item, step)
            tb_logger.add_scalar("train/grad_norm", grad_norm, step)
            tb_logger.add_scalar("train/lr", opt.param_groups[0]["lr"], step)

        opt.step()
            
        update_progress_bar(pbar, step, loss=loss.item(), grad_norm=grad_norm, 
                          lr=opt.param_groups[0]["lr"], max_steps=args.max_steps)

        running_loss += loss_item
        running_grad_norm += grad_norm
        log_steps += 1
        if (step + 1) % args.log_every == 0:
            torch.cuda.synchronize()
            end_time = time()
            secs_per_step = (end_time - start_time) / log_steps
            imgs_per_sec = args.global_batch_size * log_steps / (end_time - start_time)
            
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()
            avg_grad_norm = running_grad_norm / log_steps
            logger.info(
                f"(step={step + 1:07d}) "
                f"Train Loss: {avg_loss:.4f}, "
                f"Train Grad Norm: {avg_grad_norm:.4f}, "
                f"Train Secs/Step: {secs_per_step:.2f}, "
                f"Train Imgs/Sec: {imgs_per_sec:.2f}"
            )
            
            running_loss = 0
            running_grad_norm = 0
            log_steps = 0
            start_time = time()

        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.max_steps:
            checkpoint_path = f"{checkpoint_dir}/{step + 1:07d}"
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # 核心优化：剥离 5GB 的大模型，只保存你训练的 mask_embedder 参数 (几MB)！
            adapter_weights = {k: v for k, v in model.state_dict().items() if "mask_embedder" in k}
            torch.save(adapter_weights, os.path.join(checkpoint_path, "adapter.pth"))
            torch.save(opt.state_dict(), os.path.join(checkpoint_path, "optimizer.pth"))
            
            if dist.get_rank() == 0:
                torch.save(args, os.path.join(checkpoint_path, "model_args.pth"))
                with open(os.path.join(checkpoint_path, "resume_step.txt"), "w") as f:
                    print(step + 1, file=f)
            
            logger.info(f"Saved lightweight adapter to {checkpoint_path}")
        
    dist.barrier()
    model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='debugging', required=False)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cache_data_on_disk", default=False, action="store_true")
    parser.add_argument("--results_dir", type=str, required=False, default='results/debugging/')
    parser.add_argument("--model", type=str, default="NextDiT_2B_GQA_patch2")
    parser.add_argument("--image_size", type=int, choices=[256, 512, 1024], default=256)
    parser.add_argument("--max_steps", type=int, default=100_000, help="Number of training steps.")
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse", "sdxl", "sd3"], default="ema")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--master_port", type=int, default=18181)
    parser.add_argument("--model_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel", type=str, choices=["sdp", "fsdp"], default="fsdp")
    parser.add_argument("--precision", choices=["fp32", "tf32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--grad_precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--checkpointing", action="store_true", default=False, help="Enable gradient checkpointing")
    parser.add_argument(
        "--local_diffusers_model_root",
        type=str,
        help="Specify the root directory if diffusers models are to be loaded "
        "from the local filesystem (instead of being automatically "
        "downloaded from the Internet). Useful in environments without "
        "Internet access.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--no_auto_resume",
        action="store_false",
        dest="auto_resume",
        help="Do NOT auto resume from the last checkpoint in --results_dir.",
    )
    parser.add_argument("--resume", type=str, help="Resume training from a checkpoint folder.")
    parser.add_argument(
        "--init_from",
        type=str,
        help="Initialize the model weights from a checkpoint folder. "
        "Compared to --resume, this loads neither the optimizer states "
        "nor the data loader states.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=2.0,
        help="Clip the L2 norm of the gradients to the given value.",
    )
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay for the optimizer.")
    parser.add_argument("--qk_norm", action="store_true")
    parser.add_argument(
        "--caption_dropout_prob",
        type=float,
        default=0.1,
        help="Randomly change the caption of a sample to a blank string with the given probability.",
    )
    parser.add_argument("--snr_type", type=str, default="uniform")
    parser.add_argument("--sample_every", type=int, default=100)
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=10,
        help="Maximum number of checkpoints to keep. Older checkpoints will be deleted.",
    )
    
    args = parser.parse_args()
    main(args)