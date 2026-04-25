# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Lumina-T2I using PyTorch FSDP.
"""
import argparse
from collections import OrderedDict, defaultdict
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
import subprocess
from time import time
from PIL import Image
from diffusers.models import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
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
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.logdir = kwargs.get("log_dir")

        def add_scalar(self, *args, **kwargs):
            return None

        def add_image(self, *args, **kwargs):
            return None

        def add_text(self, *args, **kwargs):
            return None

        def flush(self):
            return None

        def close(self):
            return None

from data import ItemProcessor, MyDataset, read_general2
from fundus_geometry import canonicalize_fundus_image
from grad_norm import calculate_l2_grad_norm, get_model_parallel_dim_dict, scale_grad
from imgproc import generate_crop_size_list, var_center_crop
import models
from parallel import distributed_init, get_intra_node_process_group
from struct_mask_utils import (
    augment_caption_with_struct_hints,
    FUSION_PARSER_NUM_CLASSES,
    get_struct_mask_channel_names,
    get_fusion_parser_class_names,
    load_struct_mask_tensor,
    make_empty_struct_mask,
)
from transport import create_transport, Sampler
from tqdm import tqdm 
from torchvision.transforms import ToPILImage

#############################################################################
#                            Data item Processor                            #
#############################################################################


class T2IItemProcessor(ItemProcessor):
    def __init__(
        self,
        transform,
        image_size: int,
        struct_mask_channels: int = 1,
        *,
        fundus_geometry_align: bool = False,
        fundus_padding_ratio: float = 0.01,
    ):
        self.image_transform = transform
        self.image_size = int(image_size)
        self.struct_mask_channels = int(struct_mask_channels)
        self.fundus_geometry_align = bool(fundus_geometry_align)
        self.fundus_padding_ratio = float(fundus_padding_ratio)
    
    def process_item(self, data_item, training_mode=False):
        try:
            image_root = data_item.get("_image_root", None)
            if "caption" in data_item:
                # 淇锛氬皢姝ゅ鐨勫彉閲忓悕涓ユ牸瀹氫箟涓?image_path
                image_path = data_item.get("image") or data_item.get("image_path")
                if not image_path:
                    raise ValueError("Missing 'image_path' key.")
                
                full_image_path = read_general2(image_path, image_root)
                image = Image.open(full_image_path).convert("RGB")
                text = data_item.get("caption", "")

                if self.fundus_geometry_align:
                    image = canonicalize_fundus_image(
                        image,
                        output_size=self.image_size,
                        padding_ratio=self.fundus_padding_ratio,
                    )
                
                mask_val = data_item.get("mask") or data_item.get("mask_path")
                full_mask_path = None
                if mask_val is not None:
                    full_mask_path = read_general2(mask_val, image_root)
                    mask = load_struct_mask_tensor(
                        full_mask_path,
                        image_size=self.image_size,
                        struct_mask_channels=self.struct_mask_channels,
                    )
                else:
                    mask = make_empty_struct_mask(
                        image_size=self.image_size,
                        struct_mask_channels=self.struct_mask_channels,
                    )

                text = augment_caption_with_struct_hints(
                    text,
                    image_path=image_path,
                    mask_path=full_mask_path,
                    struct_mask_channels=self.struct_mask_channels,
                )
                
                image = self.image_transform(image)
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


def masks_to_class_map(mask_tensor: torch.Tensor) -> torch.Tensor:
    if mask_tensor.ndim != 4:
        raise ValueError(f"Expected mask tensor [B,C,H,W], got {tuple(mask_tensor.shape)}")
    has_fg = (mask_tensor > 0.5).any(dim=1)
    class_map = mask_tensor.argmax(dim=1) + 1
    class_map = class_map.where(has_fg, torch.zeros_like(class_map))
    return class_map.long()


def latent_to_image_tensor(
    latent: torch.Tensor,
    vae: AutoencoderKL,
    *,
    vae_scale: float,
    vae_shift: float,
) -> torch.Tensor:
    latent = latent.float() / vae_scale + vae_shift
    decode_dtype = next(vae.parameters()).dtype
    if decode_dtype not in (torch.float16, torch.bfloat16):
        decode_dtype = torch.bfloat16
    with torch.amp.autocast(
        "cuda",
        enabled=latent.is_cuda,
        dtype=decode_dtype,
    ):
        decoded = vae.decode(latent).sample
    return (decoded.float() / 2 + 0.5).clamp(0, 1)


def build_fundus_region_mask(image_01: torch.Tensor) -> torch.Tensor:
    if image_01.ndim != 4:
        raise ValueError(f"Expected image tensor [B,3,H,W], got {tuple(image_01.shape)}")
    luminance = 0.2126 * image_01[:, 0:1] + 0.7152 * image_01[:, 1:2] + 0.0722 * image_01[:, 2:3]
    return (luminance > 0.03).float()


def masked_mean_std(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    mean = (x * mask).sum(dim=(2, 3), keepdim=True) / denom
    var = ((x - mean) ** 2 * mask).sum(dim=(2, 3), keepdim=True) / denom
    std = torch.sqrt(var + 1e-6)
    return mean, std


def rgb_mean_to_chroma(rgb_mean: torch.Tensor) -> torch.Tensor:
    denom = rgb_mean.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return rgb_mean / denom


def rb_gap_from_rgb_mean(rgb_mean: torch.Tensor) -> torch.Tensor:
    return rgb_mean[:, 0:1] - rgb_mean[:, 2:3]


def soft_center_and_area(prob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, _, h, w = prob.shape
    yy = torch.linspace(-1.0, 1.0, steps=h, device=prob.device, dtype=prob.dtype).view(1, 1, h, 1)
    xx = torch.linspace(-1.0, 1.0, steps=w, device=prob.device, dtype=prob.dtype).view(1, 1, 1, w)
    mass = prob.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    cy = (prob * yy).sum(dim=(2, 3), keepdim=True) / mass
    cx = (prob * xx).sum(dim=(2, 3), keepdim=True) / mass
    area = mass / float(h * w)
    return torch.cat([cx, cy], dim=1).flatten(1), area.flatten(1)


def dice_loss_multiclass(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    dice = (2.0 * intersection + 1e-6) / (denom + 1e-6)
    return 1.0 - dice[1:].mean()


def dice_loss_for_class_subset(
    probs: torch.Tensor,
    target: torch.Tensor,
    class_indices: list[int],
) -> torch.Tensor:
    one_hot = torch.nn.functional.one_hot(target, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
    losses = []
    for cls_idx in class_indices:
        pred_cls = probs[:, cls_idx : cls_idx + 1]
        target_cls = one_hot[:, cls_idx : cls_idx + 1]
        intersection = (pred_cls * target_cls).sum(dim=(1, 2, 3))
        denom = pred_cls.sum(dim=(1, 2, 3)) + target_cls.sum(dim=(1, 2, 3))
        losses.append(1.0 - ((2.0 * intersection + 1e-6) / (denom + 1e-6)))
    return torch.stack(losses, dim=0).mean()


def _annotation_to_dict(annotation) -> dict:
    if isinstance(annotation, dict):
        return annotation
    if isinstance(annotation, bytes):
        annotation = annotation.decode("utf-8")
    if isinstance(annotation, str):
        return json.loads(annotation)
    return dict(annotation)


def build_balanced_field_epoch(groups: dict[str, list[int]], generator: torch.Generator) -> torch.Tensor:
    max_count = max(len(indices) for indices in groups.values())
    chunks = []
    for key in sorted(groups):
        values = torch.tensor(groups[key], dtype=torch.long)
        if values.numel() >= max_count:
            choice = values[torch.randperm(values.numel(), generator=generator)[:max_count]]
        else:
            choice = values[torch.randint(values.numel(), (max_count,), generator=generator)]
        chunks.append(choice)
    epoch_indices = torch.cat(chunks, dim=0)
    return epoch_indices[torch.randperm(epoch_indices.numel(), generator=generator)]


def get_train_sampler(
    dataset,
    rank,
    world_size,
    global_batch_size,
    max_steps,
    resume_step,
    seed,
    balance_field: str | None = None,
    logger: logging.Logger | None = None,
):
    if balance_field:
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, annotation in enumerate(dataset.ann):
            record = _annotation_to_dict(annotation)
            value = str(record.get(balance_field) or "__missing__")
            groups[value].append(idx)
        if len(groups) < 2:
            if logger is not None:
                logger.warning(
                    f"Balance field {balance_field!r} has fewer than 2 groups; falling back to uniform shuffling."
                )
        else:
            if logger is not None:
                logger.info(
                    f"Using balanced sampler over field={balance_field}: "
                    + ", ".join(f"{key}={len(groups[key])}" for key in sorted(groups))
                )
            global_sample_indices = torch.empty([max_steps * global_batch_size], dtype=torch.long)
            epoch_id, fill_ptr = 0, 0
            while fill_ptr < global_sample_indices.size(0):
                g = torch.Generator()
                g.manual_seed(seed + epoch_id)
                epoch_sample_indices = build_balanced_field_epoch(groups, g)
                epoch_id += 1
                epoch_sample_indices = epoch_sample_indices[: global_sample_indices.size(0) - fill_ptr]
                global_sample_indices[fill_ptr : fill_ptr + epoch_sample_indices.size(0)] = epoch_sample_indices
                fill_ptr += epoch_sample_indices.size(0)

            sample_indices = global_sample_indices[rank::world_size]
            return sample_indices[resume_step * global_batch_size // world_size :].tolist()

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


def get_git_snapshot(repo_root: str) -> dict:
    snapshot = {
        "repo_root": repo_root,
        "commit": None,
        "branch": None,
        "dirty": None,
    }
    try:
        snapshot["commit"] = (
            subprocess.check_output(["git", "-C", repo_root, "rev-parse", "HEAD"], text=True).strip()
        )
        snapshot["branch"] = (
            subprocess.check_output(["git", "-C", repo_root, "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        )
        status_output = subprocess.check_output(["git", "-C", repo_root, "status", "--short"], text=True).strip()
        snapshot["dirty"] = bool(status_output)
    except Exception:
        pass
    return snapshot


def write_run_manifest(manifest_path: str, args: argparse.Namespace, experiment_dir: str, checkpoint_dir: str, text_model_path: str):
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    preflight_path = os.path.join(os.path.dirname(__file__), "results", "preflight", "training_ready_report.json")
    manifest = {
        "created_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment_dir": experiment_dir,
        "checkpoint_dir": checkpoint_dir,
        "results_dir": args.results_dir,
        "data_path": args.data_path,
        "init_from": args.init_from,
        "resume": args.resume,
        "model": args.model,
        "vae": args.vae,
        "trainable_strategy": args.trainable_strategy,
        "image_size": args.image_size,
        "global_batch_size": args.global_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "precision": args.precision,
        "text_model_path": text_model_path,
        "preflight_report": preflight_path if os.path.exists(preflight_path) else None,
        "git": get_git_snapshot(workspace_root),
    }
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)


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
            input_ids=text_input_ids.cuda(), attention_mask=prompt_masks.cuda(), output_hidden_states=True
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


def unwrap_and_clean_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    for key in ("model", "state_dict", "module"):
        if key in state_dict and isinstance(state_dict[key], dict):
            state_dict = state_dict[key]

    clean_state_dict = {}
    for key, value in state_dict.items():
        k = key.replace("_fsdp_wrapped_module.", "")
        k = k.replace("_checkpoint_wrapped_module.", "")
        clean_state_dict[k] = value

    return clean_state_dict


def configure_trainable_parameters(model: nn.Module, args: argparse.Namespace, logger: logging.Logger):
    """
    Configure trainable parameters for lightweight adaptation.

    Strategy summary:
    - mask_only: mask latent adapter + gate
    - mask_plus_x: mask latent adapter + x_embedder
    - mask_plus_y_norm: mask latent adapter + attention_y_norm in all blocks
    - mask_plus_x_y_norm: mask latent adapter + x_embedder + attention_y_norm
    - mask_plus_y_norm_capln: mask latent adapter + attention_y_norm + cap_embedder LayerNorm
    - mask_plus_cap: mask latent adapter + attention_y_norm + full cap_embedder
    - struct_strict: all mask adapters + x_embedder + attention_y_norm + full cap_embedder + final_layer
    """
    for _, param in model.named_parameters():
        param.requires_grad = False

    trainable_names = []
    strategy = args.trainable_strategy

    def enable(name: str, param: torch.nn.Parameter):
        param.requires_grad = True
        trainable_names.append(name)

    for name, param in model.named_parameters():
        if name.startswith("mask_"):
            enable(name, param)
            continue

        if strategy in {"mask_plus_x", "mask_plus_x_y_norm", "struct_strict"} and name.startswith("x_embedder."):
            enable(name, param)
            continue

        if strategy in {
            "mask_plus_y_norm",
            "mask_plus_x_y_norm",
            "mask_plus_y_norm_capln",
            "mask_plus_cap",
            "struct_strict",
        } and ".attention_y_norm." in name:
            enable(name, param)
            continue

        if strategy in {"mask_plus_y_norm_capln", "mask_plus_cap", "struct_strict"} and name.startswith("cap_embedder.0."):
            enable(name, param)
            continue

        if strategy in {"mask_plus_cap", "struct_strict"} and name.startswith("cap_embedder.1."):
            enable(name, param)
            continue

        if strategy == "struct_strict" and name.startswith("final_layer."):
            enable(name, param)
            continue

    trainable_name_set = set(trainable_names)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_numel = sum(p.numel() for p in trainable_params)

    module_counter = {}
    for name in trainable_names:
        module_name = name.split(".")[0]
        module_counter[module_name] = module_counter.get(module_name, 0) + 1

    logger.info(
        f"Trainable strategy={strategy}, train_mask_bias={args.train_mask_bias}, "
        f"trainable_tensors={len(trainable_names)}, trainable_params={trainable_numel:,}"
    )
    if args.train_mask_bias:
        logger.warning("--train_mask_bias is a legacy flag and is ignored by the latent adapter path.")
    if module_counter:
        logger.info(f"Trainable module breakdown: {module_counter}")

    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters selected. Please check --trainable_strategy.")

    return trainable_params, trainable_name_set


def build_optimizer_param_groups(
    model: nn.Module,
    trainable_name_set: set[str],
    args: argparse.Namespace,
    logger: logging.Logger,
):
    mask_lr = args.mask_adapter_lr if args.mask_adapter_lr is not None else args.lr
    cap_embedder_lr = args.cap_embedder_lr if args.cap_embedder_lr is not None else args.lr
    base_lr = args.lr

    mask_params = []
    cap_embedder_params = []
    base_params = []
    mask_names = []
    cap_embedder_names = []
    base_names = []

    for name, param in model.named_parameters():
        clean_name = name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
        if clean_name not in trainable_name_set or not param.requires_grad:
            continue
        if clean_name.startswith("mask_"):
            mask_params.append(param)
            mask_names.append(clean_name)
        elif clean_name.startswith("cap_embedder."):
            cap_embedder_params.append(param)
            cap_embedder_names.append(clean_name)
        else:
            base_params.append(param)
            base_names.append(clean_name)

    param_groups = []
    if mask_params:
        param_groups.append(
            {
                "params": mask_params,
                "lr": mask_lr,
                "weight_decay": args.wd,
                "group_name": "mask_adapter",
            }
        )
    if cap_embedder_params:
        param_groups.append(
            {
                "params": cap_embedder_params,
                "lr": cap_embedder_lr,
                "weight_decay": args.wd,
                "group_name": "cap_embedder",
            }
        )
    if base_params:
        param_groups.append(
            {
                "params": base_params,
                "lr": base_lr,
                "weight_decay": args.wd,
                "group_name": "base_trainables",
            }
        )

    logger.info(
        "Optimizer param groups: "
        f"mask_adapter={len(mask_names)} tensors @ lr={mask_lr:.6g}, "
        f"cap_embedder={len(cap_embedder_names)} tensors @ lr={cap_embedder_lr:.6g}, "
        f"base_trainables={len(base_names)} tensors @ lr={base_lr:.6g}"
    )
    return param_groups


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

    text_model_path = os.path.join(os.path.dirname(__file__), "google_gemma-2b")
    manifest_path = os.path.join(time_based_dir, "run_manifest.json")
    write_run_manifest(
        manifest_path=manifest_path,
        args=args,
        experiment_dir=time_based_dir,
        checkpoint_dir=checkpoint_dir,
        text_model_path=text_model_path,
    )

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
    logger.info(f"Run manifest: {manifest_path}")
    logger.info(f"Setting up language model: {text_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(text_model_path)
    tokenizer.padding_side = "right"

    text_encoder = (
        AutoModelForCausalLM.from_pretrained(
            text_model_path,
            torch_dtype=torch.bfloat16,
        )
        .get_decoder()
    ).to(device)
    text_encoder.eval()
    cap_feat_dim = text_encoder.config.hidden_size
        
    if args.model in ["NextDiT_2B_patch2", "DiT_Llama2_7B_patch2", "NextDiT_2B_GQA_patch2"]:
        model = models.__dict__[args.model](
            in_channels=16 if args.vae == "sd3" else 4,
            qk_norm=args.qk_norm,
            cap_feat_dim=cap_feat_dim,
            struct_mask_channels=args.struct_mask_channels,
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

    # Load base DiT weights (pretrained backbone).
    if args.init_from:
        logger.info(f"Loading Base DiT weights from: {args.init_from}")
        base_ckpt_path = os.path.join(args.init_from, "consolidated.00-of-01.pth")
        if os.path.exists(base_ckpt_path):
            base_state_dict = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
            base_state_dict = unwrap_and_clean_state_dict(base_state_dict)
            missing, unexpected = model.load_state_dict(base_state_dict, strict=False)
            logger.info(f"Base weights loaded. missing={len(missing)}, unexpected={len(unexpected)}")
            if missing:
                logger.info(f"Missing keys (first 20): {missing[:20]}")
            if unexpected:
                logger.warning(f"Unexpected keys (first 20): {unexpected[:20]}")

            qnorm_unexpected = [
                k for k in unexpected if (".q_norm." in k or ".k_norm." in k or ".ky_norm." in k)
            ]
            if qnorm_unexpected:
                raise RuntimeError(
                    "Checkpoint architecture mismatch detected: q_norm/k_norm keys were not loaded. "
                    "This usually means the base checkpoint expects qk_norm=True. "
                    "Please relaunch training with --qk_norm."
                )

            # Zero-init mask branch only for fresh training (not when resuming).
            if not args.resume:
                logger.info("Resetting mask latent adapter to zero-impact initialization...")
                if hasattr(model, "reset_mask_conditioning"):
                    model.reset_mask_conditioning()

    # Load adapter weights when resuming from a checkpoint.
    if args.resume:
        logger.info(f"Loading Adapter weights from: {args.resume}")
        adapter_ckpt_path = os.path.join(args.resume, "adapter.pth")
        if os.path.exists(adapter_ckpt_path):
            adapter_state_dict = torch.load(adapter_ckpt_path, map_location="cpu")
            adapter_state_dict = unwrap_and_clean_state_dict(adapter_state_dict)
            missing, unexpected = model.load_state_dict(adapter_state_dict, strict=False)
            logger.info(
                f"Adapter weights loaded. missing={len(missing)}, unexpected={len(unexpected)}"
            )

    # === 鏋佺畝鐗堝垵濮嬪寲 ===
    model = model.to(device)
    
    if args.checkpointing:
        print("Applying gradient checkpointing")
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            # 灏?NO_REENTRANT 鏀逛负 REENTRANT锛岃繖浼氫互鐣ュ井澧炲姞璁＄畻鏃堕棿涓轰唬浠凤紝
            # 鏋佸害涓斾弗鏍煎湴鍘嬫Θ涓棿婵€娲绘樉瀛樸€?            checkpoint_impl=CheckpointImpl.REENTRANT, 
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in list(model.layers),
        )
    logger.info(f"Model:\n{model}\n")

    transport = create_transport("Linear", "velocity", None, None, None, snr_type=args.snr_type)
    vae_weight_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "tf32": torch.float32,
    }[args.precision]
    
    if args.vae == "sd3":
        logger.info("Using SD3 VAE")
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            subfolder="vae",
            torch_dtype=vae_weight_dtype,
        ).to(device=device, dtype=vae_weight_dtype)
    elif args.vae == "sdxl":
        logger.info("Using SDXL VAE")
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sdxl-vae",
            torch_dtype=vae_weight_dtype,
        ).to(device=device, dtype=vae_weight_dtype)
    else:
        vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{args.vae}"
            if args.local_diffusers_model_root is None
            else os.path.join(args.local_diffusers_model_root, f"stabilityai/sd-vae-ft-{args.vae}"),
            torch_dtype=vae_weight_dtype,
        ).to(device=device, dtype=vae_weight_dtype)
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()

    parser_model = None
    parser_ce_weight = None
    parser_enabled = (
        args.parser_ckpt is not None
        and (
            args.parser_loss_weight > 0.0
            or args.parser_dice_weight > 0.0
            or args.optic_disc_weight > 0.0
            or args.vessel_weight > 0.0
            or args.lesion_weight > 0.0
        )
    )
    if parser_enabled:
        parser_ckpt = args.parser_ckpt
        if os.path.isdir(parser_ckpt):
            candidate = os.path.join(parser_ckpt, "parser_best.pt")
            if os.path.exists(candidate):
                parser_ckpt = candidate
            else:
                parser_ckpt = os.path.join(parser_ckpt, "parser_last.pt")
        logger.info(f"Loading frozen fusion parser from: {parser_ckpt}")
        parser_payload = torch.load(parser_ckpt, map_location="cpu")
        parser_state = parser_payload.get("model", parser_payload)
        parser_model = models.FusionMaskParser(
            num_classes=FUSION_PARSER_NUM_CLASSES,
            base_channels=args.parser_base_channels,
        ).to(device)
        parser_model.load_state_dict(parser_state, strict=True)
        parser_model.eval()
        for param in parser_model.parameters():
            param.requires_grad = False
        parser_ce_weight = torch.ones(FUSION_PARSER_NUM_CLASSES, device=device)
        parser_ce_weight[0] = 0.25
        logger.info(
            "Fusion parser ready with classes: "
            f"{get_fusion_parser_class_names()}"
        )

    trainable_params, trainable_name_set = configure_trainable_parameters(model, args, logger)
    optimizer_param_groups = build_optimizer_param_groups(model, trainable_name_set, args, logger)
    opt = torch.optim.AdamW(optimizer_param_groups, lr=args.lr, weight_decay=args.wd)

    if args.resume:
        if args.resume_adapter_only:
            logger.info(
                "Resume adapter-only mode is enabled; "
                f"skipping optimizer state restore from: {args.resume}"
            )
        else:
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
        # 淇锛氬姞杞介璁粌鏉冮噸鏃讹紝鐩存帴灏嗘鏁拌涓?0锛屼笉寮烘眰瀛樺湪 resume_step.txt
        resume_step = 0
    else:
        resume_step = 0

    logger.info(f"Resume step: {resume_step}")
    
    if args.fundus_geometry_align:
        logger.info(
            "Creating canonical fundus-square transforms for real images "
            f"(padding_ratio={args.fundus_padding_ratio:.4f})..."
        )
        transform_ops = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    else:
        logger.info("Creating direct square-resize transforms for image and structural mask...")
        transform_ops = [
            transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    image_transform = transforms.Compose(transform_ops)
    logger.info(
        "Structural mask channels: "
        f"{args.struct_mask_channels} -> {get_struct_mask_channel_names(args.struct_mask_channels)}"
    )

    dataset = MyDataset(
        args.data_path,
        item_processor=T2IItemProcessor(
            image_transform,
            image_size=args.image_size,
            struct_mask_channels=args.struct_mask_channels,
            fundus_geometry_align=args.fundus_geometry_align,
            fundus_padding_ratio=args.fundus_padding_ratio,
        ),
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
        balance_field=args.balance_field,
        logger=logger,
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
    if parser_model is not None:
        parser_model.eval()

    extra_supervision_enabled = parser_model is not None or any(
        weight > 0.0
        for weight in (
            args.color_l1_weight,
            args.luma_loss_weight,
            args.color_stat_weight,
            args.color_chroma_weight,
            args.rb_gap_weight,
        )
    )

    log_steps = 0
    running_loss = 0
    running_grad_norm = 0
    start_time = time()
    
    pbar = initialize_progress_bar(total_steps=len(loader))

    logger.info(f"Training for {args.max_steps:,} steps...")
    for step, (x, masks, caps) in enumerate(loader, start=resume_step):
        # 寮哄埗灏?List 鍫嗗彔涓哄舰鐘朵竴鑷寸殑寮犻噺 [B, C, H, W]
        x_tensor = torch.stack(x).to(device, non_blocking=True)
        masks_tensor = torch.stack(masks).to(device, non_blocking=True)
        target_class_tensor = masks_to_class_map(masks_tensor)
        x_image_01 = ((x_tensor.float() + 1.0) / 2.0).clamp(0, 1)
        fundus_region_mask = build_fundus_region_mask(x_image_01)
         
        with torch.no_grad():
            vae_scale = {"sdxl": 0.13025, "sd3": 1.5305, "ema": 0.18215, "mse": 0.18215}[args.vae]
            vae_shift = {"sdxl": 0.0, "sd3": 0.0609, "ema": 0.0, "mse": 0.0}[args.vae]
            vae_dtype = next(vae.parameters()).dtype
            if step == resume_step:
                logger.warning(f"VAE scale: {vae_scale}, VAE shift: {vae_shift}")
            
            # === 闃插脊琛ｏ細鍒嗗潡杩涜 VAE 缂栫爜锛屾秷鐏?8GB 鏄惧瓨宄板€?===
            x_latent_list = []
            for i in range(0, local_batch_size, args.micro_batch_size):
                chunk = x_tensor[i:i+args.micro_batch_size].to(dtype=vae_dtype)
                encoded_chunk = (vae.encode(chunk).latent_dist.sample() - vae_shift) * vae_scale
                x_latent_list.append(encoded_chunk)
            x_latent = torch.cat(x_latent_list, dim=0)
            # ========================

        with torch.no_grad():
            cap_feats, cap_mask = encode_prompt(caps, text_encoder, tokenizer, args.caption_dropout_prob)
            
            # Use masks_tensor directly and resize it to latent spatial size.
            target_h, target_w = x_latent.shape[2], x_latent.shape[3]
            masks_latent = torch.nn.functional.interpolate(
                masks_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False
            )
            masks_latent = masks_latent * float(args.mask_scale)
            if step == resume_step:
                nonzero_ratio = (masks_latent > 1e-6).float().mean().item()
                logger.info(
                    f"Mask latent stats: min={masks_latent.min().item():.6f}, "
                    f"max={masks_latent.max().item():.6f}, mean={masks_latent.mean().item():.6f}, "
                    f"nonzero_ratio={nonzero_ratio:.6f}, mask_scale={args.mask_scale}, "
                    f"channels={masks_latent.shape[1]}"
                )

        loss_item = 0.0
        opt.zero_grad()
        torch.cuda.empty_cache()
        for mb_idx in range((local_batch_size - 1) // args.micro_batch_size + 1):
            mb_st = mb_idx * args.micro_batch_size
            mb_ed = min((mb_idx + 1) * args.micro_batch_size, local_batch_size)
            last_mb = mb_ed == local_batch_size

            # === 璇︾粏娉ㄩ噴锛氭彁鍙栧綋鍓嶅井鎵规鐨勫彉閲忓垏鐗?===
            x_mb = x_latent[mb_st:mb_ed] 
            cap_feats_mb = cap_feats[mb_st:mb_ed]
            cap_mask_mb = cap_mask[mb_st:mb_ed]
            masks_latent_mb = masks_latent[mb_st:mb_ed]
            
            model_kwargs = dict(cap_feats=cap_feats_mb, cap_mask=cap_mask_mb, struct_mask=masks_latent_mb)
            
            # Mixed precision forward.
            with {
                "bf16": torch.amp.autocast("cuda", dtype=torch.bfloat16),
                "fp16": torch.amp.autocast("cuda", dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.precision]:
                loss_dict = transport.training_losses(model, x_mb, model_kwargs, return_extra=extra_supervision_enabled)

            loss = loss_dict["loss"].sum() / local_batch_size

            if extra_supervision_enabled:
                model_output = loss_dict["model_output"]
                xt = loss_dict["xt"]
                t_mb = loss_dict["t"]
                t_view = t_mb.view(t_mb.size(0), 1, 1, 1)
                x1_pred = xt + (1.0 - t_view) * model_output
                pred_image = latent_to_image_tensor(
                    x1_pred,
                    vae,
                    vae_scale=vae_scale,
                    vae_shift=vae_shift,
                )
                gt_image = x_image_01[mb_st:mb_ed]
                gt_class = target_class_tensor[mb_st:mb_ed]
                gt_fundus_mask = fundus_region_mask[mb_st:mb_ed]

                if parser_model is not None:
                    parser_logits = parser_model(pred_image)
                    parser_ce = F.cross_entropy(parser_logits, gt_class, weight=parser_ce_weight)
                    parser_dice = dice_loss_multiclass(parser_logits, gt_class, FUSION_PARSER_NUM_CLASSES)
                    loss = loss + args.parser_loss_weight * parser_ce + args.parser_dice_weight * parser_dice

                    parser_probs = torch.softmax(parser_logits, dim=1)
                    vessel_prob = parser_probs[:, 1:2]
                    disc_prob = parser_probs[:, 2:3]
                    target_vessel = (gt_class == 1).float().unsqueeze(1)
                    target_disc = (gt_class == 2).float().unsqueeze(1)

                    if args.vessel_weight > 0.0:
                        vessel_intersection = (vessel_prob * target_vessel).sum(dim=(1, 2, 3))
                        vessel_denom = vessel_prob.sum(dim=(1, 2, 3)) + target_vessel.sum(dim=(1, 2, 3))
                        vessel_dice_loss = 1.0 - ((2.0 * vessel_intersection + 1e-6) / (vessel_denom + 1e-6))
                        loss = loss + args.vessel_weight * vessel_dice_loss.mean()

                    if args.lesion_weight > 0.0:
                        lesion_dice_loss = dice_loss_for_class_subset(
                            parser_probs,
                            gt_class,
                            class_indices=[3, 4, 5, 6],
                        )
                        loss = loss + args.lesion_weight * lesion_dice_loss

                    if args.optic_disc_weight > 0.0:
                        disc_present = (target_disc.sum(dim=(1, 2, 3)) > 0).float()
                        pred_center, pred_area = soft_center_and_area(disc_prob)
                        gt_center, gt_area = soft_center_and_area(target_disc)
                        center_loss = torch.abs(pred_center - gt_center).mean(dim=1)
                        area_loss = torch.abs(pred_area - gt_area).mean(dim=1)
                        disc_loss = ((center_loss + area_loss) * disc_present).sum() / disc_present.sum().clamp_min(1.0)
                        loss = loss + args.optic_disc_weight * disc_loss

                if args.color_l1_weight > 0.0:
                    color_l1 = (
                        (pred_image - gt_image).abs() * gt_fundus_mask
                    ).sum() / gt_fundus_mask.sum().clamp_min(1.0) / pred_image.shape[1]
                    loss = loss + args.color_l1_weight * color_l1

                if args.luma_loss_weight > 0.0:
                    pred_luma = 0.2126 * pred_image[:, 0:1] + 0.7152 * pred_image[:, 1:2] + 0.0722 * pred_image[:, 2:3]
                    gt_luma = 0.2126 * gt_image[:, 0:1] + 0.7152 * gt_image[:, 1:2] + 0.0722 * gt_image[:, 2:3]
                    luma_l1 = ((pred_luma - gt_luma).abs() * gt_fundus_mask).sum() / gt_fundus_mask.sum().clamp_min(1.0)
                    loss = loss + args.luma_loss_weight * luma_l1

                if args.color_stat_weight > 0.0:
                    pred_mean, pred_std = masked_mean_std(pred_image, gt_fundus_mask)
                    gt_mean, gt_std = masked_mean_std(gt_image, gt_fundus_mask)
                    color_stats = (pred_mean - gt_mean).abs().mean() + (pred_std - gt_std).abs().mean()
                    loss = loss + args.color_stat_weight * color_stats

                if args.color_chroma_weight > 0.0 or args.rb_gap_weight > 0.0:
                    if args.color_stat_weight <= 0.0:
                        pred_mean, _ = masked_mean_std(pred_image, gt_fundus_mask)
                        gt_mean, _ = masked_mean_std(gt_image, gt_fundus_mask)

                    if args.color_chroma_weight > 0.0:
                        pred_chroma = rgb_mean_to_chroma(pred_mean)
                        gt_chroma = rgb_mean_to_chroma(gt_mean)
                        chroma_l1 = (pred_chroma - gt_chroma).abs().mean()
                        loss = loss + args.color_chroma_weight * chroma_l1

                    if args.rb_gap_weight > 0.0:
                        pred_rb_gap = rb_gap_from_rgb_mean(pred_mean)
                        gt_rb_gap = rb_gap_from_rgb_mean(gt_mean)
                        rb_gap_l1 = (pred_rb_gap - gt_rb_gap).abs().mean()
                        loss = loss + args.rb_gap_weight * rb_gap_l1

            loss_item += loss.item()
            
            loss.backward()

        # 浣跨敤 PyTorch 鍘熺敓姊害瑁佸壀锛堟浛浠ｅ師鏈夊鏉傚嚱鏁帮級
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
            # Save only selected trainable tensors to keep checkpoints lightweight.
            adapter_weights = {}
            for key, val in model.state_dict().items():
                clean_key = key.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
                if clean_key in trainable_name_set:
                    adapter_weights[clean_key] = val.detach().cpu()
            torch.save(
                {
                    "state_dict": adapter_weights,
                    "trainable_strategy": args.trainable_strategy,
                    "train_mask_bias": args.train_mask_bias,
                    "trainable_names": sorted(adapter_weights.keys()),
                },
                os.path.join(checkpoint_path, "adapter.pth"),
            )
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
        "--mask_adapter_lr",
        type=float,
        default=None,
        help="Optional higher learning rate applied only to mask_latent_adapter parameters.",
    )
    parser.add_argument(
        "--cap_embedder_lr",
        type=float,
        default=None,
        help="Optional higher learning rate applied only to cap_embedder parameters.",
    )
    parser.add_argument(
        "--no_auto_resume",
        action="store_false",
        dest="auto_resume",
        help="Do NOT auto resume from the last checkpoint in --results_dir.",
    )
    parser.add_argument("--resume", type=str, help="Resume training from a checkpoint folder.")
    parser.add_argument(
        "--resume_adapter_only",
        action="store_true",
        default=False,
        help="Load adapter weights and resume step from --resume, but skip optimizer state restore.",
    )
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
    parser.add_argument(
        "--mask_scale",
        type=float,
        default=1.0,
        help="Scale factor applied to structural mask latent before fusion.",
    )
    parser.add_argument(
        "--struct_mask_channels",
        type=int,
        default=1,
        help="Number of structural mask channels. Use 1 for grayscale/binary masks or 6 for fusion masks.",
    )
    parser.add_argument(
        "--fundus_geometry_align",
        action="store_true",
        default=False,
        help="Canonicalize rectangular fundus photos into a square fundus crop instead of stretching them.",
    )
    parser.add_argument(
        "--fundus_padding_ratio",
        type=float,
        default=0.01,
        help="Extra margin kept around the detected fundus circle when --fundus_geometry_align is enabled.",
    )
    parser.add_argument(
        "--trainable_strategy",
        type=str,
        choices=[
            "mask_only",
            "mask_plus_x",
            "mask_plus_y_norm",
            "mask_plus_x_y_norm",
            "mask_plus_y_norm_capln",
            "mask_plus_cap",
            "struct_strict",
        ],
        default="mask_only",
        help=(
            "Select which lightweight modules to train. "
            "mask_only trains the mask adapters only; other options increase capacity, "
            "with struct_strict providing the strongest spatial/domain adaptation."
        ),
    )
    parser.add_argument(
        "--train_mask_bias",
        action="store_true",
        help="Legacy flag kept for CLI compatibility. It is ignored by the latent adapter path.",
    )
    parser.add_argument("--parser_ckpt", type=str, default=None, help="Frozen fusion parser checkpoint for anatomy consistency.")
    parser.add_argument("--parser_base_channels", type=int, default=32, help="Base channels used by the fusion parser architecture.")
    parser.add_argument("--parser_loss_weight", type=float, default=0.0, help="Cross-entropy weight for parser consistency.")
    parser.add_argument("--parser_dice_weight", type=float, default=0.0, help="Dice loss weight for parser consistency.")
    parser.add_argument("--optic_disc_weight", type=float, default=0.0, help="Extra optic disc center/area consistency weight.")
    parser.add_argument("--vessel_weight", type=float, default=0.0, help="Extra vessel consistency weight.")
    parser.add_argument("--lesion_weight", type=float, default=0.0, help="Extra lesion-class consistency weight for the 4 lesion colors.")
    parser.add_argument("--color_l1_weight", type=float, default=0.0, help="Fundus RGB L1 supervision weight.")
    parser.add_argument("--luma_loss_weight", type=float, default=0.0, help="Fundus luminance supervision weight.")
    parser.add_argument("--color_stat_weight", type=float, default=0.0, help="Fundus color mean/std supervision weight.")
    parser.add_argument("--color_chroma_weight", type=float, default=0.0, help="Fundus RGB chromaticity supervision weight.")
    parser.add_argument("--rb_gap_weight", type=float, default=0.0, help="Fundus red-blue mean gap supervision weight.")
    parser.add_argument("--snr_type", type=str, default="uniform")
    parser.add_argument("--sample_every", type=int, default=100)
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=10,
        help="Maximum number of checkpoints to keep. Older checkpoints will be deleted.",
    )
    parser.add_argument(
        "--balance_field",
        type=str,
        default=None,
        help="Optional metadata field to balance during sampling, e.g. color_style_code.",
    )
    
    args = parser.parse_args()
    main(args)
