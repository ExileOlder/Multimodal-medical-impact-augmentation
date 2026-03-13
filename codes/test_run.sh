#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

# 强行关闭 PyTorch 2.0 的静默图编译
export TORCH_COMPILE_DISABLE=1 

# 限制 PyTorch 的显存分配器，防止碎片化导致 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=1 \
    train.py \
    --model NextDiT_2B_GQA_patch2 \
    --data_path ./configs/train/idrid_data.yaml \
    --results_dir ./results/debug_struct_prior \
    --image_size 256 \
    --global_batch_size 1 \
    --micro_batch_size 1 \
    --log_every 1 \
    --ckpt_every 100 \
    --max_steps 1500 \
    --checkpointing \
    --precision bf16 \
    --grad_precision bf16 \
    --lr 0.00002 \
    --num_workers 0 \
    --init_from ./results/pretrained_weights \
    --resume ./results/debug_struct_prior/2026-03-05_00-43-41debugging/checkpoints/0000100