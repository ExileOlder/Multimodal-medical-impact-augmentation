import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from diffusers.models import AutoencoderKL
from transformers import AutoModelForCausalLM, AutoTokenizer

import models
from transport import create_transport, Sampler
from torchvision.transforms import ToPILImage

import torch.distributed as dist
import fairscale.nn.model_parallel.initialize as fs_init

# ==========================================
# 1. 核心数学模块：无分类器引导 (CFG) 包装器
# ==========================================
class CFGWrapper(nn.Module):
    def __init__(self, model, cfg_scale):
        super().__init__()
        self.model = model
        self.cfg_scale = cfg_scale

    def forward(self, x, t, **kwargs):
        """
        处理拼接后的输入张量 (Batch_size = 2, [Cond, Uncond])
        """
        # 前向传播：同时计算条件与无条件输出
        out = self.model(x, t, **kwargs)
        # 拆分结果
        cond, uncond = out.chunk(2)
        # 执行 CFG 向量计算
        cfg_out = uncond + self.cfg_scale * (cond - uncond)
        
        # === 修复：欺骗 ODE 求解器 ===
        # 将 [1, C, H, W] 沿着 Batch 维度复制成 [2, C, H, W] 
        return torch.cat([cfg_out, cfg_out], dim=0)

# ==========================================
# 2. 文本编码模块 (Gemma 2B -> CPU)
# ==========================================
# ==========================================
# 2. 文本编码模块 (Gemma 2B -> CPU)
# ==========================================
def encode_prompt(prompt, text_encoder, tokenizer, target_length=None):
    with torch.no_grad():
        # 如果提供了 target_length，则强制 Padding 到该长度
        padding_kwargs = {}
        if target_length is not None:
            padding_kwargs = {
                "padding": "max_length",
                "max_length": target_length,
            }
        else:
            padding_kwargs = {
                "padding": True,
                "pad_to_multiple_of": 8,
                "max_length": 256,
            }
            
        text_inputs = tokenizer(
            [prompt], truncation=True, return_tensors="pt", **padding_kwargs
        )
        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask
        prompt_embeds = text_encoder(
            input_ids=text_input_ids, attention_mask=prompt_masks, output_hidden_states=True
        ).hidden_states[-2]
    
    return prompt_embeds.cuda(), prompt_masks.cuda()

# ==========================================
# 3. 主推理逻辑
# ==========================================
@torch.no_grad()
def main(args):
    # 0. 初始化“伪分布式”环境
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", "12345"
        os.environ["RANK"], os.environ["WORLD_SIZE"] = "0", "1"
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    if not fs_init.model_parallel_is_initialized():
        fs_init.initialize_model_parallel(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 核心：自动选择精度，RTX 4070 必须统一使用 bfloat16 或 float32
    inference_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    
    print(f"[*] 初始化基座模型 (精度: {inference_dtype})")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    # 修复：文本编码器必须与模型精度一致
    text_encoder = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b", torch_dtype=inference_dtype
    ).get_decoder().eval()
    
    cap_feat_dim = text_encoder.config.hidden_size
    model = models.__dict__[args.model](
        in_channels=16 if args.vae == "sd3" else 4,
        qk_norm=False,
        cap_feat_dim=cap_feat_dim,
    )
    
    # 加载权重
    print(f"[*] 加载权重并切换设备...")
    base_state_dict = torch.load(os.path.join(args.base_ckpt, "consolidated.00-of-01.pth"), map_location="cpu")
    model.load_state_dict(base_state_dict, strict=False)
    adapter_state_dict = torch.load(os.path.join(args.adapter_ckpt, "adapter.pth"), map_location="cpu")
    model.load_state_dict(adapter_state_dict, strict=False)

    model.to(device, dtype=inference_dtype).eval()

    # 初始化 VAE (删除重复定义，确保精度统一)
    print("[*] 初始化 VAE 解码器...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device, dtype=inference_dtype)
    vae_scale, vae_shift = 0.18215, 0.0

    cfg_model = CFGWrapper(model, args.cfg_scale)

    # 4. 数据预处理
    print(f"[*] 处理文本与掩码...")
    cond_feats, cond_cap_mask = encode_prompt(args.prompt, text_encoder, tokenizer)
    target_seq_len = cond_feats.shape[1]
    uncond_feats, uncond_cap_mask = encode_prompt("", text_encoder, tokenizer, target_length=target_seq_len)

    # 确保掩码张量精度正确
    mask_image = Image.open(args.mask_path).convert("L")
    mask_transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])
    mask_tensor = mask_transform(mask_image).unsqueeze(0).to(device, dtype=inference_dtype)
    
    latent_size = args.image_size // 8
    struct_mask_latent = torch.nn.functional.interpolate(
        mask_tensor, size=(latent_size, latent_size), mode='bilinear', align_corners=False
    )
    zero_mask_latent = torch.zeros_like(struct_mask_latent)

    model_kwargs = {
        "cap_feats": torch.cat([cond_feats, uncond_feats], dim=0).to(inference_dtype),
        "cap_mask": torch.cat([cond_cap_mask, uncond_cap_mask], dim=0),
        "struct_mask": torch.cat([struct_mask_latent, zero_mask_latent], dim=0).to(inference_dtype) # 恢复这一行！
    }

    # 5. ODE 采样
    print("[*] 开始 ODE 采样循环...")
    torch.manual_seed(args.seed)
    z0 = torch.randn(1, 4, latent_size, latent_size, device=device, dtype=inference_dtype)
    z0_batched = torch.cat([z0, z0], dim=0)

    transport = create_transport("Linear", "velocity", None, None, None)
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method="euler", 
        num_steps=args.num_sampling_steps,
        atol=1e-6,
        rtol=1e-3,
        reverse=True  # <--- 核心修复：反向积分（从噪声到数据）
    )
    
    samples_latent = sample_fn(z0_batched, cfg_model, **model_kwargs)[-1]
    final_latent = samples_latent[0:1] 
    print(f"[!] 潜变量诊断 -> Min: {final_latent.min().item():.2f}, Max: {final_latent.max().item():.2f}, Mean: {final_latent.mean().item():.2f}")

    # 6. 解码与保存
    print("[*] VAE 解码...")
    final_latent = final_latent / vae_scale + vae_shift
    decoded_image = vae.decode(final_latent).sample
    decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
    
    os.makedirs(args.out_dir, exist_ok=True)
    import time
    out_path = os.path.join(args.out_dir, f"gen_s{args.seed}_cfg{args.cfg_scale}_{time.strftime('%H%M%S')}.png")
    ToPILImage()(decoded_image[0].cpu().float()).save(out_path)
    print(f"[*] 成功保存至: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="NextDiT_2B_GQA_patch2")
    parser.add_argument("--base_ckpt", type=str, required=True, help="官方 DiT 预训练权重目录")
    parser.add_argument("--adapter_ckpt", type=str, required=True, help="微调生成的 checkpoints/0000010 目录")
    parser.add_argument("--prompt", type=str, required=True, help="控制生成的医学文本提示词")
    parser.add_argument("--mask_path", type=str, required=True, help="输入的医学掩码结构图路径")
    parser.add_argument("--out_dir", type=str, default="./results/inference_output")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_sampling_steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG 权重。越大越贴合文本和掩码")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae", type=str, default="ema")
    
    args = parser.parse_args()
    main(args)

