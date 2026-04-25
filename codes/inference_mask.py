import argparse
import contextlib
import json
import logging
import os
import random
import socket
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
from PIL import Image
from PIL import ImageFilter
from torchvision import transforms
from torchvision.transforms import ToPILImage
from transformers import AutoModelForCausalLM, AutoTokenizer

import models
from struct_mask_utils import load_struct_mask_tensor
from transport import Sampler, create_transport

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


def pick_free_port() -> int:
    for _ in range(50):
        port = random.randint(12000, 65000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def init_single_process_dist() -> None:
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        preferred_port = os.environ.get("MASTER_PORT")
        tried_ports = []
        last_err = None

        for attempt in range(8):
            if attempt == 0 and preferred_port:
                port = int(preferred_port)
            else:
                port = pick_free_port()
            os.environ["MASTER_PORT"] = str(port)

            try:
                dist.init_process_group(backend=backend)
                if attempt > 0:
                    print(f"[WARN] MASTER_PORT conflict detected; switched to free port {port}.")
                break
            except RuntimeError as err:
                last_err = err
                msg = str(err)
                port_in_use = (
                    "EADDRINUSE" in msg
                    or "address already in use" in msg.lower()
                    or "failed to listen" in msg.lower()
                )
                if not port_in_use:
                    raise
                tried_ports.append(port)
                if dist.is_initialized():
                    dist.destroy_process_group()
        else:
            raise RuntimeError(
                f"Failed to initialize distributed process group due to port conflicts. Tried ports: {tried_ports}"
            ) from last_err

    if not fs_init.model_parallel_is_initialized():
        fs_init.initialize_model_parallel(1)


def unwrap_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    for key in ("model", "state_dict", "module"):
        if key in state_dict and isinstance(state_dict[key], dict):
            state_dict = state_dict[key]
    return state_dict


def clean_state_dict_prefixes(state_dict):
    clean = {}
    for key, value in state_dict.items():
        k = key.replace("_fsdp_wrapped_module.", "")
        k = k.replace("_checkpoint_wrapped_module.", "")
        clean[k] = value
    return clean


def load_robust(model: torch.nn.Module, ckpt_path: str, required: bool = True):
    if not ckpt_path or not os.path.exists(ckpt_path):
        if required:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"[WARN] Skip missing checkpoint: {ckpt_path}")
        return [], []

    print(f"[*] Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = unwrap_state_dict(state_dict)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint format is not a dict: {type(state_dict)}")

    state_dict = clean_state_dict_prefixes(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"    -> done. missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"    -> missing (first 8): {missing[:8]}")
    if unexpected:
        print(f"    -> unexpected (first 8): {unexpected[:8]}")
        if any(".q_norm." in k or ".k_norm." in k or ".ky_norm." in k for k in unexpected):
            print(
                "    [HINT] q_norm/k_norm keys are unexpected. "
                "This usually means qk_norm flag does not match the checkpoint architecture."
            )
    return missing, unexpected


def resolve_ckpt_path(path_or_dir: str, default_filename: str) -> str:
    if os.path.isdir(path_or_dir):
        return os.path.join(path_or_dir, default_filename)
    return path_or_dir


def looks_like_local_vae(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    has_config = os.path.exists(os.path.join(path, "config.json"))
    has_weights = any(
        os.path.exists(os.path.join(path, name))
        for name in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin")
    )
    return has_config and has_weights


def get_vae_config(vae_name: str, local_diffusers_model_root: str | None = None):
    vae_scale = {"sdxl": 0.13025, "sd3": 1.5305, "ema": 0.18215, "mse": 0.18215}[vae_name]
    vae_shift = {"sdxl": 0.0, "sd3": 0.0609, "ema": 0.0, "mse": 0.0}[vae_name]

    if local_diffusers_model_root is not None:
        local_root = os.path.expanduser(local_diffusers_model_root)
        if looks_like_local_vae(local_root):
            return {
                "path": local_root,
                "subfolder": None,
                "scale": vae_scale,
                "shift": vae_shift,
            }
        local_candidates = [
            os.path.join(local_root, "sdxl-vae"),
            os.path.join(local_root, "stabilityai", "sdxl-vae"),
            os.path.join(local_root, f"sd-vae-ft-{vae_name}"),
            os.path.join(local_root, "stabilityai", f"sd-vae-ft-{vae_name}"),
        ]
        for candidate in local_candidates:
            if looks_like_local_vae(candidate):
                return {
                    "path": candidate,
                    "subfolder": None,
                    "scale": vae_scale,
                    "shift": vae_shift,
                }

    if vae_name == "sd3":
        model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        model_path = (
            model_id
            if local_diffusers_model_root is None
            else os.path.join(local_diffusers_model_root, model_id)
        )
        return {
            "path": model_path,
            "subfolder": "vae",
            "scale": vae_scale,
            "shift": vae_shift,
        }

    if vae_name == "sdxl":
        model_id = "stabilityai/sdxl-vae"
        model_path = (
            model_id
            if local_diffusers_model_root is None
            else os.path.join(local_diffusers_model_root, model_id)
        )
        return {
            "path": model_path,
            "subfolder": None,
            "scale": vae_scale,
            "shift": vae_shift,
        }

    model_id = f"stabilityai/sd-vae-ft-{vae_name}"
    model_path = (
        model_id
        if local_diffusers_model_root is None
        else os.path.join(local_diffusers_model_root, model_id)
    )
    return {
        "path": model_path,
        "subfolder": None,
        "scale": vae_scale,
        "shift": vae_shift,
    }


def get_fundus_mask(image_np: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError(f"Expected RGB image array, got shape={image_np.shape}")
    return image_np.mean(axis=2) > threshold


def get_fundus_radius_map(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("fundus mask is empty")
    cx = float(xs.mean())
    cy = float(ys.mean())
    radius = float(
        max(
            np.percentile(np.abs(xs - cx), 95),
            np.percentile(np.abs(ys - cy), 95),
        )
    )
    radius = max(radius, 1.0)
    yy, xx = np.indices(mask.shape)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
    return rr.astype(np.float32)


def compute_radial_luma_profile(
    image_np: np.ndarray,
    mask: np.ndarray,
    radius_map: np.ndarray,
    bins: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    luma = 0.299 * image_np[..., 0] + 0.587 * image_np[..., 1] + 0.114 * image_np[..., 2]
    edges = np.linspace(0.0, 1.05, bins + 1, dtype=np.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = []
    prev = None
    for left, right in zip(edges[:-1], edges[1:]):
        sel = mask & (radius_map >= left) & (radius_map < right)
        if sel.any():
            prev = float(luma[sel].mean())
        values.append(prev if prev is not None else 0.0)
    values = np.asarray(values, dtype=np.float32)
    if values.size >= 3:
        kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
        values = np.convolve(np.pad(values, (1, 1), mode="edge"), kernel, mode="valid")
    return centers, values


def apply_fundus_color_transfer(
    image: torch.Tensor,
    reference_path: str,
    gamma: float = 1.0,
    warmth_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    saturation: float = 1.0,
    contrast: float = 1.0,
    shadow_lift: float = 0.0,
    edge_lift: float = 0.0,
    flatten_strength: float = 0.0,
    profile_json: str | None = None,
    shade_correction_strength: float = 0.0,
    shade_blur_radius: float = 41.0,
) -> torch.Tensor:
    image_np = image.detach().cpu().permute(1, 2, 0).float().numpy().clip(0.0, 1.0)
    reference_np = np.asarray(Image.open(reference_path).convert("RGB")).astype(np.float32) / 255.0

    src_mask = get_fundus_mask(image_np)
    ref_mask = get_fundus_mask(reference_np)
    if src_mask.sum() < 64 or ref_mask.sum() < 64:
        print("[WARN] fundus color transfer skipped due to insufficient mask coverage.")
        return image

    src_r = get_fundus_radius_map(src_mask)
    ref_r = get_fundus_radius_map(ref_mask)

    # Match statistics on the interior retina rather than the bright peripheral rim.
    src_core = src_mask & (src_r <= 0.82)
    ref_core = ref_mask & (ref_r <= 0.82)
    if src_core.sum() < 64 or ref_core.sum() < 64:
        src_core = src_mask
        ref_core = ref_mask

    src_pixels = image_np[src_core]
    ref_pixels = reference_np[ref_core]

    src_mean = src_pixels.mean(axis=0)
    src_std = src_pixels.std(axis=0) + 1e-6
    ref_mean = ref_pixels.mean(axis=0)
    ref_std = ref_pixels.std(axis=0) + 1e-6

    corrected = image_np.copy()
    transformed = ((image_np - src_mean) / src_std) * ref_std + ref_mean
    transformed = np.clip(transformed, 0.0, 1.0)

    # Keep the edge ring close to the source to avoid the "glowing eyeball" effect.
    blend = np.clip((0.94 - src_r) / 0.34, 0.0, 1.0) ** 1.4
    blend = blend[..., None]
    corrected[src_mask] = (
        blend[src_mask] * transformed[src_mask] + (1.0 - blend[src_mask]) * image_np[src_mask]
    )
    corrected = np.clip(corrected, 0.0, 1.0)

    if gamma != 1.0:
        corrected[src_core] = np.clip(corrected[src_core], 0.0, 1.0) ** gamma

    warmth = np.array(warmth_rgb, dtype=np.float32)
    corrected = np.clip(corrected * warmth[None, None, :], 0.0, 1.0)

    if contrast != 1.0:
        core_mean = corrected[src_core].mean(axis=0, keepdims=True)
        corrected[src_core] = np.clip((corrected[src_core] - core_mean) * contrast + core_mean, 0.0, 1.0)

    if saturation != 1.0:
        gray = corrected[src_core].mean(axis=1, keepdims=True)
        corrected[src_core] = np.clip(gray + (corrected[src_core] - gray) * saturation, 0.0, 1.0)

    if shadow_lift > 0.0:
        luma = 0.299 * corrected[src_mask, 0] + 0.587 * corrected[src_mask, 1] + 0.114 * corrected[src_mask, 2]
        dark_weight = np.clip((0.52 - luma) / 0.52, 0.0, 1.0) ** 1.35
        corrected[src_mask] = np.clip(corrected[src_mask] + shadow_lift * dark_weight[:, None], 0.0, 1.0)

    if edge_lift > 0.0:
        edge_weight = np.clip((src_r[src_mask] - 0.42) / 0.50, 0.0, 1.0) ** 1.6
        corrected[src_mask] = np.clip(corrected[src_mask] + edge_lift * edge_weight[:, None], 0.0, 1.0)

    if flatten_strength > 0.0:
        src_centers, src_profile = compute_radial_luma_profile(corrected, src_mask, src_r, bins=18)
        if profile_json is not None and os.path.exists(profile_json):
            with open(profile_json, "r", encoding="utf-8") as handle:
                profile_stats = json.load(handle)
            ref_profile = np.asarray(profile_stats["radial_luma_median"], dtype=np.float32)
            ref_centers = np.linspace(0.5 / len(ref_profile), 1.0 - 0.5 / len(ref_profile), len(ref_profile), dtype=np.float32)
        else:
            ref_centers, ref_profile = compute_radial_luma_profile(reference_np, ref_mask, ref_r, bins=18)
        ref_interp = np.interp(src_centers, ref_centers, ref_profile).astype(np.float32)
        gains = ref_interp / np.maximum(src_profile, 1e-4)
        gains = np.clip(gains, 1.0, 1.85)
        gain_map = np.interp(src_r, src_centers, gains, left=gains[0], right=gains[-1]).astype(np.float32)
        edge_weight = np.clip((src_r - 0.28) / 0.72, 0.0, 1.0) ** 1.35
        gain_map = 1.0 + (gain_map - 1.0) * flatten_strength * edge_weight
        corrected[src_mask] = np.clip(corrected[src_mask] * gain_map[src_mask, None], 0.0, 1.0)

    if shade_correction_strength > 0.0:
        luma = 0.299 * corrected[..., 0] + 0.587 * corrected[..., 1] + 0.114 * corrected[..., 2]
        luma_img = Image.fromarray(np.uint8(np.clip(luma * 255.0, 0.0, 255.0)), mode="L")
        illum = np.asarray(
            luma_img.filter(ImageFilter.GaussianBlur(radius=max(1.0, float(shade_blur_radius))))
        ).astype(np.float32) / 255.0
        target = float(np.mean(illum[src_mask]))
        gain = target / np.maximum(illum, 1e-3)
        gain = np.clip(gain, 0.85, 1.55)
        gain = 1.0 + (gain - 1.0) * shade_correction_strength
        corrected[src_mask] = np.clip(corrected[src_mask] * gain[src_mask, None], 0.0, 1.0)

    corrected[~src_mask] = 0.0

    corrected_tensor = torch.from_numpy(corrected).permute(2, 0, 1).to(dtype=image.dtype)
    return corrected_tensor


def apply_fundus_tone_adjustment(
    image: torch.Tensor,
    gamma: float = 1.0,
    warmth_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    saturation: float = 1.0,
    contrast: float = 1.0,
    shadow_lift: float = 0.0,
    edge_lift: float = 0.0,
) -> torch.Tensor:
    image_np = image.detach().cpu().permute(1, 2, 0).float().numpy().clip(0.0, 1.0)
    fundus_mask = get_fundus_mask(image_np)
    if fundus_mask.sum() < 64:
        print("[WARN] generic fundus tone adjustment skipped due to insufficient mask coverage.")
        return image

    radius_map = get_fundus_radius_map(fundus_mask)
    core_mask = fundus_mask & (radius_map <= 0.84)
    if core_mask.sum() < 64:
        core_mask = fundus_mask

    corrected = image_np.copy()

    if gamma != 1.0:
        corrected[core_mask] = np.clip(corrected[core_mask], 0.0, 1.0) ** gamma

    warmth = np.array(warmth_rgb, dtype=np.float32)
    corrected[fundus_mask] = np.clip(corrected[fundus_mask] * warmth[None, :], 0.0, 1.0)

    if contrast != 1.0:
        core_mean = corrected[core_mask].mean(axis=0, keepdims=True)
        corrected[core_mask] = np.clip((corrected[core_mask] - core_mean) * contrast + core_mean, 0.0, 1.0)

    if saturation != 1.0:
        gray = corrected[core_mask].mean(axis=1, keepdims=True)
        corrected[core_mask] = np.clip(gray + (corrected[core_mask] - gray) * saturation, 0.0, 1.0)

    if shadow_lift > 0.0:
        luma = 0.299 * corrected[fundus_mask, 0] + 0.587 * corrected[fundus_mask, 1] + 0.114 * corrected[fundus_mask, 2]
        dark_weight = np.clip((0.55 - luma) / 0.55, 0.0, 1.0) ** 1.3
        corrected[fundus_mask] = np.clip(corrected[fundus_mask] + shadow_lift * dark_weight[:, None], 0.0, 1.0)

    if edge_lift > 0.0:
        edge_weight = np.clip((radius_map[fundus_mask] - 0.42) / 0.50, 0.0, 1.0) ** 1.5
        corrected[fundus_mask] = np.clip(corrected[fundus_mask] + edge_lift * edge_weight[:, None], 0.0, 1.0)

    corrected[~fundus_mask] = 0.0
    return torch.from_numpy(corrected).permute(2, 0, 1).to(dtype=image.dtype)


@torch.no_grad()
def main(args):
    init_single_process_dist()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    device = torch.device("cuda")
    inference_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.precision]

    tokenizer_path = args.tokenizer_path or os.path.join(os.path.dirname(__file__), "google_gemma-2b")
    print(f"[*] Loading tokenizer/text encoder from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    text_encoder = (
        AutoModelForCausalLM.from_pretrained(tokenizer_path, torch_dtype=inference_dtype).get_decoder().to(device).eval()
    )

    model = models.__dict__[args.model](
        in_channels=4,
        qk_norm=args.qk_norm,
        cap_feat_dim=text_encoder.config.hidden_size,
        struct_mask_channels=args.struct_mask_channels,
    )
    print(f"[*] Model config: qk_norm={args.qk_norm}")

    base_ckpt_path = resolve_ckpt_path(args.base_ckpt, "consolidated.00-of-01.pth")
    load_robust(model, base_ckpt_path, required=True)

    if args.disable_adapter:
        print("[*] Adapter loading disabled by flag.")
    elif args.adapter_ckpt:
        adapter_ckpt_path = resolve_ckpt_path(args.adapter_ckpt, "adapter.pth")
        load_robust(model, adapter_ckpt_path, required=False)
    else:
        print("[WARN] --adapter_ckpt not provided, running base model only.")

    if args.zero_mask_embedder:
        print("[*] Zeroing mask conditioning path by flag.")
        if hasattr(model, "reset_mask_conditioning"):
            model.reset_mask_conditioning()
        elif hasattr(model, "mask_embedder"):
            torch.nn.init.zeros_(model.mask_embedder.weight)
            if hasattr(model.mask_embedder, "bias") and model.mask_embedder.bias is not None:
                torch.nn.init.zeros_(model.mask_embedder.bias)

    model = model.to(device, dtype=inference_dtype).eval()

    base_seqlen = (args.image_size // 16) ** 2
    for layer in model.layers:
        layer.attention.base_seqlen = base_seqlen
        layer.attention.proportional_attn = True

    print(f"[*] Loading VAE decoder: {args.vae}")
    vae_cfg = get_vae_config(args.vae, args.local_diffusers_model_root)
    vae_kwargs = {"torch_dtype": torch.float32}
    if vae_cfg["subfolder"] is not None:
        vae_kwargs["subfolder"] = vae_cfg["subfolder"]
    vae = AutoencoderKL.from_pretrained(vae_cfg["path"], **vae_kwargs).to(device)

    print("[*] Encoding prompt and mask...")
    captions = [args.prompt, ""]
    text_inputs = tokenizer(
        captions,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device)
    cap_feats = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_masks,
        output_hidden_states=True,
    ).hidden_states[-2]

    if args.mask_path and os.path.exists(args.mask_path):
        mask_tensor = load_struct_mask_tensor(
            args.mask_path,
            image_size=args.image_size,
            struct_mask_channels=args.struct_mask_channels,
        ).unsqueeze(0).to(device=device, dtype=inference_dtype)
    elif args.disable_mask_condition:
        print("[WARN] --mask_path missing, fallback to all-zero mask because --disable_mask_condition is enabled.")
        mask_tensor = torch.zeros(
            (1, args.struct_mask_channels, args.image_size, args.image_size),
            device=device,
            dtype=inference_dtype,
        )
    else:
        raise FileNotFoundError(
            f"--mask_path not found: {args.mask_path}. "
            "Please provide a valid mask image path."
        )

    latent_size = args.image_size // 8
    struct_mask_latent = torch.nn.functional.interpolate(
        mask_tensor,
        size=(latent_size, latent_size),
        mode="bilinear",
        align_corners=False,
    )
    struct_mask_latent = struct_mask_latent * float(args.mask_scale)
    cond_struct_mask = torch.zeros_like(struct_mask_latent) if args.disable_mask_condition else struct_mask_latent
    uncond_struct_mask = torch.zeros_like(struct_mask_latent) if args.zero_uncond_mask else cond_struct_mask
    struct_mask = torch.cat([cond_struct_mask, uncond_struct_mask], dim=0)
    if args.disable_mask_condition:
        print("[WARN] Mask condition is disabled by --disable_mask_condition.")
    else:
        nonzero_ratio = (struct_mask_latent > 1e-6).float().mean().item()
        print(
            "[diag] cond mask latent stats: "
            f"min={struct_mask_latent.min().item():.6f} "
            f"max={struct_mask_latent.max().item():.6f} "
            f"mean={struct_mask_latent.mean().item():.6f} "
            f"nonzero_ratio={nonzero_ratio:.6f} "
            f"mask_scale={args.mask_scale}"
        )

    model_kwargs = dict(
        cap_feats=cap_feats,
        cap_mask=prompt_masks,
        cfg_scale=args.cfg_scale,
        struct_mask=struct_mask,
        base_seqlen=base_seqlen,
        proportional_attn=True,
    )

    print(
        f"[*] Sampling with ODE ({args.sampling_method}), "
        f"steps={args.num_sampling_steps}, cfg={args.cfg_scale}, seed={args.seed}"
    )
    torch.manual_seed(args.seed)
    z0 = torch.randn([1, 4, latent_size, latent_size], device=device, dtype=inference_dtype)
    z0 = torch.cat([z0, z0], dim=0)

    sampler = Sampler(create_transport("Linear", "velocity", None, None, None))
    sample_fn = sampler.sample_ode(
        sampling_method=args.sampling_method,
        num_steps=args.num_sampling_steps,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    )

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=inference_dtype)
        if inference_dtype in (torch.float16, torch.bfloat16)
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        samples_latent = sample_fn(z0, model.forward_with_cfg, **model_kwargs)[-1]
    final_latent = samples_latent[:1]

    print(
        "[diag] latent stats: "
        f"min={final_latent.min().item():.4f} "
        f"max={final_latent.max().item():.4f} "
        f"mean={final_latent.mean().item():.4f}"
    )

    print("[*] Decoding image...")
    with torch.no_grad():
        final_latent = final_latent.float() / vae_cfg["scale"] + vae_cfg["shift"]
        decoded_image = (vae.decode(final_latent).sample / 2 + 0.5).clamp(0, 1)

    tone_requested = any(
        [
            args.fundus_color_gamma != 1.0,
            args.fundus_warm_r != 1.0,
            args.fundus_warm_g != 1.0,
            args.fundus_warm_b != 1.0,
            args.fundus_saturation != 1.0,
            args.fundus_contrast != 1.0,
            args.fundus_shadow_lift > 0.0,
            args.fundus_edge_lift > 0.0,
        ]
    )

    if args.fundus_color_reference is not None:
        print(f"[*] Applying fundus color transfer using reference: {args.fundus_color_reference}")
        decoded_image[0] = apply_fundus_color_transfer(
            decoded_image[0],
            args.fundus_color_reference,
            gamma=args.fundus_color_gamma,
            warmth_rgb=(args.fundus_warm_r, args.fundus_warm_g, args.fundus_warm_b),
            saturation=args.fundus_saturation,
            contrast=args.fundus_contrast,
            shadow_lift=args.fundus_shadow_lift,
            edge_lift=args.fundus_edge_lift,
            flatten_strength=args.fundus_flatten_strength,
            profile_json=args.fundus_profile_json,
            shade_correction_strength=args.fundus_shade_correction_strength,
            shade_blur_radius=args.fundus_shade_blur_radius,
        )
    elif tone_requested:
        print("[*] Applying generic fundus tone adjustment without reference image.")
        decoded_image[0] = apply_fundus_tone_adjustment(
            decoded_image[0],
            gamma=args.fundus_color_gamma,
            warmth_rgb=(args.fundus_warm_r, args.fundus_warm_g, args.fundus_warm_b),
            saturation=args.fundus_saturation,
            contrast=args.fundus_contrast,
            shadow_lift=args.fundus_shadow_lift,
            edge_lift=args.fundus_edge_lift,
        )

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = f"{args.filename_prefix}_" if args.filename_prefix else ""
    filename = (
        args.output_name
        if args.output_name
        else f"{prefix}fundus_seed{args.seed}_cfg{args.cfg_scale}_steps{args.num_sampling_steps}_{time.strftime('%H%M%S')}.png"
    )
    if args.output_name and not filename.lower().endswith(".png"):
        filename = f"{filename}.png"
    out_path = os.path.join(args.out_dir, filename)
    ToPILImage()(decoded_image[0].cpu().float()).save(out_path)
    print(f"[ok] saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="NextDiT_2B_GQA_patch2")
    parser.add_argument("--base_ckpt", type=str, required=True)
    parser.add_argument("--adapter_ckpt", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--struct_mask_channels", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="./results/inference_output")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_sampling_steps", type=int, default=80)
    parser.add_argument("--sampling_method", type=str, default="euler")
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--vae", type=str, choices=["sdxl", "sd3", "ema", "mse"], default="sdxl")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--local_diffusers_model_root", type=str, default=None)
    parser.add_argument("--mask_scale", type=float, default=2.0)
    parser.add_argument("--filename_prefix", type=str, default="")
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--qk_norm", action="store_true")
    parser.add_argument("--disable_adapter", action="store_true")
    parser.add_argument("--disable_mask_condition", action="store_true")
    parser.add_argument("--zero_mask_embedder", action="store_true")
    parser.add_argument("--zero_uncond_mask", action="store_true")
    parser.add_argument("--fundus_color_reference", type=str, default=None)
    parser.add_argument("--fundus_color_gamma", type=float, default=1.0)
    parser.add_argument("--fundus_warm_r", type=float, default=1.0)
    parser.add_argument("--fundus_warm_g", type=float, default=1.0)
    parser.add_argument("--fundus_warm_b", type=float, default=1.0)
    parser.add_argument("--fundus_saturation", type=float, default=1.0)
    parser.add_argument("--fundus_contrast", type=float, default=1.0)
    parser.add_argument("--fundus_shadow_lift", type=float, default=0.0)
    parser.add_argument("--fundus_edge_lift", type=float, default=0.0)
    parser.add_argument("--fundus_flatten_strength", type=float, default=0.0)
    parser.add_argument("--fundus_profile_json", type=str, default=None)
    parser.add_argument("--fundus_shade_correction_strength", type=float, default=0.0)
    parser.add_argument("--fundus_shade_blur_radius", type=float, default=41.0)
    try:
        main(parser.parse_args())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
