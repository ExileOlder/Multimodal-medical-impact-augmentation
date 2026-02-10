"""Extended NextDiT model with mask conditioning support."""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional, List, Tuple

# Add codes directory to path to import original model
codes_path = Path(__file__).parent.parent.parent / "codes"
sys.path.insert(0, str(codes_path))

from models.model import NextDiT as OriginalNextDiT
from .mask_utils import prepare_mask


class NextDiTWithMask(nn.Module):
    """
    Extended NextDiT model that supports segmentation mask conditioning.
    
    This wraps the original NextDiT and adds mask input support by:
    1. Concatenating mask with image before patchify
    2. Adjusting input channels accordingly
    3. Preserving Flash Attention and all other optimizations
    """
    
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 3,  # RGB input
        mask_channels: int = 1,  # Single channel mask
        dim: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        learn_sigma: bool = True,
        qk_norm: bool = False,
        cap_feat_dim: int = 5120,
        scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.mask_channels = mask_channels
        self.total_in_channels = in_channels + mask_channels
        
        # Initialize base NextDiT with combined channels
        self.base_model = OriginalNextDiT(
            patch_size=patch_size,
            in_channels=self.total_in_channels,  # RGB + Mask
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            norm_eps=norm_eps,
            learn_sigma=learn_sigma,
            qk_norm=qk_norm,
            cap_feat_dim=cap_feat_dim,
            scale_factor=scale_factor,
        )
        
        # Store output channels (excluding mask channels)
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.learn_sigma = learn_sigma
        self.patch_size = patch_size
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cap_feats: torch.Tensor,
        cap_mask: torch.Tensor,
        condition_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional mask conditioning.
        
        Args:
            x: Image tensor (B, C, H, W)
            t: Timestep tensor (B,)
            cap_feats: Caption features (B, L, D)
            cap_mask: Caption mask (B, L)
            condition_mask: Optional segmentation mask (B, 1, H, W)
                           If None, uses zero mask (text-only mode)
        
        Returns:
            Model output (B, C, H, W)
        """
        batch_size, _, height, width = x.shape
        
        # Prepare mask - create zero mask if None
        if condition_mask is None:
            condition_mask = torch.zeros(
                batch_size, self.mask_channels, height, width,
                device=x.device, dtype=x.dtype
            )
        else:
            # Ensure mask has correct shape and size
            condition_mask = prepare_mask(
                condition_mask,
                target_size=(height, width),
                device=x.device
            )
        
        # Concatenate image and mask along channel dimension
        # Shape: (B, C+1, H, W)
        x_with_mask = torch.cat([x, condition_mask], dim=1)
        
        # Forward through base model
        output = self.base_model(x_with_mask, t, cap_feats, cap_mask)
        
        return output
    
    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cap_feats: torch.Tensor,
        cap_mask: torch.Tensor,
        condition_mask: Optional[torch.Tensor] = None,
        cfg_scale: float = 7.5,
        scale_factor: float = 1.0,
        scale_watershed: float = 1.0,
        base_seqlen: Optional[int] = None,
        proportional_attn: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with classifier-free guidance and mask conditioning.
        
        Args:
            x: Image tensor (B, C, H, W) - should be doubled for CFG
            t: Timestep tensor (B,)
            cap_feats: Caption features (B, L, D) - should include unconditional
            cap_mask: Caption mask (B, L)
            condition_mask: Optional segmentation mask (B, 1, H, W)
            cfg_scale: Classifier-free guidance scale
            scale_factor: RoPE scale factor
            scale_watershed: RoPE scale watershed
            base_seqlen: Base sequence length for proportional attention
            proportional_attn: Whether to use proportional attention
        
        Returns:
            Model output with CFG applied (B, C, H, W)
        """
        batch_size, _, height, width = x.shape
        
        # Prepare mask
        if condition_mask is None:
            condition_mask = torch.zeros(
                batch_size, self.mask_channels, height, width,
                device=x.device, dtype=x.dtype
            )
        else:
            condition_mask = prepare_mask(
                condition_mask,
                target_size=(height, width),
                device=x.device
            )
        
        # Concatenate image and mask
        x_with_mask = torch.cat([x, condition_mask], dim=1)
        
        # Forward through base model with CFG
        output = self.base_model.forward_with_cfg(
            x_with_mask,
            t,
            cap_feats,
            cap_mask,
            cfg_scale=cfg_scale,
            scale_factor=scale_factor,
            scale_watershed=scale_watershed,
            base_seqlen=base_seqlen,
            proportional_attn=proportional_attn,
        )
        
        return output
    
    def parameter_count(self) -> int:
        """Get total parameter count."""
        return self.base_model.parameter_count()
    
    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        """Get modules for FSDP wrapping."""
        return self.base_model.get_fsdp_wrap_module_list()
    
    def get_checkpointing_wrap_module_list(self) -> List[nn.Module]:
        """Get modules for gradient checkpointing."""
        return self.base_model.get_checkpointing_wrap_module_list()


def NextDiTWithMask_2B_patch2(**kwargs):
    """NextDiT-2B model with mask conditioning."""
    return NextDiTWithMask(
        patch_size=2,
        dim=2304,
        n_layers=24,
        n_heads=32,
        **kwargs
    )


def NextDiTWithMask_2B_GQA_patch2(**kwargs):
    """NextDiT-2B model with GQA and mask conditioning."""
    return NextDiTWithMask(
        patch_size=2,
        dim=2304,
        n_layers=24,
        n_heads=32,
        n_kv_heads=8,  # Grouped Query Attention
        **kwargs
    )
