"""Mask processing utilities for model input."""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def prepare_mask(
    mask: Optional[torch.Tensor],
    target_size: Tuple[int, int],
    device: torch.device = None
) -> torch.Tensor:
    """
    Prepare mask for model input: resize and handle channel dimensions.
    
    Args:
        mask: Input mask tensor of shape (B, 1, H, W) or None
        target_size: Target spatial resolution (H, W)
        device: Target device
        
    Returns:
        Prepared mask tensor of shape (B, 1, H, W)
        Returns zero tensor if mask is None
    """
    if mask is None:
        # Create zero mask for text-only mode
        batch_size = 1
        mask = torch.zeros(batch_size, 1, *target_size)
    
    # Move to device if specified
    if device is not None:
        mask = mask.to(device)
    
    # Get current size
    current_size = mask.shape[-2:]
    
    # Resize if needed using bilinear interpolation
    if current_size != target_size:
        mask = F.interpolate(
            mask,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
    
    # Ensure shape is (B, 1, H, W)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    
    return mask


def create_zero_mask(batch_size: int, size: Tuple[int, int], device: torch.device = None) -> torch.Tensor:
    """
    Create a zero mask tensor for text-only samples.
    
    Args:
        batch_size: Batch size
        size: Spatial size (H, W)
        device: Target device
        
    Returns:
        Zero mask tensor of shape (B, 1, H, W)
    """
    mask = torch.zeros(batch_size, 1, *size)
    if device is not None:
        mask = mask.to(device)
    return mask


def normalize_mask(mask: torch.Tensor, max_val: float = 255.0) -> torch.Tensor:
    """
    Normalize mask values to [0, 1] range.
    
    Args:
        mask: Input mask tensor
        max_val: Maximum value in the mask (default 255 for uint8 images)
        
    Returns:
        Normalized mask tensor
    """
    return mask / max_val


def binarize_mask(mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Binarize mask using a threshold.
    
    Args:
        mask: Input mask tensor
        threshold: Binarization threshold
        
    Returns:
        Binary mask tensor (0 or 1)
    """
    return (mask > threshold).float()
