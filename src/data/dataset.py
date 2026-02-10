"""PyTorch Dataset for multimodal medical image data."""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from .preprocessing import preprocess_image, preprocess_mask, label_to_caption


class MultimodalDataset(Dataset):
    """
    Dataset for multimodal medical image generation.
    
    Supports:
    - Image + Text + Mask (full multimodal)
    - Image + Text (text-only mode, no mask)
    """
    
    def __init__(
        self,
        data_list: List[Dict],
        image_size: Tuple[int, int] = (1024, 1024),
        normalize: bool = True,
        auto_caption: bool = True,
        enable_denoising: bool = False,
        denoising_method: str = "gaussian",
        gaussian_kernel_size: int = 5,
        gaussian_sigma: float = 1.0,
        median_kernel_size: int = 5
    ):
        """
        Initialize dataset.
        
        Args:
            data_list: List of data entries from JSONL loader
            image_size: Target image resolution
            normalize: Whether to normalize images to [-1, 1]
            auto_caption: Whether to convert numeric labels to text captions
            enable_denoising: Whether to apply denoising (default: False)
            denoising_method: Denoising method - "gaussian" or "median"
            gaussian_kernel_size: Kernel size for Gaussian filter
            gaussian_sigma: Sigma for Gaussian filter
            median_kernel_size: Kernel size for median filter
        """
        self.data_list = data_list
        self.image_size = image_size
        self.normalize = normalize
        self.auto_caption = auto_caption
        self.enable_denoising = enable_denoising
        self.denoising_method = denoising_method
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self.median_kernel_size = median_kernel_size
        
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single data sample.
        
        Returns:
            Dictionary containing:
            - 'image': Image tensor (C, H, W)
            - 'caption': Text caption (string)
            - 'mask': Mask tensor (1, H, W) or None
            - 'text_only': Boolean flag
        """
        entry = self.data_list[idx]
        
        # Load and preprocess image
        image = preprocess_image(
            entry['image_path'],
            target_size=self.image_size,
            normalize=self.normalize,
            enable_denoising=self.enable_denoising,
            denoising_method=self.denoising_method,
            gaussian_kernel_size=self.gaussian_kernel_size,
            gaussian_sigma=self.gaussian_sigma,
            median_kernel_size=self.median_kernel_size
        )
        
        # Get caption (with optional label-to-caption conversion)
        caption = entry['caption']
        if self.auto_caption:
            # Try to convert numeric labels to text
            caption = label_to_caption(caption)
        
        # Load and preprocess mask if available
        mask = None
        text_only = entry.get('text_only', False)
        
        if not text_only and entry['mask_path'] is not None:
            try:
                mask = preprocess_mask(
                    entry['mask_path'],
                    target_size=self.image_size
                )
            except Exception as e:
                print(f"Warning: Failed to load mask {entry['mask_path']}: {e}")
                text_only = True
        
        return {
            'image': image,
            'caption': caption,
            'mask': mask,
            'text_only': text_only
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Handles batches with mixed text-only and multimodal samples.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched dictionary with:
        - 'images': (B, C, H, W)
        - 'captions': List of strings
        - 'masks': (B, 1, H, W) or None for text-only samples
        - 'text_only_flags': (B,) boolean tensor
    """
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    text_only_flags = torch.tensor([item['text_only'] for item in batch])
    
    # Handle masks - create zero masks for text-only samples
    masks = []
    for item in batch:
        if item['mask'] is not None:
            masks.append(item['mask'])
        else:
            # Create zero mask for text-only samples
            h, w = item['image'].shape[1:]
            masks.append(torch.zeros(1, h, w))
    
    masks = torch.stack(masks)
    
    return {
        'images': images,
        'captions': captions,
        'masks': masks,
        'text_only_flags': text_only_flags
    }
