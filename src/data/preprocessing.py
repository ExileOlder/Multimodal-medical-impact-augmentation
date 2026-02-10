"""Image and mask preprocessing utilities."""

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import cv2


# DR Grade to Caption Mapping (Label-to-Caption for text modality)
DR_GRADE_TO_TEXT = {
    0: "No diabetic retinopathy",
    1: "Mild non-proliferative diabetic retinopathy",
    2: "Moderate non-proliferative diabetic retinopathy",
    3: "Severe non-proliferative diabetic retinopathy",
    4: "Proliferative diabetic retinopathy"
}


def label_to_caption(label: Union[int, str], grade_mapping: dict = None) -> str:
    """
    Convert DR grade label to pathological text description.
    
    This implements the "病理文本整理" (pathological text organization) requirement.
    
    Args:
        label: DR grade (0-4) as int or string
        grade_mapping: Optional custom mapping dict, defaults to DR_GRADE_TO_TEXT
        
    Returns:
        Pathological text description
    """
    if grade_mapping is None:
        grade_mapping = DR_GRADE_TO_TEXT
    
    # Convert string to int if needed
    if isinstance(label, str):
        try:
            label = int(label)
        except ValueError:
            return label  # Return as-is if not a number
    
    return grade_mapping.get(label, f"Unknown grade: {label}")


def apply_gaussian_filter(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Apply Gaussian filter for image denoising.
    
    Gaussian filtering is effective for reducing Gaussian noise while preserving edges.
    Commonly used in medical image preprocessing.
    
    Args:
        image: Input image as numpy array (H, W, C) or (H, W)
        kernel_size: Size of the Gaussian kernel (must be odd). Default: 5
        sigma: Standard deviation of the Gaussian kernel. Default: 1.0
        
    Returns:
        Denoised image as numpy array with same shape as input
    """
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Apply Gaussian blur
    denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    return denoised


def apply_median_filter(
    image: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Apply median filter for image denoising.
    
    Median filtering is particularly effective for removing salt-and-pepper noise
    while preserving edges. Useful for medical images with impulse noise.
    
    Args:
        image: Input image as numpy array (H, W, C) or (H, W)
        kernel_size: Size of the median filter kernel (must be odd). Default: 5
        
    Returns:
        Denoised image as numpy array with same shape as input
    """
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Apply median filter
    denoised = cv2.medianBlur(image, kernel_size)
    
    return denoised


def preprocess_image(
    image: Union[Image.Image, np.ndarray, str],
    target_size: Tuple[int, int] = (1024, 1024),
    normalize: bool = True,
    enable_denoising: bool = False,
    denoising_method: str = "gaussian",
    gaussian_kernel_size: int = 5,
    gaussian_sigma: float = 1.0,
    median_kernel_size: int = 5
) -> torch.Tensor:
    """
    Preprocess image: resize, optional denoising, and normalize.
    
    Args:
        image: PIL Image, numpy array, or path to image file
        target_size: Target resolution (height, width)
        normalize: Whether to normalize pixel values to [-1, 1]
        enable_denoising: Whether to apply denoising (default: False)
        denoising_method: Denoising method - "gaussian" or "median" (default: "gaussian")
        gaussian_kernel_size: Kernel size for Gaussian filter (default: 5)
        gaussian_sigma: Sigma for Gaussian filter (default: 1.0)
        median_kernel_size: Kernel size for median filter (default: 5)
        
    Returns:
        Preprocessed image tensor of shape (C, H, W)
        
    Note:
        Medical image datasets are typically pre-processed, so denoising is disabled
        by default. Enable it only if your dataset contains noisy images.
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    # Convert numpy array to PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Ensure RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to target resolution
    image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to numpy array for optional denoising
    image_np = np.array(image)
    
    # Apply denoising if enabled
    if enable_denoising:
        if denoising_method == "gaussian":
            image_np = apply_gaussian_filter(
                image_np,
                kernel_size=gaussian_kernel_size,
                sigma=gaussian_sigma
            )
        elif denoising_method == "median":
            image_np = apply_median_filter(
                image_np,
                kernel_size=median_kernel_size
            )
        else:
            print(f"Warning: Unknown denoising method '{denoising_method}', skipping denoising")
    
    # Convert to tensor (C, H, W) with values in [0, 1]
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    
    # Normalize to [-1, 1] if requested
    if normalize:
        image_tensor = image_tensor * 2.0 - 1.0
    
    return image_tensor


def preprocess_mask(
    mask: Union[Image.Image, np.ndarray, str],
    target_size: Tuple[int, int] = (1024, 1024),
    num_classes: Optional[int] = None
) -> torch.Tensor:
    """
    Preprocess segmentation mask: resize and convert to integer labels.
    
    Args:
        mask: PIL Image, numpy array, or path to mask file
        target_size: Target resolution (height, width)
        num_classes: Number of classes (for validation, optional)
        
    Returns:
        Preprocessed mask tensor of shape (1, H, W) with integer labels
    """
    # Load mask if path is provided
    if isinstance(mask, str):
        mask = Image.open(mask)
    
    # Convert numpy array to PIL Image
    if isinstance(mask, np.ndarray):
        # Handle multi-channel masks (merge channels)
        if mask.ndim == 3 and mask.shape[2] > 1:
            # Merge multiple lesion masks into single channel
            mask = np.max(mask, axis=2)
        mask = Image.fromarray(mask.astype(np.uint8))
    
    # Convert to grayscale if needed
    if mask.mode != 'L':
        mask = mask.convert('L')
    
    # Resize using nearest neighbor to preserve label values
    mask = mask.resize(target_size, Image.NEAREST)
    
    # Convert to tensor (1, H, W)
    mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0).float()
    
    # Validate class range if specified
    if num_classes is not None:
        max_val = mask_tensor.max().item()
        if max_val >= num_classes:
            print(f"Warning: Mask contains value {max_val} >= num_classes {num_classes}")
    
    return mask_tensor


def log_data_statistics(data_list: list, name: str = "Dataset"):
    """
    Log statistics about the dataset.
    
    Args:
        data_list: List of data entries
        name: Name of the dataset for logging
    """
    total = len(data_list)
    text_only = sum(1 for entry in data_list if entry.get('text_only', False))
    with_mask = total - text_only
    
    print(f"\n{'='*50}")
    print(f"{name} Statistics:")
    print(f"{'='*50}")
    print(f"Total entries: {total}")
    print(f"Entries with masks: {with_mask} ({with_mask/total*100:.1f}%)")
    print(f"Text-only entries: {text_only} ({text_only/total*100:.1f}%)")
    
    # Count captions
    if data_list:
        caption_lengths = [len(entry.get('caption', '')) for entry in data_list]
        print(f"Caption length - Min: {min(caption_lengths)}, "
              f"Max: {max(caption_lengths)}, "
              f"Avg: {sum(caption_lengths)/len(caption_lengths):.1f}")
    
    print(f"{'='*50}\n")


def merge_lesion_masks(mask_paths: list, target_size: Tuple[int, int] = (1024, 1024)) -> torch.Tensor:
    """
    Merge multiple lesion masks (MA, HE, EX, SE) into a single channel.
    
    Useful for FGADR dataset which provides separate masks for different lesions.
    
    Args:
        mask_paths: List of paths to individual lesion masks
        target_size: Target resolution
        
    Returns:
        Merged mask tensor of shape (1, H, W)
    """
    merged_mask = None
    
    for mask_path in mask_paths:
        if mask_path and Image.open(mask_path):
            mask = preprocess_mask(mask_path, target_size)
            if merged_mask is None:
                merged_mask = mask
            else:
                # Take maximum to merge lesions
                merged_mask = torch.maximum(merged_mask, mask)
    
    if merged_mask is None:
        # Return zero mask if no valid masks found
        merged_mask = torch.zeros(1, *target_size)
    
    return merged_mask
