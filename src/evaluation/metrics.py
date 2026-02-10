"""Image quality metrics for evaluation."""

import torch
import numpy as np
from PIL import Image
from typing import Union, Tuple
import math


def calculate_psnr(
    img1: Union[np.ndarray, torch.Tensor, Image.Image],
    img2: Union[np.ndarray, torch.Tensor, Image.Image],
    max_value: float = 255.0
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1: First image
        img2: Second image
        max_value: Maximum possible pixel value (255 for uint8)
        
    Returns:
        PSNR value in dB
    """
    # Convert to numpy arrays
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    # Ensure same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have same shape: {img1.shape} vs {img2.shape}")
    
    # Convert to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    psnr = 20 * math.log10(max_value / math.sqrt(mse))
    
    return psnr


def calculate_ssim(
    img1: Union[np.ndarray, torch.Tensor, Image.Image],
    img2: Union[np.ndarray, torch.Tensor, Image.Image],
    max_value: float = 255.0,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03
) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Simplified implementation for grayscale or RGB images.
    
    Args:
        img1: First image
        img2: Second image
        max_value: Maximum possible pixel value
        window_size: Size of the Gaussian window
        k1: Constant for stability (default 0.01)
        k2: Constant for stability (default 0.03)
        
    Returns:
        SSIM value between -1 and 1 (1 means identical)
    """
    # Convert to numpy arrays
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    # Ensure same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have same shape: {img1.shape} vs {img2.shape}")
    
    # Convert to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # If RGB, convert to grayscale for simplicity
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        img1 = np.mean(img1, axis=2)
        img2 = np.mean(img2, axis=2)
    
    # Constants
    c1 = (k1 * max_value) ** 2
    c2 = (k2 * max_value) ** 2
    
    # Calculate means
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # Calculate variances and covariance
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]
    
    # Calculate SSIM
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    
    ssim = numerator / denominator
    
    return float(ssim)


def calculate_ssim_skimage(
    img1: Union[np.ndarray, torch.Tensor, Image.Image],
    img2: Union[np.ndarray, torch.Tensor, Image.Image],
    max_value: float = 255.0
) -> float:
    """
    Calculate SSIM using scikit-image (more accurate).
    
    Args:
        img1: First image
        img2: Second image
        max_value: Maximum possible pixel value
        
    Returns:
        SSIM value
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        
        # Convert to numpy arrays
        if isinstance(img1, Image.Image):
            img1 = np.array(img1)
        if isinstance(img2, Image.Image):
            img2 = np.array(img2)
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()
        
        # Ensure same shape
        if img1.shape != img2.shape:
            raise ValueError(f"Images must have same shape: {img1.shape} vs {img2.shape}")
        
        # Calculate SSIM
        if len(img1.shape) == 3:
            # RGB image
            return ssim(img1, img2, data_range=max_value, channel_axis=2)
        else:
            # Grayscale image
            return ssim(img1, img2, data_range=max_value)
    
    except ImportError:
        # Fallback to simple implementation
        return calculate_ssim(img1, img2, max_value)


def calculate_mae(
    img1: Union[np.ndarray, torch.Tensor, Image.Image],
    img2: Union[np.ndarray, torch.Tensor, Image.Image]
) -> float:
    """
    Calculate Mean Absolute Error (MAE) between two images.
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        MAE value
    """
    # Convert to numpy arrays
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    # Ensure same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have same shape: {img1.shape} vs {img2.shape}")
    
    # Convert to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Calculate MAE
    mae = np.mean(np.abs(img1 - img2))
    
    return float(mae)


def calculate_mse(
    img1: Union[np.ndarray, torch.Tensor, Image.Image],
    img2: Union[np.ndarray, torch.Tensor, Image.Image]
) -> float:
    """
    Calculate Mean Squared Error (MSE) between two images.
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        MSE value
    """
    # Convert to numpy arrays
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    # Ensure same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have same shape: {img1.shape} vs {img2.shape}")
    
    # Convert to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    
    return float(mse)


def calculate_dice_coefficient(
    mask1: Union[np.ndarray, torch.Tensor, Image.Image],
    mask2: Union[np.ndarray, torch.Tensor, Image.Image],
    threshold: float = 0.5
) -> float:
    """
    Calculate Dice Coefficient (F1 Score) between two binary masks.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    用于评估生成图像的病灶区域是否与输入掩码一致。
    
    Args:
        mask1: First mask (predicted/generated)
        mask2: Second mask (ground truth/input)
        threshold: Threshold for binarization (default: 0.5)
        
    Returns:
        Dice coefficient value between 0 and 1 (1 means perfect overlap)
    """
    # Convert to numpy arrays
    if isinstance(mask1, Image.Image):
        mask1 = np.array(mask1)
    if isinstance(mask2, Image.Image):
        mask2 = np.array(mask2)
    if isinstance(mask1, torch.Tensor):
        mask1 = mask1.cpu().numpy()
    if isinstance(mask2, torch.Tensor):
        mask2 = mask2.cpu().numpy()
    
    # Ensure same shape
    if mask1.shape != mask2.shape:
        raise ValueError(f"Masks must have same shape: {mask1.shape} vs {mask2.shape}")
    
    # Convert to float and normalize to [0, 1]
    mask1 = mask1.astype(np.float64)
    mask2 = mask2.astype(np.float64)
    
    if mask1.max() > 1.0:
        mask1 = mask1 / 255.0
    if mask2.max() > 1.0:
        mask2 = mask2 / 255.0
    
    # Binarize masks
    mask1_binary = (mask1 > threshold).astype(np.float64)
    mask2_binary = (mask2 > threshold).astype(np.float64)
    
    # Calculate intersection and union
    intersection = np.sum(mask1_binary * mask2_binary)
    sum_masks = np.sum(mask1_binary) + np.sum(mask2_binary)
    
    # Handle edge case: both masks are empty
    if sum_masks == 0:
        return 1.0  # Perfect match if both are empty
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection) / sum_masks
    
    return float(dice)


def calculate_iou(
    mask1: Union[np.ndarray, torch.Tensor, Image.Image],
    mask2: Union[np.ndarray, torch.Tensor, Image.Image],
    threshold: float = 0.5
) -> float:
    """
    Calculate Intersection over Union (IoU / Jaccard Index) between two binary masks.
    
    IoU = |A ∩ B| / |A ∪ B|
    
    用于评估生成图像的病灶区域是否与输入掩码一致。
    
    Args:
        mask1: First mask (predicted/generated)
        mask2: Second mask (ground truth/input)
        threshold: Threshold for binarization (default: 0.5)
        
    Returns:
        IoU value between 0 and 1 (1 means perfect overlap)
    """
    # Convert to numpy arrays
    if isinstance(mask1, Image.Image):
        mask1 = np.array(mask1)
    if isinstance(mask2, Image.Image):
        mask2 = np.array(mask2)
    if isinstance(mask1, torch.Tensor):
        mask1 = mask1.cpu().numpy()
    if isinstance(mask2, torch.Tensor):
        mask2 = mask2.cpu().numpy()
    
    # Ensure same shape
    if mask1.shape != mask2.shape:
        raise ValueError(f"Masks must have same shape: {mask1.shape} vs {mask2.shape}")
    
    # Convert to float and normalize to [0, 1]
    mask1 = mask1.astype(np.float64)
    mask2 = mask2.astype(np.float64)
    
    if mask1.max() > 1.0:
        mask1 = mask1 / 255.0
    if mask2.max() > 1.0:
        mask2 = mask2 / 255.0
    
    # Binarize masks
    mask1_binary = (mask1 > threshold).astype(np.float64)
    mask2_binary = (mask2 > threshold).astype(np.float64)
    
    # Calculate intersection and union
    intersection = np.sum(mask1_binary * mask2_binary)
    union = np.sum(mask1_binary) + np.sum(mask2_binary) - intersection
    
    # Handle edge case: both masks are empty
    if union == 0:
        return 1.0  # Perfect match if both are empty
    
    # Calculate IoU
    iou = intersection / union
    
    return float(iou)


def extract_lesion_mask_from_image(
    image: Union[np.ndarray, torch.Tensor, Image.Image],
    method: str = "red_channel",
    threshold: float = 0.5
) -> np.ndarray:
    """
    从生成的图像中提取病灶掩码（用于结构一致性验证）。
    
    Args:
        image: Input image (RGB)
        method: Extraction method:
            - "red_channel": Use red channel intensity (病灶通常呈红色)
            - "saturation": Use HSV saturation (病灶区域饱和度高)
            - "brightness": Use brightness difference
        threshold: Threshold for binarization
        
    Returns:
        Binary mask as numpy array
    """
    # Convert to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # Ensure RGB format
    if len(image.shape) == 2:
        # Grayscale, convert to RGB
        image = np.stack([image, image, image], axis=-1)
    
    # Normalize to [0, 1]
    if image.max() > 1.0:
        image = image.astype(np.float64) / 255.0
    
    if method == "red_channel":
        # 提取红色通道（病灶通常呈红色）
        red_channel = image[:, :, 0]
        mask = (red_channel > threshold).astype(np.float64)
    
    elif method == "saturation":
        # 使用HSV饱和度（病灶区域饱和度高）
        from colorsys import rgb_to_hsv
        
        # Convert to HSV
        h, w = image.shape[:2]
        hsv_image = np.zeros_like(image)
        
        for i in range(h):
            for j in range(w):
                r, g, b = image[i, j]
                h_val, s_val, v_val = rgb_to_hsv(r, g, b)
                hsv_image[i, j] = [h_val, s_val, v_val]
        
        # Extract saturation channel
        saturation = hsv_image[:, :, 1]
        mask = (saturation > threshold).astype(np.float64)
    
    elif method == "brightness":
        # 使用亮度差异
        brightness = np.mean(image, axis=2)
        # 病灶区域通常比背景暗
        mask = (brightness < (1.0 - threshold)).astype(np.float64)
    
    else:
        raise ValueError(f"Unknown extraction method: {method}")
    
    return mask
