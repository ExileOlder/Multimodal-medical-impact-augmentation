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
