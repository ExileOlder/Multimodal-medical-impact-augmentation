"""Tests for noise processing functionality."""

import pytest
import numpy as np
import torch
from PIL import Image

from src.data.preprocessing import (
    apply_gaussian_filter,
    apply_median_filter,
    preprocess_image
)


def test_apply_gaussian_filter():
    """Test Gaussian filter denoising."""
    # Create a test image with noise
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Apply Gaussian filter
    denoised = apply_gaussian_filter(image, kernel_size=5, sigma=1.0)
    
    # Check output shape matches input
    assert denoised.shape == image.shape
    
    # Check output dtype
    assert denoised.dtype == np.uint8
    
    # Check that values are in valid range
    assert denoised.min() >= 0
    assert denoised.max() <= 255


def test_apply_gaussian_filter_even_kernel():
    """Test Gaussian filter with even kernel size (should be converted to odd)."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Even kernel size should be converted to odd
    denoised = apply_gaussian_filter(image, kernel_size=4, sigma=1.0)
    
    assert denoised.shape == image.shape


def test_apply_median_filter():
    """Test median filter denoising."""
    # Create a test image with salt-and-pepper noise
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Apply median filter
    denoised = apply_median_filter(image, kernel_size=5)
    
    # Check output shape matches input
    assert denoised.shape == image.shape
    
    # Check output dtype
    assert denoised.dtype == np.uint8
    
    # Check that values are in valid range
    assert denoised.min() >= 0
    assert denoised.max() <= 255


def test_apply_median_filter_even_kernel():
    """Test median filter with even kernel size (should be converted to odd)."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Even kernel size should be converted to odd
    denoised = apply_median_filter(image, kernel_size=4)
    
    assert denoised.shape == image.shape


def test_preprocess_image_with_gaussian_denoising():
    """Test image preprocessing with Gaussian denoising enabled."""
    # Create a test image
    test_image = Image.new('RGB', (256, 256), color=(128, 128, 128))
    
    # Preprocess with Gaussian denoising
    result = preprocess_image(
        test_image,
        target_size=(128, 128),
        normalize=True,
        enable_denoising=True,
        denoising_method="gaussian",
        gaussian_kernel_size=5,
        gaussian_sigma=1.0
    )
    
    # Check output is a tensor
    assert isinstance(result, torch.Tensor)
    
    # Check output shape
    assert result.shape == (3, 128, 128)
    
    # Check normalized range [-1, 1]
    assert result.min() >= -1.0
    assert result.max() <= 1.0


def test_preprocess_image_with_median_denoising():
    """Test image preprocessing with median denoising enabled."""
    # Create a test image
    test_image = Image.new('RGB', (256, 256), color=(128, 128, 128))
    
    # Preprocess with median denoising
    result = preprocess_image(
        test_image,
        target_size=(128, 128),
        normalize=True,
        enable_denoising=True,
        denoising_method="median",
        median_kernel_size=5
    )
    
    # Check output is a tensor
    assert isinstance(result, torch.Tensor)
    
    # Check output shape
    assert result.shape == (3, 128, 128)
    
    # Check normalized range [-1, 1]
    assert result.min() >= -1.0
    assert result.max() <= 1.0


def test_preprocess_image_without_denoising():
    """Test image preprocessing with denoising disabled (default)."""
    # Create a test image
    test_image = Image.new('RGB', (256, 256), color=(128, 128, 128))
    
    # Preprocess without denoising (default)
    result = preprocess_image(
        test_image,
        target_size=(128, 128),
        normalize=True,
        enable_denoising=False
    )
    
    # Check output is a tensor
    assert isinstance(result, torch.Tensor)
    
    # Check output shape
    assert result.shape == (3, 128, 128)
    
    # Check normalized range [-1, 1]
    assert result.min() >= -1.0
    assert result.max() <= 1.0


def test_preprocess_image_invalid_denoising_method():
    """Test image preprocessing with invalid denoising method."""
    # Create a test image
    test_image = Image.new('RGB', (256, 256), color=(128, 128, 128))
    
    # Preprocess with invalid denoising method (should skip denoising)
    result = preprocess_image(
        test_image,
        target_size=(128, 128),
        normalize=True,
        enable_denoising=True,
        denoising_method="invalid_method"
    )
    
    # Should still work, just skip denoising
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 128, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
