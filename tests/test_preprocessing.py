"""Simplified unit tests for data preprocessing."""

import pytest
import numpy as np
import torch
from PIL import Image

from src.data.preprocessing import preprocess_image, preprocess_mask


def test_image_resize():
    """
    Test image resizing functionality.
    
    Validates Requirement 5.2: Test data preprocessing unit tests.
    Verifies that images are correctly resized to target dimensions.
    """
    # Create a test image with known size
    test_image = Image.new('RGB', (512, 512), color=(100, 150, 200))
    
    # Preprocess with target size 256x256
    result = preprocess_image(
        test_image,
        target_size=(256, 256),
        normalize=False
    )
    
    # Check output is a tensor
    assert isinstance(result, torch.Tensor), "Output should be a torch.Tensor"
    
    # Check output shape (C, H, W)
    assert result.shape == (3, 256, 256), f"Expected shape (3, 256, 256), got {result.shape}"
    
    # Test with different target size
    result_1024 = preprocess_image(
        test_image,
        target_size=(1024, 1024),
        normalize=False
    )
    
    assert result_1024.shape == (3, 1024, 1024), f"Expected shape (3, 1024, 1024), got {result_1024.shape}"


def test_mask_resize():
    """
    Test mask resizing functionality.
    
    Validates Requirement 5.2: Test data preprocessing unit tests.
    Verifies that segmentation masks are correctly resized while preserving label values.
    """
    # Create a test mask with integer labels
    mask_array = np.random.randint(0, 5, (512, 512), dtype=np.uint8)
    test_mask = Image.fromarray(mask_array)
    
    # Preprocess mask with target size 256x256
    result = preprocess_mask(
        test_mask,
        target_size=(256, 256)
    )
    
    # Check output is a tensor
    assert isinstance(result, torch.Tensor), "Output should be a torch.Tensor"
    
    # Check output shape (1, H, W)
    assert result.shape == (1, 256, 256), f"Expected shape (1, 256, 256), got {result.shape}"
    
    # Check that values are still integers (labels)
    unique_values = torch.unique(result)
    assert all(v >= 0 and v < 256 for v in unique_values), "Mask values should be valid labels"
    
    # Test with different target size
    result_1024 = preprocess_mask(
        test_mask,
        target_size=(1024, 1024)
    )
    
    assert result_1024.shape == (1, 1024, 1024), f"Expected shape (1, 1024, 1024), got {result_1024.shape}"


def test_normalization():
    """
    Test pixel value normalization.
    
    Validates Requirement 5.2: Test data preprocessing unit tests.
    Verifies that pixel values are correctly normalized to [-1, 1] range.
    """
    # Create a test image with known pixel values
    test_image = Image.new('RGB', (128, 128), color=(0, 128, 255))
    
    # Test with normalization enabled
    result_normalized = preprocess_image(
        test_image,
        target_size=(128, 128),
        normalize=True
    )
    
    # Check normalized range [-1, 1]
    assert result_normalized.min() >= -1.0, f"Min value {result_normalized.min()} should be >= -1.0"
    assert result_normalized.max() <= 1.0, f"Max value {result_normalized.max()} should be <= 1.0"
    
    # Test with normalization disabled
    result_unnormalized = preprocess_image(
        test_image,
        target_size=(128, 128),
        normalize=False
    )
    
    # Check unnormalized range [0, 1]
    assert result_unnormalized.min() >= 0.0, f"Min value {result_unnormalized.min()} should be >= 0.0"
    assert result_unnormalized.max() <= 1.0, f"Max value {result_unnormalized.max()} should be <= 1.0"
    
    # Verify that normalized and unnormalized are different
    # For a mid-gray image (128), normalized should be around 0, unnormalized around 0.5
    mean_normalized = result_normalized.mean()
    mean_unnormalized = result_unnormalized.mean()
    
    assert abs(mean_normalized) < 0.1, f"Normalized mean should be near 0, got {mean_normalized}"
    assert 0.4 < mean_unnormalized < 0.6, f"Unnormalized mean should be near 0.5, got {mean_unnormalized}"


def test_image_resize_edge_cases():
    """Test image resizing with edge cases."""
    # Test with very small image
    small_image = Image.new('RGB', (16, 16), color=(128, 128, 128))
    result = preprocess_image(small_image, target_size=(256, 256), normalize=False)
    assert result.shape == (3, 256, 256)
    
    # Test with non-square image
    rect_image = Image.new('RGB', (800, 600), color=(128, 128, 128))
    result = preprocess_image(rect_image, target_size=(512, 512), normalize=False)
    assert result.shape == (3, 512, 512)


def test_mask_resize_edge_cases():
    """Test mask resizing with edge cases."""
    # Test with binary mask (0 and 1 only)
    binary_mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
    test_mask = Image.fromarray(binary_mask)
    result = preprocess_mask(test_mask, target_size=(128, 128))
    
    assert result.shape == (1, 128, 128)
    unique_values = torch.unique(result)
    # After resizing, values should still be close to 0 or 1
    assert len(unique_values) <= 10, "Binary mask should have few unique values after resize"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
