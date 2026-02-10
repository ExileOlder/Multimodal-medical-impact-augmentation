"""
Tests for structure consistency evaluation metrics (Dice coefficient and IoU).
"""

import numpy as np
import pytest
from PIL import Image

from src.evaluation.metrics import (
    calculate_dice_coefficient,
    calculate_iou,
    extract_lesion_mask_from_image
)


def test_dice_coefficient_perfect_match():
    """Test Dice coefficient with identical masks (should be 1.0)."""
    mask1 = np.ones((100, 100)) * 255
    mask2 = np.ones((100, 100)) * 255
    
    dice = calculate_dice_coefficient(mask1, mask2)
    assert dice == 1.0, f"Expected 1.0, got {dice}"


def test_dice_coefficient_no_overlap():
    """Test Dice coefficient with no overlap (should be 0.0)."""
    mask1 = np.zeros((100, 100))
    mask1[:50, :] = 255  # Top half
    
    mask2 = np.zeros((100, 100))
    mask2[50:, :] = 255  # Bottom half
    
    dice = calculate_dice_coefficient(mask1, mask2)
    assert dice == 0.0, f"Expected 0.0, got {dice}"


def test_dice_coefficient_partial_overlap():
    """Test Dice coefficient with partial overlap."""
    mask1 = np.zeros((100, 100))
    mask1[:60, :] = 255  # Top 60%
    
    mask2 = np.zeros((100, 100))
    mask2[40:, :] = 255  # Bottom 60%
    
    dice = calculate_dice_coefficient(mask1, mask2)
    # Intersection: 20 rows, Union: 60 + 60 = 120 rows
    # Dice = 2 * 20*100 / (60*100 + 60*100) = 2 * 2000 / 12000 = 0.333...
    assert 0.3 < dice < 0.4, f"Expected ~0.33, got {dice}"


def test_dice_coefficient_empty_masks():
    """Test Dice coefficient with both masks empty (should be 1.0)."""
    mask1 = np.zeros((100, 100))
    mask2 = np.zeros((100, 100))
    
    dice = calculate_dice_coefficient(mask1, mask2)
    assert dice == 1.0, f"Expected 1.0 for empty masks, got {dice}"


def test_iou_perfect_match():
    """Test IoU with identical masks (should be 1.0)."""
    mask1 = np.ones((100, 100)) * 255
    mask2 = np.ones((100, 100)) * 255
    
    iou = calculate_iou(mask1, mask2)
    assert iou == 1.0, f"Expected 1.0, got {iou}"


def test_iou_no_overlap():
    """Test IoU with no overlap (should be 0.0)."""
    mask1 = np.zeros((100, 100))
    mask1[:50, :] = 255  # Top half
    
    mask2 = np.zeros((100, 100))
    mask2[50:, :] = 255  # Bottom half
    
    iou = calculate_iou(mask1, mask2)
    assert iou == 0.0, f"Expected 0.0, got {iou}"


def test_iou_partial_overlap():
    """Test IoU with partial overlap."""
    mask1 = np.zeros((100, 100))
    mask1[:60, :] = 255  # Top 60%
    
    mask2 = np.zeros((100, 100))
    mask2[40:, :] = 255  # Bottom 60%
    
    iou = calculate_iou(mask1, mask2)
    # Intersection: 20 rows, Union: 100 rows
    # IoU = 20*100 / 100*100 = 0.2
    assert 0.19 < iou < 0.21, f"Expected ~0.2, got {iou}"


def test_iou_empty_masks():
    """Test IoU with both masks empty (should be 1.0)."""
    mask1 = np.zeros((100, 100))
    mask2 = np.zeros((100, 100))
    
    iou = calculate_iou(mask1, mask2)
    assert iou == 1.0, f"Expected 1.0 for empty masks, got {iou}"


def test_dice_and_iou_with_pil_images():
    """Test that Dice and IoU work with PIL Images."""
    # Create PIL images
    mask1_array = np.ones((100, 100), dtype=np.uint8) * 255
    mask2_array = np.ones((100, 100), dtype=np.uint8) * 255
    
    mask1 = Image.fromarray(mask1_array)
    mask2 = Image.fromarray(mask2_array)
    
    dice = calculate_dice_coefficient(mask1, mask2)
    iou = calculate_iou(mask1, mask2)
    
    assert dice == 1.0, f"Expected Dice=1.0, got {dice}"
    assert iou == 1.0, f"Expected IoU=1.0, got {iou}"


def test_extract_lesion_mask_red_channel():
    """Test lesion mask extraction using red channel method."""
    # Create a synthetic image with red regions
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[25:75, 25:75, 0] = 200  # Red region in center
    
    mask = extract_lesion_mask_from_image(image, method="red_channel", threshold=0.5)
    
    # Check that mask is binary
    assert set(np.unique(mask)).issubset({0.0, 1.0}), "Mask should be binary"
    
    # Check that center region is detected
    center_sum = np.sum(mask[25:75, 25:75])
    assert center_sum > 0, "Center region should be detected"


def test_extract_lesion_mask_brightness():
    """Test lesion mask extraction using brightness method."""
    # Create a synthetic image with dark regions
    image = np.ones((100, 100, 3), dtype=np.uint8) * 200
    image[25:75, 25:75, :] = 50  # Dark region in center
    
    mask = extract_lesion_mask_from_image(image, method="brightness", threshold=0.5)
    
    # Check that mask is binary
    assert set(np.unique(mask)).issubset({0.0, 1.0}), "Mask should be binary"
    
    # Check that center region is detected
    center_sum = np.sum(mask[25:75, 25:75])
    assert center_sum > 0, "Center dark region should be detected"


def test_dice_coefficient_with_different_shapes():
    """Test that Dice coefficient raises error for different shapes."""
    mask1 = np.ones((100, 100)) * 255
    mask2 = np.ones((50, 50)) * 255
    
    with pytest.raises(ValueError, match="must have same shape"):
        calculate_dice_coefficient(mask1, mask2)


def test_iou_with_different_shapes():
    """Test that IoU raises error for different shapes."""
    mask1 = np.ones((100, 100)) * 255
    mask2 = np.ones((50, 50)) * 255
    
    with pytest.raises(ValueError, match="must have same shape"):
        calculate_iou(mask1, mask2)


def test_dice_coefficient_with_threshold():
    """Test Dice coefficient with different thresholds."""
    # Create masks with intermediate values
    mask1 = np.ones((100, 100)) * 128  # Gray
    mask2 = np.ones((100, 100)) * 128  # Gray
    
    # With threshold 0.3, both should be considered "positive"
    dice_low = calculate_dice_coefficient(mask1, mask2, threshold=0.3)
    assert dice_low == 1.0, f"Expected 1.0 with low threshold, got {dice_low}"
    
    # With threshold 0.7, both should be considered "negative"
    dice_high = calculate_dice_coefficient(mask1, mask2, threshold=0.7)
    assert dice_high == 1.0, f"Expected 1.0 with high threshold (both empty), got {dice_high}"


if __name__ == "__main__":
    print("Running structure consistency tests...")
    
    # Run all tests
    test_dice_coefficient_perfect_match()
    print("✓ Dice coefficient perfect match test passed")
    
    test_dice_coefficient_no_overlap()
    print("✓ Dice coefficient no overlap test passed")
    
    test_dice_coefficient_partial_overlap()
    print("✓ Dice coefficient partial overlap test passed")
    
    test_dice_coefficient_empty_masks()
    print("✓ Dice coefficient empty masks test passed")
    
    test_iou_perfect_match()
    print("✓ IoU perfect match test passed")
    
    test_iou_no_overlap()
    print("✓ IoU no overlap test passed")
    
    test_iou_partial_overlap()
    print("✓ IoU partial overlap test passed")
    
    test_iou_empty_masks()
    print("✓ IoU empty masks test passed")
    
    test_dice_and_iou_with_pil_images()
    print("✓ Dice and IoU with PIL images test passed")
    
    test_extract_lesion_mask_red_channel()
    print("✓ Extract lesion mask (red channel) test passed")
    
    test_extract_lesion_mask_brightness()
    print("✓ Extract lesion mask (brightness) test passed")
    
    test_dice_coefficient_with_different_shapes()
    print("✓ Dice coefficient shape validation test passed")
    
    test_iou_with_different_shapes()
    print("✓ IoU shape validation test passed")
    
    test_dice_coefficient_with_threshold()
    print("✓ Dice coefficient threshold test passed")
    
    print("\n✅ All structure consistency tests passed!")
