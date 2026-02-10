"""Simplified unit tests for model functionality."""

import pytest
import torch

from src.models.nexdit_mask import NextDiTWithMask


def test_model_initialization():
    """
    Test model initialization.
    
    Validates Requirement 5.4: Test model initialization.
    Verifies that the NextDiTWithMask model can be initialized with various configurations.
    """
    # Test with small configuration for fast testing
    model = NextDiTWithMask(
        patch_size=2,
        in_channels=3,
        mask_channels=1,
        dim=512,
        n_layers=4,
        n_heads=8
    )
    
    # Check model is created
    assert model is not None, "Model should be initialized"
    
    # Check model attributes
    assert model.in_channels == 3, "Input channels should be 3"
    assert model.mask_channels == 1, "Mask channels should be 1"
    assert model.total_in_channels == 4, "Total input channels should be 4 (3 + 1)"
    assert model.patch_size == 2, "Patch size should be 2"
    
    # Check model has base_model
    assert hasattr(model, 'base_model'), "Model should have base_model attribute"
    
    # Test with different configuration
    model_large = NextDiTWithMask(
        patch_size=2,
        in_channels=3,
        mask_channels=1,
        dim=1024,
        n_layers=8,
        n_heads=16
    )
    
    assert model_large is not None, "Large model should be initialized"
    assert model_large.total_in_channels == 4, "Total input channels should be 4"


def test_forward_pass():
    """
    Test forward pass through the model.
    
    Validates Requirement 5.5: Test forward propagation.
    Verifies that the model can perform forward pass with correct input/output shapes.
    """
    # Create a small model for testing
    model = NextDiTWithMask(
        patch_size=2,
        in_channels=3,
        mask_channels=1,
        dim=512,
        n_layers=4,
        n_heads=8,
        cap_feat_dim=512  # Match caption feature dimension
    )
    
    # Set model to eval mode
    model.eval()
    
    # Create test inputs
    batch_size = 2
    height, width = 64, 64
    seq_len = 77
    
    x = torch.randn(batch_size, 3, height, width)
    t = torch.rand(batch_size)
    cap_feats = torch.randn(batch_size, seq_len, 512)
    cap_mask = torch.ones(batch_size, seq_len)
    condition_mask = torch.randn(batch_size, 1, height, width)
    
    # Forward pass
    with torch.no_grad():
        output = model(x, t, cap_feats, cap_mask, condition_mask)
    
    # Check output shape
    expected_shape = (batch_size, 3, height, width)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"
    
    # Check output is not NaN or Inf
    assert not torch.isnan(output).any(), "Output should not contain NaN values"
    assert not torch.isinf(output).any(), "Output should not contain Inf values"


def test_forward_pass_without_mask():
    """
    Test forward pass without condition mask (text-only mode).
    
    Validates Requirement 5.5: Test forward propagation.
    Verifies that the model can handle text-only inputs without mask conditioning.
    """
    # Create a small model for testing
    model = NextDiTWithMask(
        patch_size=2,
        in_channels=3,
        mask_channels=1,
        dim=512,
        n_layers=4,
        n_heads=8,
        cap_feat_dim=512
    )
    
    model.eval()
    
    # Create test inputs without mask
    batch_size = 2
    height, width = 64, 64
    seq_len = 77
    
    x = torch.randn(batch_size, 3, height, width)
    t = torch.rand(batch_size)
    cap_feats = torch.randn(batch_size, seq_len, 512)
    cap_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass without mask (should use zero mask internally)
    with torch.no_grad():
        output = model(x, t, cap_feats, cap_mask, condition_mask=None)
    
    # Check output shape
    expected_shape = (batch_size, 3, height, width)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"
    
    # Check output is valid
    assert not torch.isnan(output).any(), "Output should not contain NaN values"
    assert not torch.isinf(output).any(), "Output should not contain Inf values"


def test_model_with_different_image_sizes():
    """
    Test model with different input image sizes.
    
    Validates Requirement 5.5: Test forward propagation.
    Verifies that the model can handle various image resolutions.
    """
    model = NextDiTWithMask(
        patch_size=2,
        in_channels=3,
        mask_channels=1,
        dim=512,
        n_layers=4,
        n_heads=8,
        cap_feat_dim=512
    )
    
    model.eval()
    
    # Test with different image sizes
    test_sizes = [(32, 32), (64, 64), (128, 128)]
    
    for height, width in test_sizes:
        batch_size = 1
        seq_len = 77
        
        x = torch.randn(batch_size, 3, height, width)
        t = torch.rand(batch_size)
        cap_feats = torch.randn(batch_size, seq_len, 512)
        cap_mask = torch.ones(batch_size, seq_len)
        condition_mask = torch.randn(batch_size, 1, height, width)
        
        with torch.no_grad():
            output = model(x, t, cap_feats, cap_mask, condition_mask)
        
        expected_shape = (batch_size, 3, height, width)
        assert output.shape == expected_shape, f"For size {height}x{width}, expected {expected_shape}, got {output.shape}"


def test_model_parameter_count():
    """
    Test model parameter count method.
    
    Validates Requirement 5.4: Test model initialization.
    Verifies that the model can report its parameter count.
    """
    model = NextDiTWithMask(
        patch_size=2,
        in_channels=3,
        mask_channels=1,
        dim=512,
        n_layers=4,
        n_heads=8
    )
    
    # Get parameter count
    param_count = model.parameter_count()
    
    # Check that parameter count is positive
    assert param_count > 0, "Parameter count should be positive"
    
    # Check that it's a reasonable number (not too small or too large)
    assert param_count > 1_000_000, "Model should have at least 1M parameters"
    assert param_count < 10_000_000_000, "Model should have less than 10B parameters"


def test_model_batch_sizes():
    """
    Test model with different batch sizes.
    
    Validates Requirement 5.5: Test forward propagation.
    Verifies that the model can handle various batch sizes.
    """
    model = NextDiTWithMask(
        patch_size=2,
        in_channels=3,
        mask_channels=1,
        dim=512,
        n_layers=4,
        n_heads=8,
        cap_feat_dim=512
    )
    
    model.eval()
    
    # Test with different batch sizes
    test_batch_sizes = [1, 2, 4]
    
    for batch_size in test_batch_sizes:
        height, width = 64, 64
        seq_len = 77
        
        x = torch.randn(batch_size, 3, height, width)
        t = torch.rand(batch_size)
        cap_feats = torch.randn(batch_size, seq_len, 512)
        cap_mask = torch.ones(batch_size, seq_len)
        condition_mask = torch.randn(batch_size, 1, height, width)
        
        with torch.no_grad():
            output = model(x, t, cap_feats, cap_mask, condition_mask)
        
        expected_shape = (batch_size, 3, height, width)
        assert output.shape == expected_shape, f"For batch size {batch_size}, expected {expected_shape}, got {output.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
