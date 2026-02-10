#!/usr/bin/env python3
"""
Unit tests for validate_config.py

Tests the configuration validation functions to ensure they correctly
validate batch configurations, estimate GPU memory, and check paths.
"""

import sys
import os
import pytest
import tempfile
import yaml

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from validate_config import (
    validate_batch_config,
    estimate_gpu_memory,
    validate_paths
)


class TestValidateBatchConfig:
    """Tests for validate_batch_config function."""
    
    def test_normal_batch_size(self):
        """Test with normal batch size."""
        valid, msg = validate_batch_config(32, 1)
        assert valid is True
        assert "32" in msg
        assert "✓" in msg
    
    def test_small_batch_with_gradient_accumulation(self):
        """Test small batch size with gradient accumulation."""
        valid, msg = validate_batch_config(2, 16)
        assert valid is True
        assert "32" in msg  # Effective batch size
    
    def test_effective_batch_size_too_small(self):
        """Test when effective batch size is too small."""
        valid, msg = validate_batch_config(1, 8)
        assert valid is False
        assert "too small" in msg.lower()
        assert "8" in msg  # Effective batch size
    
    def test_effective_batch_size_acceptable_but_small(self):
        """Test when effective batch size is acceptable but small."""
        valid, msg = validate_batch_config(2, 8)
        assert valid is True
        assert "⚠️" in msg or "acceptable" in msg.lower()
        assert "16" in msg  # Effective batch size
    
    def test_invalid_batch_size(self):
        """Test with invalid batch size."""
        valid, msg = validate_batch_config(0, 1)
        assert valid is False
        assert "must be at least 1" in msg.lower()
    
    def test_large_effective_batch_size(self):
        """Test with large effective batch size."""
        valid, msg = validate_batch_config(64, 1)
        assert valid is True
        assert "64" in msg


class TestEstimateGPUMemory:
    """Tests for estimate_gpu_memory function."""
    
    def test_small_model(self):
        """Test memory estimation for small model."""
        config = {
            "model": {"hidden_size": 512, "depth": 12, "patch_size": 2},
            "data": {"batch_size": 4, "image_size": 512},
            "training": {"use_amp": True}
        }
        memory = estimate_gpu_memory(config)
        assert memory > 0
        assert memory < 10  # Should be less than 10GB for small model
    
    def test_medium_model(self):
        """Test memory estimation for medium model."""
        config = {
            "model": {"hidden_size": 1152, "depth": 16, "patch_size": 2},
            "data": {"batch_size": 2, "image_size": 1024},
            "training": {"use_amp": True}
        }
        memory = estimate_gpu_memory(config)
        assert memory > 5
        assert memory < 50  # Should be reasonable for medium model
    
    def test_memory_increases_with_batch_size(self):
        """Test that memory increases with batch size."""
        config_small = {
            "model": {"hidden_size": 512, "depth": 12, "patch_size": 2},
            "data": {"batch_size": 2, "image_size": 512},
            "training": {"use_amp": True}
        }
        config_large = {
            "model": {"hidden_size": 512, "depth": 12, "patch_size": 2},
            "data": {"batch_size": 8, "image_size": 512},
            "training": {"use_amp": True}
        }
        memory_small = estimate_gpu_memory(config_small)
        memory_large = estimate_gpu_memory(config_large)
        assert memory_large > memory_small
    
    def test_memory_increases_with_image_size(self):
        """Test that memory increases with image size."""
        config_small = {
            "model": {"hidden_size": 512, "depth": 12, "patch_size": 2},
            "data": {"batch_size": 4, "image_size": 256},
            "training": {"use_amp": True}
        }
        config_large = {
            "model": {"hidden_size": 512, "depth": 12, "patch_size": 2},
            "data": {"batch_size": 4, "image_size": 1024},
            "training": {"use_amp": True}
        }
        memory_small = estimate_gpu_memory(config_small)
        memory_large = estimate_gpu_memory(config_large)
        assert memory_large > memory_small
    
    def test_amp_reduces_memory(self):
        """Test that mixed precision reduces memory usage."""
        config_fp32 = {
            "model": {"hidden_size": 512, "depth": 12, "patch_size": 2},
            "data": {"batch_size": 4, "image_size": 512},
            "training": {"use_amp": False}
        }
        config_amp = {
            "model": {"hidden_size": 512, "depth": 12, "patch_size": 2},
            "data": {"batch_size": 4, "image_size": 512},
            "training": {"use_amp": True}
        }
        memory_fp32 = estimate_gpu_memory(config_fp32)
        memory_amp = estimate_gpu_memory(config_amp)
        assert memory_amp < memory_fp32


class TestValidatePaths:
    """Tests for validate_paths function."""
    
    def test_relative_paths_valid(self):
        """Test that relative paths are valid."""
        config = {
            "data": {
                "train_data_path": "data/train",
                "val_data_path": "data/val"
            },
            "training": {
                "checkpoint_dir": "checkpoints",
                "log_dir": "logs"
            }
        }
        valid, issues = validate_paths(config, "configs/train_config.yaml")
        assert valid is True
        assert len(issues) == 0
    
    def test_allowed_absolute_paths_valid(self):
        """Test that absolute paths within allowed prefix are valid."""
        config = {
            "data": {
                "train_data_path": "/home/Backup/maziheng/data/train"
            }
        }
        valid, issues = validate_paths(config, "configs/train_config.yaml")
        assert valid is True
        assert len(issues) == 0
    
    def test_disallowed_absolute_paths_invalid(self):
        """Test that absolute paths outside allowed prefix are invalid."""
        config = {
            "data": {
                "train_data_path": "/some/other/path/data"
            }
        }
        valid, issues = validate_paths(config, "configs/train_config.yaml")
        assert valid is False
        assert len(issues) > 0
        assert "/some/other/path/data" in str(issues)
    
    def test_nested_paths_checked(self):
        """Test that nested paths are checked."""
        config = {
            "data": {
                "paths": {
                    "train_path": "/invalid/path/train",
                    "val_path": "data/val"
                }
            }
        }
        valid, issues = validate_paths(config, "configs/train_config.yaml")
        assert valid is False
        assert len(issues) > 0
    
    def test_empty_paths_ignored(self):
        """Test that empty path strings are ignored."""
        config = {
            "data": {
                "train_data_path": "",
                "val_data_path": "data/val"
            }
        }
        valid, issues = validate_paths(config, "configs/train_config.yaml")
        assert valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
