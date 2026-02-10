#!/usr/bin/env python3
"""
Configuration Validation Script

This script validates training configuration files to ensure they are suitable
for the A100 GPU environment and follow best practices.

Requirements validated:
- 4.1: Reasonable batch_size and gradient_accumulation_steps
- 4.2: Effective Batch Size >= 16 (preferably >= 32)
- 7.2: Batch configuration validation
- 7.3: Path validation (within /home/Backup/maziheng)
- 7.4: GPU memory estimation
"""

import os
import sys
import yaml
from typing import Tuple, Dict, Any
from pathlib import Path


def validate_batch_config(batch_size: int, grad_accum_steps: int = 1) -> Tuple[bool, str]:
    """
    Validate batch configuration.
    
    Args:
        batch_size: Batch size per GPU
        grad_accum_steps: Gradient accumulation steps
        
    Returns:
        (is_valid, message): Validation result and message
    """
    effective_batch_size = batch_size * grad_accum_steps
    
    # Check if batch_size is too small
    if batch_size < 1:
        return False, f"batch_size ({batch_size}) must be at least 1"
    
    # Check if effective batch size is too small
    if effective_batch_size < 16:
        return False, (
            f"Effective Batch Size ({effective_batch_size}) is too small. "
            f"Recommended: >= 32 for stable training. "
            f"Current: batch_size={batch_size} × gradient_accumulation_steps={grad_accum_steps}"
        )
    
    # Warn if effective batch size is small but acceptable
    if effective_batch_size < 32:
        return True, (
            f"⚠️  Effective Batch Size ({effective_batch_size}) is acceptable but small. "
            f"Recommended: >= 32 for optimal training stability. "
            f"Current: batch_size={batch_size} × gradient_accumulation_steps={grad_accum_steps}"
        )
    
    # All good
    return True, (
        f"✓ Batch configuration is good. "
        f"Effective Batch Size: {effective_batch_size} "
        f"(batch_size={batch_size} × gradient_accumulation_steps={grad_accum_steps})"
    )


def estimate_gpu_memory(config: Dict[str, Any]) -> float:
    """
    Estimate GPU memory requirements (rough estimation).
    
    This is a simplified estimation based on:
    - Model parameters
    - Activation memory
    - Optimizer states
    - Batch size
    
    Note: This is a rough estimate and actual memory usage may vary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Estimated GPU memory in GB
    """
    # Extract configuration
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    
    # Model parameters
    hidden_size = model_config.get('hidden_size', model_config.get('dim', 1152))
    depth = model_config.get('depth', model_config.get('n_layers', 28))
    num_heads = model_config.get('num_heads', model_config.get('n_heads', 16))
    
    # Data parameters
    batch_size = data_config.get('batch_size', training_config.get('batch_size', 32))
    image_size = data_config.get('image_size', 1024)
    in_channels = model_config.get('in_channels', 3)
    
    # Estimate model parameters (simplified)
    # Transformer: approximately 12 * hidden_size^2 * depth parameters
    # This includes attention (4 * hidden_size^2) and FFN (8 * hidden_size^2) per layer
    model_params = 12 * (hidden_size ** 2) * depth
    
    # If using mixed precision, parameters are stored in fp32 but computations in fp16/bf16
    use_amp = training_config.get('use_amp', False)
    param_dtype_bytes = 4  # fp32 for parameters
    activation_dtype_bytes = 2 if use_amp else 4  # fp16/bf16 or fp32
    
    # Model memory (parameters)
    model_memory_gb = model_params * param_dtype_bytes / 1e9
    
    # Estimate activation memory (simplified)
    # For transformers: batch_size * sequence_length * hidden_size * num_layers
    patch_size = model_config.get('patch_size', 2)
    num_patches = (image_size // patch_size) ** 2
    
    # Activations are stored per layer, but not all at once due to gradient checkpointing
    # Rough estimate: batch_size * num_patches * hidden_size * sqrt(depth)
    # Using sqrt(depth) as approximation for gradient checkpointing
    activation_memory_gb = (
        batch_size * num_patches * hidden_size * (depth ** 0.5) * activation_dtype_bytes / 1e9
    )
    
    # Optimizer states (AdamW: 2x model parameters for momentum and variance)
    optimizer_memory_gb = model_memory_gb * 2
    
    # Gradients (same size as model parameters)
    gradient_memory_gb = model_memory_gb
    
    # Total memory
    total_memory_gb = (
        model_memory_gb +
        activation_memory_gb +
        optimizer_memory_gb +
        gradient_memory_gb
    )
    
    # Add 30% overhead for PyTorch, CUDA, and other buffers
    total_memory_gb *= 1.3
    
    return total_memory_gb


def validate_paths(config: Dict[str, Any], config_file: str) -> Tuple[bool, list]:
    """
    Validate that all paths in config are either relative or within allowed prefix.
    
    Args:
        config: Configuration dictionary
        config_file: Path to config file (for relative path resolution)
        
    Returns:
        (is_valid, issues): Validation result and list of issues
    """
    allowed_prefix = "/home/Backup/maziheng"
    issues = []
    
    def check_path_recursive(obj, path_prefix=""):
        """Recursively check all string values that look like paths."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path_prefix}.{key}" if path_prefix else key
                
                # Check if this looks like a path field
                if any(keyword in key.lower() for keyword in ['path', 'dir', 'file']):
                    if isinstance(value, str) and value:
                        # Check if it's an absolute path (Unix-style or Windows-style)
                        is_absolute = (
                            value.startswith('/') or  # Unix absolute path
                            (len(value) > 1 and value[1] == ':')  # Windows absolute path (C:, D:, etc.)
                        )
                        
                        if is_absolute:
                            # Check if it's within allowed prefix
                            if not value.startswith(allowed_prefix):
                                issues.append(
                                    f"{current_path}: '{value}' is an absolute path "
                                    f"not within {allowed_prefix}"
                                )
                
                # Recurse into nested structures
                if isinstance(value, (dict, list)):
                    check_path_recursive(value, current_path)
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_path = f"{path_prefix}[{i}]"
                if isinstance(item, (dict, list)):
                    check_path_recursive(item, current_path)
    
    check_path_recursive(config)
    
    return len(issues) == 0, issues


def main():
    """Main function: validate configuration file."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate training configuration for A100 GPU environment"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to configuration file (default: configs/train_config.yaml)'
    )
    args = parser.parse_args()
    
    config_file = args.config
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"❌ Error: Configuration file not found: {config_file}")
        return 1
    
    # Load configuration
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error loading configuration file: {e}")
        return 1
    
    print("=" * 60)
    print("Configuration Validation Report")
    print("=" * 60)
    print(f"Config file: {config_file}\n")
    
    all_valid = True
    
    # 1. Validate batch configuration
    print("1. Batch Configuration")
    print("-" * 60)
    
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    
    # Get batch_size from either data or training section
    batch_size = data_config.get('batch_size', training_config.get('batch_size', None))
    
    if batch_size is None:
        print("❌ Error: batch_size not found in configuration")
        all_valid = False
    else:
        grad_accum_steps = training_config.get('gradient_accumulation_steps', 1)
        
        valid, msg = validate_batch_config(batch_size, grad_accum_steps)
        
        if valid:
            print(msg)
        else:
            print(f"❌ {msg}")
            all_valid = False
            
            # Provide suggestions
            print("\n💡 Suggestions:")
            if batch_size < 4:
                suggested_grad_accum = max(8, 32 // batch_size)
                print(f"   - Increase gradient_accumulation_steps to {suggested_grad_accum}")
                print(f"     (to achieve Effective Batch Size = {batch_size * suggested_grad_accum})")
            else:
                suggested_batch_size = 4
                suggested_grad_accum = 8
                print(f"   - Set batch_size = {suggested_batch_size}")
                print(f"   - Set gradient_accumulation_steps = {suggested_grad_accum}")
                print(f"     (Effective Batch Size = {suggested_batch_size * suggested_grad_accum})")
    
    print()
    
    # 2. Estimate GPU memory
    print("2. GPU Memory Estimation")
    print("-" * 60)
    
    try:
        memory_gb = estimate_gpu_memory(config)
        print(f"Estimated GPU memory requirement: {memory_gb:.2f} GB")
        
        # A100 has 40GB or 80GB variants
        a100_memory = 40  # Assume 40GB variant
        
        if memory_gb > a100_memory:
            print(f"⚠️  Warning: Estimated memory ({memory_gb:.2f} GB) exceeds A100 {a100_memory}GB")
            print("\n💡 Suggestions to reduce memory:")
            print("   - Reduce batch_size")
            print("   - Reduce image_size")
            print("   - Reduce model hidden_size or depth")
            print("   - Enable gradient checkpointing (if not already enabled)")
            all_valid = False
        elif memory_gb > a100_memory * 0.8:
            print(f"⚠️  Warning: Estimated memory ({memory_gb:.2f} GB) is close to A100 {a100_memory}GB limit")
            print("   Consider reducing batch_size for safety margin")
        else:
            print(f"✓ Memory requirement is within A100 {a100_memory}GB limit")
            print(f"  (Safety margin: {a100_memory - memory_gb:.2f} GB)")
    
    except Exception as e:
        print(f"⚠️  Warning: Could not estimate GPU memory: {e}")
    
    print()
    
    # 3. Validate paths
    print("3. Path Validation")
    print("-" * 60)
    
    paths_valid, path_issues = validate_paths(config, config_file)
    
    if paths_valid:
        print("✓ All paths are valid (relative or within /home/Backup/maziheng)")
    else:
        print("❌ Found path issues:")
        for issue in path_issues:
            print(f"   - {issue}")
        print("\n💡 Suggestion: Use relative paths instead of absolute paths")
        all_valid = False
    
    print()
    
    # 4. Additional checks
    print("4. Additional Checks")
    print("-" * 60)
    
    # Check mixed precision
    use_amp = training_config.get('use_amp', False)
    amp_dtype = training_config.get('amp_dtype', 'float16')
    
    if use_amp:
        if amp_dtype == 'bfloat16':
            print("✓ Mixed precision enabled with bfloat16 (optimal for A100)")
        else:
            print(f"⚠️  Mixed precision enabled with {amp_dtype}")
            print("   💡 Consider using 'bfloat16' for better A100 performance")
    else:
        print("⚠️  Mixed precision not enabled")
        print("   💡 Enable use_amp=true and amp_dtype='bfloat16' for A100 optimization")
    
    # Check num_workers
    num_workers = data_config.get('num_workers', 0)
    if num_workers >= 4:
        print(f"✓ num_workers = {num_workers} (good for data loading)")
    else:
        print(f"⚠️  num_workers = {num_workers} (consider increasing to 4-8)")
    
    print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    if all_valid:
        print("✅ Configuration validation passed!")
        print("   The configuration is suitable for A100 GPU training.")
        return 0
    else:
        print("❌ Configuration validation failed!")
        print("   Please address the issues above before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
