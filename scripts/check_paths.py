#!/usr/bin/env python3
"""
Path Checking Script for Server Adaptation

This script checks all YAML configuration files for paths and validates them
against server restrictions. It ensures all paths are either relative or use
the allowed server prefix: /home/Backup/maziheng

Requirements: 3.1, 3.2, 3.3, 3.4
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import yaml


def check_yaml_paths(yaml_file: str) -> List[Tuple[str, str]]:
    """
    Check YAML file for paths and identify invalid ones.
    
    Args:
        yaml_file: Path to YAML file to check
        
    Returns:
        List of tuples (key_path, invalid_path) for paths that don't meet requirements
    """
    if not os.path.exists(yaml_file):
        print(f"⚠️  Warning: File not found: {yaml_file}")
        return []
    
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"✗ Error reading {yaml_file}: {e}")
        return []
    
    invalid_paths = []
    
    def check_value(value: Any, key_path: str = ""):
        """Recursively check values in the config"""
        if isinstance(value, dict):
            for k, v in value.items():
                new_key_path = f"{key_path}.{k}" if key_path else k
                check_value(v, new_key_path)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                new_key_path = f"{key_path}[{i}]"
                check_value(item, new_key_path)
        elif isinstance(value, str):
            # Check if this looks like a path
            if _looks_like_path(value):
                if not is_valid_path(value):
                    invalid_paths.append((key_path, value))
    
    check_value(config)
    return invalid_paths


def _looks_like_path(value: str) -> bool:
    """
    Determine if a string value looks like a file path.
    
    Args:
        value: String to check
        
    Returns:
        True if the string appears to be a path
    """
    # Skip empty strings
    if not value or len(value) == 0:
        return False
    
    # Skip very short strings (likely not paths)
    if len(value) < 3:
        return False
    
    # Skip strings that are clearly not paths
    if value in ['cuda', 'cpu', 'ddpm', 'ddim', 'png', 'jpg', 'jpeg']:
        return False
    
    # Check for path indicators
    path_indicators = [
        '/',           # Contains slash
        '\\',          # Contains backslash
        'data/',       # Starts with common directory names
        'logs/',
        'checkpoints/',
        'results/',
        'configs/',
        'models/',
        '.pth',        # Contains file extensions
        '.yaml',
        '.json',
        '.jsonl',
        '.txt',
        '.log',
    ]
    
    return any(indicator in value for indicator in path_indicators)


def is_valid_path(path: str, allowed_prefix: str = "/home/Backup/maziheng") -> bool:
    """
    Check if a path is valid according to server restrictions.
    
    A path is valid if it is:
    1. A relative path (doesn't start with /)
    2. An absolute path starting with the allowed prefix
    
    Args:
        path: Path to validate
        allowed_prefix: Allowed absolute path prefix
        
    Returns:
        True if path is valid, False otherwise
    """
    # Empty paths are considered invalid
    if not path:
        return False
    
    # Relative paths are always valid
    if not path.startswith('/'):
        return True
    
    # Absolute paths must start with allowed prefix
    return path.startswith(allowed_prefix)


def convert_to_relative_path(abs_path: str, base_dir: str = None) -> str:
    """
    Convert an absolute path to a relative path.
    
    Args:
        abs_path: Absolute path to convert
        base_dir: Base directory for relative path (defaults to current directory)
        
    Returns:
        Relative path suggestion
    """
    if not abs_path.startswith('/'):
        return abs_path  # Already relative
    
    if base_dir is None:
        base_dir = os.getcwd()
    
    try:
        # Try to compute relative path
        rel_path = os.path.relpath(abs_path, base_dir)
        return rel_path
    except ValueError:
        # If paths are on different drives (Windows), suggest using project root
        # Extract the last meaningful part of the path
        parts = abs_path.split('/')
        if len(parts) > 0:
            # Find common directory names and suggest relative path
            for i, part in enumerate(parts):
                if part in ['data', 'logs', 'checkpoints', 'results', 'configs', 'models']:
                    return '/'.join(parts[i:])
        
        return abs_path  # Return original if conversion fails


def main():
    """
    Main function: Check all configuration files and generate report.
    """
    print("=" * 60)
    print("路径检查报告 (Path Check Report)")
    print("=" * 60)
    print()
    
    # Configuration files to check
    config_files = [
        'configs/train_config.yaml',
        'configs/inference_config.yaml',
        'configs/train_config_fast.yaml',  # Check fast config if exists
    ]
    
    # Filter to only existing files
    existing_files = [f for f in config_files if os.path.exists(f)]
    
    if not existing_files:
        print("✗ 未找到配置文件")
        print("  请确保在项目根目录运行此脚本")
        return 1
    
    print(f"检查 {len(existing_files)} 个配置文件...")
    print()
    
    all_issues = {}
    total_invalid = 0
    
    for config_file in existing_files:
        print(f"检查: {config_file}")
        invalid_paths = check_yaml_paths(config_file)
        
        if invalid_paths:
            all_issues[config_file] = invalid_paths
            total_invalid += len(invalid_paths)
            print(f"  ⚠️  发现 {len(invalid_paths)} 个问题路径")
        else:
            print(f"  ✓ 所有路径检查通过")
        print()
    
    # Generate detailed report
    if all_issues:
        print("=" * 60)
        print("⚠️  发现路径问题")
        print("=" * 60)
        print()
        
        for config_file, invalid_paths in all_issues.items():
            print(f"文件: {config_file}")
            print("-" * 60)
            
            for key_path, path in invalid_paths:
                print(f"  配置项: {key_path}")
                print(f"  当前路径: {path}")
                
                # Provide conversion suggestion
                if path.startswith('/'):
                    suggested = convert_to_relative_path(path)
                    print(f"  建议修改为: {suggested}")
                else:
                    print(f"  建议: 使用相对路径或 /home/Backup/maziheng 前缀")
                
                print()
        
        print("=" * 60)
        print(f"总计: {total_invalid} 个路径需要修改")
        print("=" * 60)
        print()
        print("修改建议:")
        print("1. 使用相对路径（推荐）: data/train, checkpoints/, logs/")
        print("2. 使用服务器限制路径: /home/Backup/maziheng/project/data")
        print("3. 在配置文件中添加路径说明注释")
        print()
        
        return 1
    else:
        print("=" * 60)
        print("✓ 所有路径检查通过")
        print("=" * 60)
        print()
        print("所有配置文件中的路径都符合服务器限制:")
        print("- 使用相对路径，或")
        print("- 使用 /home/Backup/maziheng 前缀的绝对路径")
        print()
        
        return 0


if __name__ == "__main__":
    sys.exit(main())
