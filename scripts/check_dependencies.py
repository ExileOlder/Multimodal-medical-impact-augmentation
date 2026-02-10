#!/usr/bin/env python3
"""
依赖检查脚本 (Dependency Check Script)

本脚本用于检查服务器上已安装的包，避免重复安装和版本冲突。
功能包括：
1. 检查已安装的包及其版本
2. 检查 PyTorch 和 CUDA 版本兼容性
3. 列出缺失的包
4. 生成依赖检查报告

Requirements: 2.4, 8.1, 8.2, 8.3, 8.4
"""

import sys
import re
from typing import Dict, List, Tuple
from pathlib import Path


def check_installed_packages() -> Dict[str, str]:
    """
    检查已安装的包及其版本
    
    Returns:
        Dict[str, str]: 包名到版本的映射
    """
    installed = {}
    
    try:
        # 优先使用 importlib.metadata (Python 3.8+)
        from importlib import metadata
        
        for dist in metadata.distributions():
            name = dist.metadata['Name']
            version = dist.metadata['Version']
            installed[name.lower()] = version
    except ImportError:
        # 回退到 pkg_resources
        try:
            import pkg_resources
            
            for dist in pkg_resources.working_set:
                installed[dist.project_name.lower()] = dist.version
        except ImportError:
            print("⚠️ 警告：无法导入 importlib.metadata 或 pkg_resources")
            print("   请确保 Python 版本 >= 3.8 或安装 setuptools")
    
    return installed


def check_pytorch_cuda_compatibility() -> Tuple[bool, str]:
    """
    检查 PyTorch 和 CUDA 版本是否兼容
    
    Returns:
        Tuple[bool, str]: (是否兼容, 详细信息)
    """
    try:
        import torch
        
        pytorch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        
        if not cuda_available:
            return False, f"PyTorch {pytorch_version} - CUDA 不可用"
        
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        
        # 检查是否支持 bf16 (A100 优化)
        bf16_support = torch.cuda.is_bf16_supported()
        
        info = (
            f"PyTorch {pytorch_version}\n"
            f"  CUDA: {cuda_version}\n"
            f"  cuDNN: {cudnn_version}\n"
            f"  BF16 支持: {'是' if bf16_support else '否'}"
        )
        
        # 基本兼容性检查
        if cuda_version is None:
            return False, f"PyTorch {pytorch_version} - CUDA 版本未知"
        
        # 检查 CUDA 版本是否 >= 11.0 (推荐用于 A100)
        cuda_major = int(cuda_version.split('.')[0])
        if cuda_major < 11:
            return False, f"{info}\n  ⚠️ CUDA 版本过低，推荐 >= 11.8 用于 A100"
        
        return True, info
        
    except ImportError:
        return False, "PyTorch 未安装"
    except Exception as e:
        return False, f"检查 PyTorch/CUDA 时出错: {e}"


def parse_requirements(requirements_file: str = "requirements.txt") -> List[Tuple[str, str]]:
    """
    解析 requirements.txt 文件
    
    Args:
        requirements_file: requirements.txt 文件路径
        
    Returns:
        List[Tuple[str, str]]: [(包名, 版本要求), ...]
    """
    requirements = []
    
    try:
        with open(requirements_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # 跳过注释和空行
                if not line or line.startswith('#'):
                    continue
                
                # 解析包名和版本要求
                # 支持格式: package>=1.0.0, package==1.0.0, package
                match = re.match(r'^([a-zA-Z0-9_-]+)([><=!]+.*)?$', line)
                if match:
                    package_name = match.group(1).lower()
                    version_spec = match.group(2) if match.group(2) else ""
                    requirements.append((package_name, version_spec))
    
    except FileNotFoundError:
        print(f"⚠️ 警告：未找到 {requirements_file}")
    except Exception as e:
        print(f"⚠️ 警告：解析 {requirements_file} 时出错: {e}")
    
    return requirements


def list_missing_packages(requirements_file: str = "requirements.txt") -> List[Tuple[str, str]]:
    """
    列出缺失的包
    
    Args:
        requirements_file: requirements.txt 文件路径
        
    Returns:
        List[Tuple[str, str]]: [(包名, 版本要求), ...]
    """
    installed = check_installed_packages()
    required = parse_requirements(requirements_file)
    
    missing = []
    for package_name, version_spec in required:
        # 处理包名的不同形式 (例如 opencv-python vs cv2)
        package_variants = [package_name]
        
        # 特殊情况处理
        if package_name == 'opencv-python':
            package_variants.extend(['cv2', 'opencv'])
        elif package_name == 'pillow':
            package_variants.append('pil')
        elif package_name == 'pyyaml':
            package_variants.append('yaml')
        
        # 检查是否已安装
        is_installed = any(variant in installed for variant in package_variants)
        
        if not is_installed:
            missing.append((package_name, version_spec))
    
    return missing


def main():
    """主函数：执行完整的依赖检查"""
    print("=" * 60)
    print("依赖检查报告 (Dependency Check Report)")
    print("=" * 60)
    print()
    
    # 1. 检查已安装的包
    print("1. 已安装的包")
    print("-" * 60)
    installed = check_installed_packages()
    print(f"已安装包数量: {len(installed)}")
    print()
    
    # 2. 检查关键依赖
    print("2. 关键依赖检查")
    print("-" * 60)
    key_packages = [
        'torch',
        'torchvision',
        'numpy',
        'pillow',
        'pyyaml',
        'gradio',
        'flash-attn',
        'scikit-image',
        'opencv-python',
        'torchmetrics',
        'tqdm',
        'einops',
        'safetensors'
    ]
    
    for pkg in key_packages:
        # 处理包名变体
        pkg_lower = pkg.lower()
        variants = [pkg_lower]
        
        if pkg_lower == 'opencv-python':
            variants.extend(['cv2', 'opencv'])
        elif pkg_lower == 'pillow':
            variants.append('pil')
        elif pkg_lower == 'pyyaml':
            variants.append('yaml')
        
        # 查找已安装的版本
        version = None
        for variant in variants:
            if variant in installed:
                version = installed[variant]
                break
        
        if version:
            print(f"✓ {pkg}: {version}")
        else:
            print(f"✗ {pkg}: 未安装")
    
    print()
    
    # 3. 检查 PyTorch 和 CUDA 兼容性
    print("3. PyTorch 和 CUDA 兼容性")
    print("-" * 60)
    compatible, info = check_pytorch_cuda_compatibility()
    
    if compatible:
        print("✓ PyTorch 和 CUDA 版本兼容")
        print(info)
    else:
        print("✗ PyTorch 和 CUDA 版本不兼容或未安装")
        print(info)
    
    print()
    
    # 4. 列出缺失的包
    print("4. 缺失的包")
    print("-" * 60)
    missing = list_missing_packages()
    
    if missing:
        print(f"发现 {len(missing)} 个缺失的包：")
        for package_name, version_spec in missing:
            print(f"  - {package_name}{version_spec}")
        
        print()
        print("安装建议：")
        print("  pip install " + " ".join([f"{pkg}{ver}" for pkg, ver in missing]))
    else:
        print("✓ 所有必需的包都已安装")
    
    print()
    
    # 5. 特殊说明
    print("5. 特殊说明")
    print("-" * 60)
    
    # flash-attn 特殊说明
    if 'flash-attn' not in installed and 'flash_attn' not in installed:
        print("⚠️ flash-attn 未安装")
        print("   flash-attn 需要手动编译安装，可以显著提升 A100 性能 (2-3x)")
        print("   安装命令：")
        print("     pip install flash-attn --no-build-isolation")
        print("   注意：需要 CUDA >= 11.8 和 gcc >= 7.0")
        print()
    
    # hypothesis 说明
    if 'hypothesis' in installed:
        print("ℹ️ hypothesis 已安装")
        print("   根据简化测试策略，本项目不使用 hypothesis 属性测试")
        print("   可以选择卸载：pip uninstall hypothesis")
        print()
    
    print("=" * 60)
    print("依赖检查完成")
    print("=" * 60)
    
    # 返回退出码
    if missing:
        return 1  # 有缺失的包
    else:
        return 0  # 所有包都已安装


if __name__ == "__main__":
    sys.exit(main())
