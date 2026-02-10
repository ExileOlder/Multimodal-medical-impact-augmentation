#!/usr/bin/env python3
"""
Smoke Test Script for Medical Image Augmentation System

This script performs quick validation of core functionality:
1. Data loading
2. Model initialization
3. Forward pass

Usage:
    python scripts/smoke_test.py

Exit codes:
    0: All tests passed
    1: One or more tests failed
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_data_loading():
    """测试数据加载功能"""
    print("测试数据加载...")
    try:
        from src.data.jsonl_loader import load_jsonl
        
        # 使用示例数据
        example_file = project_root / 'codes' / 'example_data' / 'example_ver3.json'
        
        if not example_file.exists():
            print(f"✗ 示例数据文件不存在: {example_file}")
            return False
        
        data = load_jsonl(example_file)
        
        if len(data) == 0:
            print("✗ 数据加载失败：数据为空")
            return False
        
        print(f"  成功加载 {len(data)} 条数据")
        
        # 验证数据格式
        first_entry = data[0]
        required_fields = ['image_path', 'caption']
        for field in required_fields:
            if field not in first_entry:
                print(f"✗ 数据格式错误：缺少字段 '{field}'")
                return False
        
        print("✓ 数据加载测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_initialization():
    """测试模型初始化"""
    print("\n测试模型初始化...")
    try:
        from src.models.nexdit_mask import NextDiTWithMask
        import torch
        
        # 使用小型配置进行快速测试
        model = NextDiTWithMask(
            patch_size=2,
            in_channels=4,  # VAE latent channels
            mask_channels=1,
            dim=512,  # 小型配置
            n_layers=4,
            n_heads=8,
            cap_feat_dim=512  # 匹配 dim
        )
        
        # 检查模型参数
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  模型参数量: {param_count:,}")
        
        # 检查模型是否在正确的设备上
        device = next(model.parameters()).device
        print(f"  模型设备: {device}")
        
        print("✓ 模型初始化测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 模型初始化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """测试前向传播"""
    print("\n测试前向传播...")
    try:
        from src.models.nexdit_mask import NextDiTWithMask
        import torch
        
        # 使用小型配置
        model = NextDiTWithMask(
            patch_size=2,
            in_channels=4,
            mask_channels=1,
            dim=512,
            n_layers=4,
            n_heads=8,
            cap_feat_dim=512
        )
        
        # 设置为评估模式
        model.eval()
        
        # 创建随机输入（小尺寸以加快测试）
        batch_size = 1
        height, width = 64, 64  # 小尺寸用于快速测试
        
        x = torch.randn(batch_size, 4, height, width)
        t = torch.tensor([0.5])
        cap_feats = torch.randn(batch_size, 77, 512)
        cap_mask = torch.ones(batch_size, 77)
        condition_mask = torch.randn(batch_size, 1, height, width)
        
        print(f"  输入形状: x={x.shape}, t={t.shape}, cap_feats={cap_feats.shape}")
        
        # 前向传播（不计算梯度）
        with torch.no_grad():
            output = model(x, t, cap_feats, cap_mask, condition_mask)
        
        print(f"  输出形状: {output.shape}")
        
        # 验证输出形状
        expected_channels = 4 * 2 if model.learn_sigma else 4
        expected_shape = (batch_size, expected_channels, height, width)
        
        if output.shape != expected_shape:
            print(f"✗ 输出形状不匹配: 期望 {expected_shape}, 实际 {output.shape}")
            return False
        
        # 检查输出是否包含 NaN 或 Inf
        if torch.isnan(output).any():
            print("✗ 输出包含 NaN 值")
            return False
        
        if torch.isinf(output).any():
            print("✗ 输出包含 Inf 值")
            return False
        
        print("✓ 前向传播测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数：运行所有冒烟测试"""
    print("=" * 60)
    print("冒烟测试开始")
    print("=" * 60)
    print()
    
    tests = [
        ("数据加载", test_data_loading),
        ("模型初始化", test_model_initialization),
        ("前向传播", test_forward_pass)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print()
    print("=" * 60)
    print("冒烟测试结果")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    print()
    print(f"总计: {passed}/{total} 测试通过")
    print()
    
    if passed == total:
        print("✓ 所有测试通过，可以安全删除 codes_backup/")
        return 0
    else:
        print("✗ 部分测试失败，请检查依赖问题")
        print("提示: 运行 'python scripts/rollback.py' 恢复 codes/ 目录")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
