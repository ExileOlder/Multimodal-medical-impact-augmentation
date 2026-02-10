"""
演示评估功能的简单测试脚本。

此脚本创建合成数据来演示评估指标的计算。
"""

import numpy as np
from PIL import Image
import os

# 确保可以导入评估模块
try:
    from src.evaluation.metrics import (
        calculate_psnr,
        calculate_ssim_skimage,
        calculate_mae,
        calculate_mse,
        calculate_dice_coefficient,
        calculate_iou,
        extract_lesion_mask_from_image
    )
    print("✓ 成功导入评估模块")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print("请确保在正确的环境中运行此脚本")
    exit(1)


def create_synthetic_image(size=256, lesion_center=(128, 128), lesion_radius=30):
    """创建合成医学图像（带红色病灶）"""
    # 创建背景（灰色）
    image = np.ones((size, size, 3), dtype=np.uint8) * 150
    
    # 添加红色病灶
    y, x = np.ogrid[:size, :size]
    mask_circle = (x - lesion_center[0])**2 + (y - lesion_center[1])**2 <= lesion_radius**2
    
    image[mask_circle, 0] = 200  # 红色通道
    image[mask_circle, 1] = 100  # 绿色通道
    image[mask_circle, 2] = 100  # 蓝色通道
    
    return image


def create_synthetic_mask(size=256, lesion_center=(128, 128), lesion_radius=30):
    """创建合成掩码"""
    mask = np.zeros((size, size), dtype=np.uint8)
    
    y, x = np.ogrid[:size, :size]
    mask_circle = (x - lesion_center[0])**2 + (y - lesion_center[1])**2 <= lesion_radius**2
    
    mask[mask_circle] = 255
    
    return mask


def test_image_quality_metrics():
    """测试图像质量指标"""
    print("\n" + "="*70)
    print("测试 1: 图像质量指标（PSNR, SSIM, MAE, MSE）")
    print("="*70)
    
    # 创建参考图像
    ref_image = create_synthetic_image(size=256, lesion_center=(128, 128), lesion_radius=30)
    
    # 测试场景 1: 完全相同的图像
    print("\n场景 1: 完全相同的图像")
    gen_image1 = ref_image.copy()
    
    psnr1 = calculate_psnr(gen_image1, ref_image)
    ssim1 = calculate_ssim_skimage(gen_image1, ref_image)
    mae1 = calculate_mae(gen_image1, ref_image)
    mse1 = calculate_mse(gen_image1, ref_image)
    
    print(f"  PSNR: {psnr1:.2f} dB (期望: inf)")
    print(f"  SSIM: {ssim1:.4f} (期望: 1.0)")
    print(f"  MAE:  {mae1:.2f} (期望: 0.0)")
    print(f"  MSE:  {mse1:.2f} (期望: 0.0)")
    
    # 测试场景 2: 轻微噪声
    print("\n场景 2: 添加轻微噪声")
    gen_image2 = ref_image.copy()
    noise = np.random.normal(0, 5, gen_image2.shape).astype(np.int16)
    gen_image2 = np.clip(gen_image2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    psnr2 = calculate_psnr(gen_image2, ref_image)
    ssim2 = calculate_ssim_skimage(gen_image2, ref_image)
    mae2 = calculate_mae(gen_image2, ref_image)
    mse2 = calculate_mse(gen_image2, ref_image)
    
    print(f"  PSNR: {psnr2:.2f} dB (期望: 30-40 dB)")
    print(f"  SSIM: {ssim2:.4f} (期望: 0.9-1.0)")
    print(f"  MAE:  {mae2:.2f} (期望: 3-8)")
    print(f"  MSE:  {mse2:.2f} (期望: 10-100)")
    
    # 测试场景 3: 较大噪声
    print("\n场景 3: 添加较大噪声")
    gen_image3 = ref_image.copy()
    noise = np.random.normal(0, 20, gen_image3.shape).astype(np.int16)
    gen_image3 = np.clip(gen_image3.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    psnr3 = calculate_psnr(gen_image3, ref_image)
    ssim3 = calculate_ssim_skimage(gen_image3, ref_image)
    mae3 = calculate_mae(gen_image3, ref_image)
    mse3 = calculate_mse(gen_image3, ref_image)
    
    print(f"  PSNR: {psnr3:.2f} dB (期望: 20-30 dB)")
    print(f"  SSIM: {ssim3:.4f} (期望: 0.7-0.9)")
    print(f"  MAE:  {mae3:.2f} (期望: 10-20)")
    print(f"  MSE:  {mse3:.2f} (期望: 200-600)")
    
    print("\n✓ 图像质量指标测试完成")


def test_structure_consistency_metrics():
    """测试结构一致性指标"""
    print("\n" + "="*70)
    print("测试 2: 结构一致性指标（Dice, IoU）")
    print("="*70)
    
    # 创建参考掩码
    ref_mask = create_synthetic_mask(size=256, lesion_center=(128, 128), lesion_radius=30)
    
    # 测试场景 1: 完全匹配
    print("\n场景 1: 完全匹配的掩码")
    gen_mask1 = ref_mask.copy()
    
    dice1 = calculate_dice_coefficient(gen_mask1, ref_mask)
    iou1 = calculate_iou(gen_mask1, ref_mask)
    
    print(f"  Dice: {dice1:.4f} (期望: 1.0)")
    print(f"  IoU:  {iou1:.4f} (期望: 1.0)")
    
    # 测试场景 2: 部分重叠（位置偏移）
    print("\n场景 2: 位置轻微偏移")
    gen_mask2 = create_synthetic_mask(size=256, lesion_center=(138, 138), lesion_radius=30)
    
    dice2 = calculate_dice_coefficient(gen_mask2, ref_mask)
    iou2 = calculate_iou(gen_mask2, ref_mask)
    
    print(f"  Dice: {dice2:.4f} (期望: 0.6-0.8)")
    print(f"  IoU:  {iou2:.4f} (期望: 0.4-0.7)")
    
    # 测试场景 3: 大小不同
    print("\n场景 3: 大小不同（半径 40 vs 30）")
    gen_mask3 = create_synthetic_mask(size=256, lesion_center=(128, 128), lesion_radius=40)
    
    dice3 = calculate_dice_coefficient(gen_mask3, ref_mask)
    iou3 = calculate_iou(gen_mask3, ref_mask)
    
    print(f"  Dice: {dice3:.4f} (期望: 0.7-0.9)")
    print(f"  IoU:  {iou3:.4f} (期望: 0.5-0.8)")
    
    # 测试场景 4: 无重叠
    print("\n场景 4: 完全不重叠")
    gen_mask4 = create_synthetic_mask(size=256, lesion_center=(64, 64), lesion_radius=20)
    
    dice4 = calculate_dice_coefficient(gen_mask4, ref_mask)
    iou4 = calculate_iou(gen_mask4, ref_mask)
    
    print(f"  Dice: {dice4:.4f} (期望: 0.0)")
    print(f"  IoU:  {iou4:.4f} (期望: 0.0)")
    
    print("\n✓ 结构一致性指标测试完成")


def test_lesion_extraction():
    """测试病灶提取功能"""
    print("\n" + "="*70)
    print("测试 3: 病灶掩码提取")
    print("="*70)
    
    # 创建带红色病灶的图像
    image = create_synthetic_image(size=256, lesion_center=(128, 128), lesion_radius=30)
    ref_mask = create_synthetic_mask(size=256, lesion_center=(128, 128), lesion_radius=30)
    
    # 测试红色通道法
    print("\n方法 1: 红色通道法")
    extracted_mask1 = extract_lesion_mask_from_image(image, method="red_channel", threshold=0.5)
    extracted_mask1_uint8 = (extracted_mask1 * 255).astype(np.uint8)
    
    dice1 = calculate_dice_coefficient(extracted_mask1_uint8, ref_mask)
    iou1 = calculate_iou(extracted_mask1_uint8, ref_mask)
    
    print(f"  提取的掩码与参考掩码对比:")
    print(f"  Dice: {dice1:.4f}")
    print(f"  IoU:  {iou1:.4f}")
    
    # 测试亮度法
    print("\n方法 2: 亮度法")
    extracted_mask2 = extract_lesion_mask_from_image(image, method="brightness", threshold=0.3)
    extracted_mask2_uint8 = (extracted_mask2 * 255).astype(np.uint8)
    
    dice2 = calculate_dice_coefficient(extracted_mask2_uint8, ref_mask)
    iou2 = calculate_iou(extracted_mask2_uint8, ref_mask)
    
    print(f"  提取的掩码与参考掩码对比:")
    print(f"  Dice: {dice2:.4f}")
    print(f"  IoU:  {iou2:.4f}")
    
    print("\n✓ 病灶掩码提取测试完成")


def save_demo_images():
    """保存演示图像（可选）"""
    print("\n" + "="*70)
    print("保存演示图像")
    print("="*70)
    
    # 创建输出目录
    output_dir = "results/evaluation_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建并保存图像
    ref_image = create_synthetic_image(size=256, lesion_center=(128, 128), lesion_radius=30)
    ref_mask = create_synthetic_mask(size=256, lesion_center=(128, 128), lesion_radius=30)
    
    Image.fromarray(ref_image).save(f"{output_dir}/reference_image.png")
    Image.fromarray(ref_mask).save(f"{output_dir}/reference_mask.png")
    
    # 创建不同质量的生成图像
    gen_image_good = ref_image.copy()
    noise = np.random.normal(0, 5, gen_image_good.shape).astype(np.int16)
    gen_image_good = np.clip(gen_image_good.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(gen_image_good).save(f"{output_dir}/generated_good.png")
    
    gen_image_fair = ref_image.copy()
    noise = np.random.normal(0, 20, gen_image_fair.shape).astype(np.int16)
    gen_image_fair = np.clip(gen_image_fair.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(gen_image_fair).save(f"{output_dir}/generated_fair.png")
    
    # 创建不同位置的掩码
    gen_mask_offset = create_synthetic_mask(size=256, lesion_center=(138, 138), lesion_radius=30)
    Image.fromarray(gen_mask_offset).save(f"{output_dir}/generated_mask_offset.png")
    
    print(f"\n✓ 演示图像已保存到: {output_dir}/")
    print(f"  - reference_image.png: 参考图像")
    print(f"  - reference_mask.png: 参考掩码")
    print(f"  - generated_good.png: 高质量生成图像（轻微噪声）")
    print(f"  - generated_fair.png: 中等质量生成图像（较大噪声）")
    print(f"  - generated_mask_offset.png: 位置偏移的掩码")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("医学影像评估指标演示")
    print("="*70)
    print("\n本脚本演示评估指标的计算和使用方法。")
    print("使用合成数据来展示不同场景下的指标表现。")
    
    try:
        # 运行测试
        test_image_quality_metrics()
        test_structure_consistency_metrics()
        test_lesion_extraction()
        save_demo_images()
        
        print("\n" + "="*70)
        print("✅ 所有测试完成！")
        print("="*70)
        print("\n详细的评估指标说明请参考: docs/EVALUATION_GUIDE.md")
        print("演示图像已保存到: results/evaluation_demo/")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
