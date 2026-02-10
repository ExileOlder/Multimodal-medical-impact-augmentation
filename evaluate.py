"""
Offline quality evaluation script.

This script evaluates generated images against reference images using
quality metrics (PSNR, SSIM, MAE, MSE).
"""

import argparse
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

from src.evaluation.metrics import (
    calculate_psnr,
    calculate_ssim_skimage,
    calculate_mae,
    calculate_mse,
    calculate_dice_coefficient,
    calculate_iou,
    extract_lesion_mask_from_image
)


class QualityEvaluator:
    """Quality evaluator for generated images."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = []
    
    def evaluate_pair(
        self,
        generated_path: str,
        reference_path: str,
        input_mask_path: str = None,
        evaluate_structure: bool = False
    ) -> dict:
        """
        Evaluate a single image pair.
        
        Args:
            generated_path: Path to generated image
            reference_path: Path to reference image
            input_mask_path: Path to input condition mask (optional, for structure consistency)
            evaluate_structure: Whether to evaluate structure consistency
            
        Returns:
            Dict with metrics
        """
        # Load images
        generated = Image.open(generated_path).convert('RGB')
        reference = Image.open(reference_path).convert('RGB')
        
        # Resize if needed
        if generated.size != reference.size:
            print(f"Warning: Resizing generated image from {generated.size} to {reference.size}")
            generated = generated.resize(reference.size, Image.LANCZOS)
        
        # Convert to numpy
        gen_array = np.array(generated)
        ref_array = np.array(reference)
        
        # Calculate quality metrics
        psnr = calculate_psnr(gen_array, ref_array, max_value=255.0)
        ssim = calculate_ssim_skimage(gen_array, ref_array, max_value=255.0)
        mae = calculate_mae(gen_array, ref_array)
        mse = calculate_mse(gen_array, ref_array)
        
        result = {
            'generated_path': str(generated_path),
            'reference_path': str(reference_path),
            'psnr': psnr,
            'ssim': ssim,
            'mae': mae,
            'mse': mse
        }
        
        # Calculate structure consistency metrics if requested
        if evaluate_structure and input_mask_path:
            try:
                # Load input mask
                input_mask = Image.open(input_mask_path).convert('L')
                if input_mask.size != generated.size:
                    input_mask = input_mask.resize(generated.size, Image.NEAREST)
                input_mask_array = np.array(input_mask)
                
                # Extract lesion mask from generated image
                generated_mask = extract_lesion_mask_from_image(
                    gen_array,
                    method="red_channel",
                    threshold=0.5
                )
                
                # Convert to same scale
                generated_mask = (generated_mask * 255).astype(np.uint8)
                
                # Calculate structure consistency metrics
                dice = calculate_dice_coefficient(generated_mask, input_mask_array)
                iou = calculate_iou(generated_mask, input_mask_array)
                
                result['dice_coefficient'] = dice
                result['iou'] = iou
                result['input_mask_path'] = str(input_mask_path)
                
            except Exception as e:
                print(f"Warning: Failed to calculate structure metrics: {e}")
                result['dice_coefficient'] = None
                result['iou'] = None
        
        return result
    
    def evaluate_batch(
        self,
        generated_dir: str,
        reference_dir: str,
        mask_dir: str = None,
        pattern: str = "*.png",
        evaluate_structure: bool = False
    ) -> list:
        """
        Evaluate all images in a directory.
        
        Args:
            generated_dir: Directory with generated images
            reference_dir: Directory with reference images
            mask_dir: Directory with input condition masks (optional, for structure consistency)
            pattern: File pattern to match
            evaluate_structure: Whether to evaluate structure consistency
            
        Returns:
            List of results
        """
        generated_dir = Path(generated_dir)
        reference_dir = Path(reference_dir)
        
        if mask_dir:
            mask_dir = Path(mask_dir)
        
        # Find all generated images
        generated_files = sorted(generated_dir.glob(pattern))
        
        if not generated_files:
            print(f"No files found matching {pattern} in {generated_dir}")
            return []
        
        print(f"Found {len(generated_files)} generated images")
        
        if evaluate_structure and mask_dir:
            print(f"Structure consistency evaluation enabled (using masks from {mask_dir})")
        
        results = []
        
        for gen_path in tqdm(generated_files, desc="Evaluating"):
            # Find corresponding reference image
            ref_path = reference_dir / gen_path.name
            
            if not ref_path.exists():
                print(f"Warning: Reference not found for {gen_path.name}, skipping")
                continue
            
            # Find corresponding mask if structure evaluation is enabled
            mask_path = None
            if evaluate_structure and mask_dir:
                mask_path = mask_dir / gen_path.name
                if not mask_path.exists():
                    print(f"Warning: Mask not found for {gen_path.name}, skipping structure metrics")
                    mask_path = None
            
            try:
                result = self.evaluate_pair(
                    gen_path,
                    ref_path,
                    input_mask_path=mask_path,
                    evaluate_structure=evaluate_structure
                )
                results.append(result)
            except Exception as e:
                print(f"Error evaluating {gen_path.name}: {e}")
                continue
        
        self.results = results
        return results
    
    def save_results(self, output_path: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate summary statistics
        if self.results:
            psnr_values = [r['psnr'] for r in self.results if r['psnr'] != float('inf')]
            ssim_values = [r['ssim'] for r in self.results]
            mae_values = [r['mae'] for r in self.results]
            mse_values = [r['mse'] for r in self.results]
            
            summary = {
                'num_images': len(self.results),
                'psnr': {
                    'mean': np.mean(psnr_values) if psnr_values else 0,
                    'std': np.std(psnr_values) if psnr_values else 0,
                    'min': np.min(psnr_values) if psnr_values else 0,
                    'max': np.max(psnr_values) if psnr_values else 0
                },
                'ssim': {
                    'mean': np.mean(ssim_values),
                    'std': np.std(ssim_values),
                    'min': np.min(ssim_values),
                    'max': np.max(ssim_values)
                },
                'mae': {
                    'mean': np.mean(mae_values),
                    'std': np.std(mae_values),
                    'min': np.min(mae_values),
                    'max': np.max(mae_values)
                },
                'mse': {
                    'mean': np.mean(mse_values),
                    'std': np.std(mse_values),
                    'min': np.min(mse_values),
                    'max': np.max(mse_values)
                }
            }
            
            # Add structure consistency metrics if available
            dice_values = [r['dice_coefficient'] for r in self.results if r.get('dice_coefficient') is not None]
            iou_values = [r['iou'] for r in self.results if r.get('iou') is not None]
            
            if dice_values:
                summary['dice_coefficient'] = {
                    'mean': np.mean(dice_values),
                    'std': np.std(dice_values),
                    'min': np.min(dice_values),
                    'max': np.max(dice_values)
                }
            
            if iou_values:
                summary['iou'] = {
                    'mean': np.mean(iou_values),
                    'std': np.std(iou_values),
                    'min': np.min(iou_values),
                    'max': np.max(iou_values)
                }
        else:
            summary = {'num_images': 0}
        
        # Save to file
        output_data = {
            'summary': summary,
            'results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
        # Print summary
        if self.results:
            print("\n" + "="*70)
            print("EVALUATION SUMMARY")
            print("="*70)
            print(f"Number of images: {summary['num_images']}")
            print(f"\n【图像质量指标】")
            print(f"PSNR: {summary['psnr']['mean']:.2f} ± {summary['psnr']['std']:.2f} dB")
            print(f"  Range: [{summary['psnr']['min']:.2f}, {summary['psnr']['max']:.2f}]")
            print(f"\nSSIM: {summary['ssim']['mean']:.4f} ± {summary['ssim']['std']:.4f}")
            print(f"  Range: [{summary['ssim']['min']:.4f}, {summary['ssim']['max']:.4f}]")
            print(f"\nMAE: {summary['mae']['mean']:.2f} ± {summary['mae']['std']:.2f}")
            print(f"MSE: {summary['mse']['mean']:.2f} ± {summary['mse']['std']:.2f}")
            
            # Print structure consistency metrics if available
            if 'dice_coefficient' in summary:
                print(f"\n【结构一致性指标】")
                print(f"Dice Coefficient: {summary['dice_coefficient']['mean']:.4f} ± {summary['dice_coefficient']['std']:.4f}")
                print(f"  Range: [{summary['dice_coefficient']['min']:.4f}, {summary['dice_coefficient']['max']:.4f}]")
                print(f"  (1.0 = 完美匹配, >0.7 = 良好, >0.5 = 可接受)")
            
            if 'iou' in summary:
                print(f"\nIoU (Jaccard Index): {summary['iou']['mean']:.4f} ± {summary['iou']['std']:.4f}")
                print(f"  Range: [{summary['iou']['min']:.4f}, {summary['iou']['max']:.4f}]")
                print(f"  (1.0 = 完美匹配, >0.5 = 良好, >0.3 = 可接受)")
            
            print("="*70)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate generated image quality")
    parser.add_argument(
        "--generated",
        type=str,
        required=True,
        help="Directory containing generated images"
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Directory containing reference images"
    )
    parser.add_argument(
        "--masks",
        type=str,
        default=None,
        help="Directory containing input condition masks (for structure consistency evaluation)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_results.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="File pattern to match (default: *.png)"
    )
    parser.add_argument(
        "--evaluate-structure",
        action="store_true",
        help="Enable structure consistency evaluation (requires --masks)"
    )
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("OFFLINE QUALITY EVALUATION")
    print("="*70)
    print(f"Generated images: {args.generated}")
    print(f"Reference images: {args.reference}")
    if args.masks:
        print(f"Input masks: {args.masks}")
    if args.evaluate_structure:
        print(f"Structure consistency: ENABLED")
    print(f"Output file: {args.output}")
    print("="*70 + "\n")
    
    # Create evaluator
    evaluator = QualityEvaluator()
    
    # Evaluate batch
    results = evaluator.evaluate_batch(
        generated_dir=args.generated,
        reference_dir=args.reference,
        mask_dir=args.masks,
        pattern=args.pattern,
        evaluate_structure=args.evaluate_structure
    )
    
    if not results:
        print("\n❌ No images were evaluated. Please check your input directories.")
        return 1
    
    # Save results
    evaluator.save_results(args.output)
    
    print("\n✓ Evaluation completed successfully!")
    
    return 0


if __name__ == "__main__":
    exit(main())
