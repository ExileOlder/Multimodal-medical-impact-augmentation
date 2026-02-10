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
    calculate_mse
)


class QualityEvaluator:
    """Quality evaluator for generated images."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = []
    
    def evaluate_pair(
        self,
        generated_path: str,
        reference_path: str
    ) -> dict:
        """
        Evaluate a single image pair.
        
        Args:
            generated_path: Path to generated image
            reference_path: Path to reference image
            
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
        
        # Calculate metrics
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
        
        return result
    
    def evaluate_batch(
        self,
        generated_dir: str,
        reference_dir: str,
        pattern: str = "*.png"
    ) -> list:
        """
        Evaluate all images in a directory.
        
        Args:
            generated_dir: Directory with generated images
            reference_dir: Directory with reference images
            pattern: File pattern to match
            
        Returns:
            List of results
        """
        generated_dir = Path(generated_dir)
        reference_dir = Path(reference_dir)
        
        # Find all generated images
        generated_files = sorted(generated_dir.glob(pattern))
        
        if not generated_files:
            print(f"No files found matching {pattern} in {generated_dir}")
            return []
        
        print(f"Found {len(generated_files)} generated images")
        
        results = []
        
        for gen_path in tqdm(generated_files, desc="Evaluating"):
            # Find corresponding reference image
            ref_path = reference_dir / gen_path.name
            
            if not ref_path.exists():
                print(f"Warning: Reference not found for {gen_path.name}, skipping")
                continue
            
            try:
                result = self.evaluate_pair(gen_path, ref_path)
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
            print(f"\nPSNR: {summary['psnr']['mean']:.2f} ± {summary['psnr']['std']:.2f} dB")
            print(f"  Range: [{summary['psnr']['min']:.2f}, {summary['psnr']['max']:.2f}]")
            print(f"\nSSIM: {summary['ssim']['mean']:.4f} ± {summary['ssim']['std']:.4f}")
            print(f"  Range: [{summary['ssim']['min']:.4f}, {summary['ssim']['max']:.4f}]")
            print(f"\nMAE: {summary['mae']['mean']:.2f} ± {summary['mae']['std']:.2f}")
            print(f"MSE: {summary['mse']['mean']:.2f} ± {summary['mse']['std']:.2f}")
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
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("OFFLINE QUALITY EVALUATION")
    print("="*70)
    print(f"Generated images: {args.generated}")
    print(f"Reference images: {args.reference}")
    print(f"Output file: {args.output}")
    print("="*70 + "\n")
    
    # Create evaluator
    evaluator = QualityEvaluator()
    
    # Evaluate batch
    results = evaluator.evaluate_batch(
        generated_dir=args.generated,
        reference_dir=args.reference,
        pattern=args.pattern
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
