# Image Denoising Guide

## Overview

The medical image augmentation system includes optional image denoising functionality to handle noisy input images. However, **denoising is disabled by default** because medical image datasets are typically pre-processed and cleaned before distribution.

## Why Denoising is Disabled by Default

Medical imaging datasets (such as diabetic retinopathy datasets) undergo rigorous quality control and preprocessing before being released:

1. **Professional Acquisition**: Medical images are captured using calibrated, professional equipment with minimal noise
2. **Quality Control**: Images undergo quality checks to ensure diagnostic value
3. **Standard Preprocessing**: Datasets typically include standardized preprocessing (normalization, artifact removal)
4. **Diagnostic Requirements**: Medical images must maintain high fidelity for accurate diagnosis

Therefore, applying additional denoising may:
- Introduce unnecessary computational overhead
- Potentially remove fine details important for diagnosis
- Alter the original image characteristics in unintended ways

## When to Enable Denoising

Consider enabling denoising if:

1. **Raw Images**: You're working with raw, unprocessed medical images
2. **Acquisition Noise**: Images contain visible sensor noise or compression artifacts
3. **Low-Quality Sources**: Images from older equipment or non-standard sources
4. **Specific Requirements**: Your research specifically requires denoising as part of the pipeline

## Available Denoising Methods

### 1. Gaussian Filter

**Best for**: Reducing Gaussian (random) noise while preserving overall structure

**Characteristics**:
- Smooths images using a Gaussian kernel
- Effective for random noise reduction
- May blur edges slightly

**Configuration**:
```yaml
data:
  enable_denoising: true
  denoising_method: "gaussian"
  gaussian_kernel_size: 5      # Kernel size (must be odd)
  gaussian_sigma: 1.0           # Standard deviation
```

**Recommended Settings**:
- Small noise: `kernel_size=3, sigma=0.5`
- Medium noise: `kernel_size=5, sigma=1.0` (default)
- Heavy noise: `kernel_size=7, sigma=1.5`

### 2. Median Filter

**Best for**: Removing salt-and-pepper (impulse) noise while preserving edges

**Characteristics**:
- Replaces each pixel with the median of neighboring pixels
- Excellent edge preservation
- Effective for impulse noise

**Configuration**:
```yaml
data:
  enable_denoising: true
  denoising_method: "median"
  median_kernel_size: 5         # Kernel size (must be odd)
```

**Recommended Settings**:
- Light noise: `kernel_size=3`
- Medium noise: `kernel_size=5` (default)
- Heavy noise: `kernel_size=7`

## Usage Examples

### Example 1: Enable Gaussian Denoising

Edit `configs/train_config.yaml`:

```yaml
data:
  enable_denoising: true
  denoising_method: "gaussian"
  gaussian_kernel_size: 5
  gaussian_sigma: 1.0
```

### Example 2: Enable Median Denoising

Edit `configs/train_config.yaml`:

```yaml
data:
  enable_denoising: true
  denoising_method: "median"
  median_kernel_size: 5
```

### Example 3: Disable Denoising (Default)

```yaml
data:
  enable_denoising: false
```

## Programmatic Usage

You can also use the denoising functions directly in your code:

```python
from src.data.preprocessing import apply_gaussian_filter, apply_median_filter
import numpy as np

# Load your image as numpy array
image = np.array(...)  # Shape: (H, W, C)

# Apply Gaussian filter
denoised_gaussian = apply_gaussian_filter(image, kernel_size=5, sigma=1.0)

# Apply median filter
denoised_median = apply_median_filter(image, kernel_size=5)
```

Or use the integrated preprocessing function:

```python
from src.data.preprocessing import preprocess_image
from PIL import Image

# Load image
image = Image.open('path/to/image.jpg')

# Preprocess with denoising
tensor = preprocess_image(
    image,
    target_size=(1024, 1024),
    normalize=True,
    enable_denoising=True,
    denoising_method="gaussian",
    gaussian_kernel_size=5,
    gaussian_sigma=1.0
)
```

## Performance Considerations

Enabling denoising adds computational overhead:

- **Gaussian Filter**: ~5-10ms per 1024x1024 image
- **Median Filter**: ~10-20ms per 1024x1024 image

For training with large datasets, this can add significant time:
- 10,000 images × 10ms = ~100 seconds additional preprocessing time

Consider the trade-off between image quality improvement and training time.

## Validation

To verify denoising is working correctly:

1. **Visual Inspection**: Compare original and denoised images
2. **Metrics**: Measure PSNR (Peak Signal-to-Noise Ratio) improvement
3. **Training Results**: Monitor if denoising improves model performance

## Troubleshooting

### Issue: Denoising makes images too blurry

**Solution**: Reduce kernel size or sigma:
```yaml
gaussian_kernel_size: 3  # Instead of 5
gaussian_sigma: 0.5      # Instead of 1.0
```

### Issue: Denoising doesn't seem to work

**Solution**: 
1. Verify `enable_denoising: true` in config
2. Check that opencv-python is installed: `pip install opencv-python`
3. Verify images actually contain noise (medical datasets often don't)

### Issue: Training is too slow with denoising

**Solution**:
1. Disable denoising if images are already clean
2. Reduce `num_workers` to balance CPU usage
3. Consider preprocessing images offline and saving denoised versions

## References

- OpenCV Gaussian Blur: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
- OpenCV Median Blur: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
- Medical Image Preprocessing: Standard practices in medical imaging

## Summary

- **Default**: Denoising is **disabled** (medical datasets are pre-processed)
- **Enable when**: Working with raw or noisy images
- **Methods**: Gaussian (random noise) or Median (impulse noise)
- **Configuration**: Edit `configs/train_config.yaml`
- **Performance**: Adds 5-20ms per image processing time
