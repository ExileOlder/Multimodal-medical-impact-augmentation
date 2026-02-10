"""Manual test for noise processing functionality."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

print("Testing noise processing implementation...")
print("=" * 60)

# Test 1: Check if cv2 import works in preprocessing.py
print("\n1. Testing cv2 import...")
try:
    import cv2
    print(f"   ✓ cv2 imported successfully (version: {cv2.__version__})")
except ImportError as e:
    print(f"   ✗ cv2 import failed: {e}")
    print("   Note: opencv-python needs to be installed")
    sys.exit(1)

# Test 2: Import preprocessing functions
print("\n2. Testing preprocessing imports...")
try:
    from src.data.preprocessing import (
        apply_gaussian_filter,
        apply_median_filter,
        preprocess_image
    )
    print("   ✓ All functions imported successfully")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 3: Check function signatures
print("\n3. Checking function signatures...")
import inspect

# Check apply_gaussian_filter
sig = inspect.signature(apply_gaussian_filter)
params = list(sig.parameters.keys())
expected = ['image', 'kernel_size', 'sigma']
if params == expected:
    print(f"   ✓ apply_gaussian_filter signature correct: {params}")
else:
    print(f"   ✗ apply_gaussian_filter signature incorrect")
    print(f"     Expected: {expected}")
    print(f"     Got: {params}")

# Check apply_median_filter
sig = inspect.signature(apply_median_filter)
params = list(sig.parameters.keys())
expected = ['image', 'kernel_size']
if params == expected:
    print(f"   ✓ apply_median_filter signature correct: {params}")
else:
    print(f"   ✗ apply_median_filter signature incorrect")
    print(f"     Expected: {expected}")
    print(f"     Got: {params}")

# Check preprocess_image has new parameters
sig = inspect.signature(preprocess_image)
params = list(sig.parameters.keys())
required_params = ['enable_denoising', 'denoising_method', 'gaussian_kernel_size', 
                   'gaussian_sigma', 'median_kernel_size']
has_all = all(p in params for p in required_params)
if has_all:
    print(f"   ✓ preprocess_image has all denoising parameters")
else:
    print(f"   ✗ preprocess_image missing some parameters")
    missing = [p for p in required_params if p not in params]
    print(f"     Missing: {missing}")

# Test 4: Test with actual data (if numpy and PIL are available)
print("\n4. Testing with actual data...")
try:
    import numpy as np
    from PIL import Image
    
    # Create test image
    test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Test Gaussian filter
    result_gaussian = apply_gaussian_filter(test_array, kernel_size=5, sigma=1.0)
    assert result_gaussian.shape == test_array.shape, "Gaussian filter shape mismatch"
    print("   ✓ apply_gaussian_filter works correctly")
    
    # Test median filter
    result_median = apply_median_filter(test_array, kernel_size=5)
    assert result_median.shape == test_array.shape, "Median filter shape mismatch"
    print("   ✓ apply_median_filter works correctly")
    
    # Test preprocess_image with denoising
    test_image = Image.new('RGB', (256, 256), color=(128, 128, 128))
    result = preprocess_image(
        test_image,
        target_size=(128, 128),
        normalize=True,
        enable_denoising=True,
        denoising_method="gaussian"
    )
    print(f"   ✓ preprocess_image with denoising works (output shape: {result.shape})")
    
    # Test preprocess_image without denoising
    result_no_denoise = preprocess_image(
        test_image,
        target_size=(128, 128),
        normalize=True,
        enable_denoising=False
    )
    print(f"   ✓ preprocess_image without denoising works (output shape: {result_no_denoise.shape})")
    
except ImportError as e:
    print(f"   ⚠ Skipping data tests (missing dependency: {e})")
except Exception as e:
    print(f"   ✗ Data test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Check config file
print("\n5. Checking train_config.yaml...")
try:
    import yaml
    with open('configs/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config.get('data', {})
    required_keys = ['enable_denoising', 'denoising_method', 'gaussian_kernel_size', 
                     'gaussian_sigma', 'median_kernel_size']
    
    has_all_keys = all(key in data_config for key in required_keys)
    if has_all_keys:
        print("   ✓ train_config.yaml has all denoising configuration keys")
        print(f"     enable_denoising: {data_config['enable_denoising']}")
        print(f"     denoising_method: {data_config['denoising_method']}")
    else:
        missing = [key for key in required_keys if key not in data_config]
        print(f"   ✗ train_config.yaml missing keys: {missing}")
        
except Exception as e:
    print(f"   ✗ Config check failed: {e}")

print("\n" + "=" * 60)
print("Testing complete!")
print("=" * 60)
