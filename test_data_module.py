"""Test script for data processing module validation."""

import torch
from pathlib import Path
from src.data import (
    load_jsonl,
    preprocess_image,
    preprocess_mask,
    label_to_caption,
    MultimodalDataset,
    collate_fn,
    log_data_statistics
)

def test_label_to_caption():
    """Test label-to-caption conversion."""
    print("\n" + "="*50)
    print("Testing Label-to-Caption Conversion")
    print("="*50)
    
    for grade in range(5):
        caption = label_to_caption(grade)
        print(f"Grade {grade}: {caption}")
    
    # Test string input
    caption = label_to_caption("3")
    print(f"Grade '3' (string): {caption}")
    
    print("✓ Label-to-caption conversion working")


def test_preprocessing():
    """Test image and mask preprocessing."""
    print("\n" + "="*50)
    print("Testing Image/Mask Preprocessing")
    print("="*50)
    
    # Create dummy image
    from PIL import Image
    import numpy as np
    
    # Create test image (512x512 RGB)
    test_image = Image.fromarray(
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    )
    
    # Test image preprocessing
    image_tensor = preprocess_image(test_image, target_size=(1024, 1024))
    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"Image value range: [{image_tensor.min():.2f}, {image_tensor.max():.2f}]")
    assert image_tensor.shape == (3, 1024, 1024), "Image shape mismatch"
    assert -1.1 <= image_tensor.min() <= -0.9, "Image normalization issue (min)"
    assert 0.9 <= image_tensor.max() <= 1.1, "Image normalization issue (max)"
    print("✓ Image preprocessing working")
    
    # Create test mask (512x512 grayscale)
    test_mask = Image.fromarray(
        np.random.randint(0, 5, (512, 512), dtype=np.uint8)
    )
    
    # Test mask preprocessing
    mask_tensor = preprocess_mask(test_mask, target_size=(1024, 1024))
    print(f"Mask tensor shape: {mask_tensor.shape}")
    print(f"Mask value range: [{mask_tensor.min():.0f}, {mask_tensor.max():.0f}]")
    assert mask_tensor.shape == (1, 1024, 1024), "Mask shape mismatch"
    print("✓ Mask preprocessing working")


def test_dataset_with_dummy_data():
    """Test dataset with dummy data."""
    print("\n" + "="*50)
    print("Testing Dataset with Dummy Data")
    print("="*50)
    
    from PIL import Image
    import numpy as np
    import tempfile
    import os
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create dummy images and masks
        for i in range(3):
            # Create image
            img = Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )
            img.save(tmpdir / f"image_{i}.png")
            
            # Create mask
            mask = Image.fromarray(
                np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
            )
            mask.save(tmpdir / f"mask_{i}.png")
        
        # Create dummy data list
        data_list = [
            {
                'image_path': str(tmpdir / f"image_{i}.png"),
                'caption': str(i),  # Will be converted to DR grade text
                'mask_path': str(tmpdir / f"mask_{i}.png"),
                'text_only': False
            }
            for i in range(3)
        ]
        
        # Add one text-only entry
        data_list.append({
            'image_path': str(tmpdir / "image_0.png"),
            'caption': "Test caption without mask",
            'mask_path': None,
            'text_only': True
        })
        
        # Log statistics
        log_data_statistics(data_list, "Test Dataset")
        
        # Create dataset
        dataset = MultimodalDataset(
            data_list,
            image_size=(512, 512),
            normalize=True,
            auto_caption=True
        )
        
        print(f"Dataset length: {len(dataset)}")
        assert len(dataset) == 4, "Dataset length mismatch"
        
        # Test __getitem__
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Caption: {sample['caption']}")
        print(f"Mask shape: {sample['mask'].shape if sample['mask'] is not None else 'None'}")
        print(f"Text only: {sample['text_only']}")
        
        assert sample['image'].shape == (3, 512, 512), "Image shape mismatch"
        print("✓ Dataset __getitem__ working")
        
        # Test DataLoader with collate_fn
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_fn,
            shuffle=False
        )
        
        batch = next(iter(dataloader))
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Batch images shape: {batch['images'].shape}")
        print(f"Batch captions: {batch['captions']}")
        print(f"Batch masks shape: {batch['masks'].shape}")
        print(f"Batch text_only_flags: {batch['text_only_flags']}")
        
        assert batch['images'].shape == (2, 3, 512, 512), "Batch images shape mismatch"
        assert batch['masks'].shape == (2, 1, 512, 512), "Batch masks shape mismatch"
        assert len(batch['captions']) == 2, "Batch captions length mismatch"
        print("✓ DataLoader with collate_fn working")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("DATA PROCESSING MODULE VALIDATION")
    print("="*70)
    
    try:
        test_label_to_caption()
        test_preprocessing()
        test_dataset_with_dummy_data()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED - Data processing module is working correctly!")
        print("="*70)
        print("\nNext steps:")
        print("1. Prepare your actual dataset (FGADR or IDRiD)")
        print("2. Create JSONL files for your data")
        print("3. Proceed to Task 4: Model extension")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
