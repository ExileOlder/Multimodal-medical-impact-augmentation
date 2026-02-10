# Setup Instructions

## Environment Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Flash Attention 2 requires CUDA and may take time to compile. If you encounter issues:
- Ensure you have CUDA toolkit installed
- For A100 servers, flash-attn should compile successfully
- If compilation fails, you can comment out `flash-attn` in requirements.txt and proceed without it

### 2. Verify Installation

Run the data module test:

```bash
python test_data_module.py
```

This will verify that:
- Label-to-caption conversion works
- Image/mask preprocessing works
- Dataset and DataLoader work correctly

## Dataset Preparation

### Recommended: FGADR Dataset

1. Download from: https://github.com/csyizhou/FGADR-2842-Dataset
2. Extract to `data/FGADR/`
3. Create JSONL files (see example below)

### JSONL Format

Create `data/train/data.jsonl`:

```jsonl
{"image_path": "data/FGADR/images/001.png", "caption": "2", "mask_path": "data/FGADR/masks/001.png"}
{"image_path": "data/FGADR/images/002.png", "caption": "Mild diabetic retinopathy", "mask_path": "data/FGADR/masks/002.png"}
{"image_path": "data/FGADR/images/003.png", "caption": "3", "mask_path": null}
```

Fields:
- `image_path`: Path to fundus image
- `caption`: DR grade (0-4) or text description
- `mask_path`: Path to segmentation mask (or null for text-only)

## Next Steps

After setup:
1. Verify data module: `python test_data_module.py`
2. Prepare your dataset
3. Continue with Task 4: Model extension

## A100 Server Optimization

If running on A100:
- Flash Attention 2 will provide 2-3x speedup
- Use BF16 mixed precision (configured in train_config.yaml)
- Increase batch size to 32+ (A100 has 40GB/80GB memory)
