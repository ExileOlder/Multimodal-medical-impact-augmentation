"""Image saving and dataset export utilities."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from PIL import Image


def save_generation_result(
    image: Image.Image,
    output_dir: str,
    metadata: Optional[Dict[str, Any]] = None,
    prefix: str = "generated"
) -> Dict[str, str]:
    """
    Save generated image with metadata.
    
    Args:
        image: Generated PIL Image
        output_dir: Output directory
        metadata: Optional metadata dict
        prefix: Filename prefix
        
    Returns:
        Dict with paths to saved files
    """
    output_dir = Path(output_dir)
    
    # Create unique directory for this generation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    result_dir = output_dir / f"{timestamp}_{unique_id}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Save image
    image_path = result_dir / f"{prefix}.png"
    image.save(image_path, format='PNG')
    
    # Save metadata if provided
    metadata_path = None
    if metadata is not None:
        metadata_path = result_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return {
        'result_dir': str(result_dir),
        'image_path': str(image_path),
        'metadata_path': str(metadata_path) if metadata_path else None
    }


def save_batch_results(
    images: List[Image.Image],
    output_dir: str,
    metadata_list: Optional[List[Dict[str, Any]]] = None,
    prefix: str = "generated"
) -> List[Dict[str, str]]:
    """
    Save batch of generated images.
    
    Args:
        images: List of generated PIL Images
        output_dir: Output directory
        metadata_list: Optional list of metadata dicts
        prefix: Filename prefix
        
    Returns:
        List of dicts with paths to saved files
    """
    output_dir = Path(output_dir)
    
    # Create batch directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_id = str(uuid.uuid4())[:8]
    batch_dir = output_dir / f"batch_{timestamp}_{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i, image in enumerate(images):
        # Save image
        image_path = batch_dir / f"{prefix}_{i:04d}.png"
        image.save(image_path, format='PNG')
        
        # Save metadata if provided
        metadata_path = None
        if metadata_list and i < len(metadata_list):
            metadata_path = batch_dir / f"metadata_{i:04d}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata_list[i], f, indent=2)
        
        results.append({
            'image_path': str(image_path),
            'metadata_path': str(metadata_path) if metadata_path else None
        })
    
    # Save batch summary
    summary_path = batch_dir / "batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'batch_id': batch_id,
            'timestamp': timestamp,
            'num_images': len(images),
            'results': results
        }, f, indent=2)
    
    return results


def create_dataset_manifest(
    image_dir: str,
    output_path: str,
    caption_field: str = "caption",
    mask_field: str = "mask_path"
) -> str:
    """
    Create JSONL manifest file for generated dataset.
    
    Args:
        image_dir: Directory containing generated images
        output_path: Path to output JSONL file
        caption_field: Field name for caption in metadata
        mask_field: Field name for mask path in metadata
        
    Returns:
        Path to created manifest file
    """
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    entries = []
    
    # Find all PNG images
    for image_path in sorted(image_dir.glob("**/*.png")):
        # Look for corresponding metadata
        metadata_path = image_path.parent / f"metadata_{image_path.stem.split('_')[-1]}.json"
        if not metadata_path.exists():
            metadata_path = image_path.parent / "metadata.json"
        
        entry = {
            'image_path': str(image_path.absolute())
        }
        
        # Load metadata if exists
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                entry[caption_field] = metadata.get(caption_field, "")
                if mask_field in metadata:
                    entry[mask_field] = metadata[mask_field]
                else:
                    entry[mask_field] = None
        else:
            entry[caption_field] = ""
            entry[mask_field] = None
        
        entries.append(entry)
    
    # Write JSONL file
    with open(output_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created manifest with {len(entries)} entries: {output_path}")
    return str(output_path)


def export_for_training(
    generated_dir: str,
    original_dir: str,
    output_manifest: str
) -> str:
    """
    Export combined dataset (original + generated) for training.
    
    Args:
        generated_dir: Directory with generated images
        original_dir: Directory with original images
        output_manifest: Path to output JSONL manifest
        
    Returns:
        Path to created manifest file
    """
    output_path = Path(output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    entries = []
    
    # Add generated images
    generated_dir = Path(generated_dir)
    for image_path in sorted(generated_dir.glob("**/*.png")):
        metadata_path = image_path.with_suffix('.json')
        if not metadata_path.exists():
            metadata_path = image_path.parent / "metadata.json"
        
        entry = {
            'image_path': str(image_path.absolute()),
            'source': 'generated'
        }
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                entry['caption'] = metadata.get('caption', '')
                entry['mask_path'] = metadata.get('mask_path', None)
        
        entries.append(entry)
    
    # Add original images
    original_dir = Path(original_dir)
    for image_path in sorted(original_dir.glob("**/*.png")) + sorted(original_dir.glob("**/*.jpg")):
        entry = {
            'image_path': str(image_path.absolute()),
            'source': 'original',
            'caption': '',  # To be filled
            'mask_path': None  # To be filled
        }
        entries.append(entry)
    
    # Write JSONL file
    with open(output_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Exported combined dataset with {len(entries)} entries: {output_path}")
    print(f"  Generated: {sum(1 for e in entries if e['source'] == 'generated')}")
    print(f"  Original: {sum(1 for e in entries if e['source'] == 'original')}")
    
    return str(output_path)
