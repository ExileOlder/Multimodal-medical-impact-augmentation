"""JSONL data loader for medical image dataset."""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Union


def load_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing image_path, caption, and optional mask_path
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                entry = json.loads(line)
                
                # Validate required fields
                if 'image_path' not in entry:
                    print(f"Warning: Line {line_num} missing 'image_path', skipping")
                    continue
                
                if 'caption' not in entry:
                    print(f"Warning: Line {line_num} missing 'caption', skipping")
                    continue
                
                # Handle missing mask_path - mark as text-only mode
                if 'mask_path' not in entry or entry['mask_path'] is None:
                    entry['mask_path'] = None
                    entry['text_only'] = True
                else:
                    entry['text_only'] = False
                
                data.append(entry)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num}: {e}")
                continue
    
    return data


def load_jsonl_folder(folder_path: Union[str, Path]) -> List[Dict]:
    """
    Load all JSONL files from a folder.
    
    Args:
        folder_path: Path to folder containing JSONL files
        
    Returns:
        Combined list of all data entries from all JSONL files
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    all_data = []
    jsonl_files = list(folder_path.glob("*.jsonl")) + list(folder_path.glob("*.json"))
    
    if not jsonl_files:
        print(f"Warning: No JSONL files found in {folder_path}")
        return all_data
    
    print(f"Found {len(jsonl_files)} JSONL file(s) in {folder_path}")
    
    for jsonl_file in jsonl_files:
        print(f"Loading {jsonl_file.name}...")
        try:
            data = load_jsonl(jsonl_file)
            all_data.extend(data)
            print(f"  Loaded {len(data)} entries")
        except Exception as e:
            print(f"  Error loading {jsonl_file.name}: {e}")
            continue
    
    print(f"Total entries loaded: {len(all_data)}")
    return all_data


def validate_paths(data: List[Dict], base_path: Optional[Union[str, Path]] = None) -> List[Dict]:
    """
    Validate that image and mask paths exist.
    
    Args:
        data: List of data entries
        base_path: Optional base path to prepend to relative paths
        
    Returns:
        List of valid entries (with existing files)
    """
    if base_path:
        base_path = Path(base_path)
    
    valid_data = []
    missing_images = 0
    missing_masks = 0
    
    for entry in data:
        image_path = Path(entry['image_path'])
        if base_path and not image_path.is_absolute():
            image_path = base_path / image_path
        
        if not image_path.exists():
            missing_images += 1
            continue
        
        # Update to absolute path
        entry['image_path'] = str(image_path)
        
        # Check mask path if provided
        if entry['mask_path'] is not None:
            mask_path = Path(entry['mask_path'])
            if base_path and not mask_path.is_absolute():
                mask_path = base_path / mask_path
            
            if not mask_path.exists():
                missing_masks += 1
                # Convert to text-only mode
                entry['mask_path'] = None
                entry['text_only'] = True
            else:
                entry['mask_path'] = str(mask_path)
        
        valid_data.append(entry)
    
    if missing_images > 0:
        print(f"Warning: {missing_images} entries skipped due to missing images")
    if missing_masks > 0:
        print(f"Warning: {missing_masks} entries converted to text-only mode due to missing masks")
    
    return valid_data
