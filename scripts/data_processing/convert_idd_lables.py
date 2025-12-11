#!/usr/bin/env python
"""
Script to convert IDD-AW JSON polygon annotations to PNG segmentation masks.
Based on the AutoNUE createLabels.py implementation.
"""

import logging
import os
import json
import argparse
from pathlib import Path
from multiprocessing import Pool
from functools import partial
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


# IDD-AW class definitions (update based on your specific label mapping)
IDD_CLASSES = {
    'road': 0,
    'sidewalk': 1,
    'building': 2,
    'wall': 3,
    'fence': 4,
    'pole': 5,
    'traffic light': 6,
    'traffic sign': 7,
    'vegetation': 8,
    'terrain': 9,
    'sky': 10,
    'person': 11,
    'rider': 12,
    'car': 13,
    'truck': 14,
    'bus': 15,
    'train': 16,
    'motorcycle': 17,
    'bicycle': 18,
    'autorickshaw': 19,
    'animal': 20,
    'void': 255,
}


# Common synonyms / aliases to improve robustness of label matching
LABEL_ALIASES = {
    'road': ['road', 'roadway', 'drivable', 'drivable area', 'drivable_area', 'road_kerb', 'lane', 'road marking', 'road_marking'],
    'sidewalk': ['sidewalk', 'pavement', 'footpath'],
}


def resolve_label(raw_label: str) -> str:
    """Normalize and resolve a raw label string to a canonical IDD_CLASSES key.

    Tries exact match, then token-based matching and aliases. Returns the
    canonical key if found, otherwise returns the lowercased stripped raw_label
    (which will likely fall back to void when looked up in IDD_CLASSES).
    """
    if not raw_label:
        return raw_label

    lab = raw_label.strip().lower()

    # Direct match
    if lab in IDD_CLASSES:
        return lab

    # Try to match tokens against known class keys
    tokens = [t.strip() for t in lab.replace('-', ' ').replace('_', ' ').split() if t.strip()]
    for t in tokens:
        if t in IDD_CLASSES:
            return t

    # Try alias mapping
    for canonical, aliases in LABEL_ALIASES.items():
        for a in aliases:
            if a in lab:
                return canonical

    # last resort: return original normalized label
    return lab


def polygon_to_mask(polygon_points, img_shape, class_id):
    """
    Convert polygon coordinates to a mask array.
    
    Args:
        polygon_points: List of [x, y] coordinates
        img_shape: Tuple of (height, width)
        class_id: Integer ID for the class
    
    Returns:
        numpy array with the polygon filled
    """
    height, width = img_shape
    
    # Create blank image
    mask = Image.new('L', (width, height), 0)
    
    # Convert polygon points to tuple format
    polygon_tuples = [tuple(point) for point in polygon_points]
    
    # Draw filled polygon as a binary mask (1 inside polygon, 0 outside).
    # We return a binary mask so callers can apply the intended class id
    # explicitly. This avoids the problem when class_id == 0 (e.g. 'road')
    # where fill=0 would be indistinguishable from background.
    ImageDraw.Draw(mask).polygon(polygon_tuples, outline=1, fill=1)

    return np.array(mask)


def process_json_file(json_path, output_dir, id_type='id', save_color=False):
    """
    Process a single JSON annotation file and generate PNG mask.
    
    Args:
        json_path: Path to JSON annotation file
        output_dir: Directory to save output PNG masks
        id_type: Type of ID to use for class mapping ('id', 'trainId', etc.)
        save_color: Whether to save color-coded visualization
    """
    try:
        # Read JSON file
        with open(json_path, 'r') as f:
            annotation = json.load(f)
        
        # Get image dimensions
        img_height = annotation.get('imgHeight', annotation.get('height', 1024))
        img_width = annotation.get('imgWidth', annotation.get('width', 1920))
        
        # Initialize mask with void/ignore class
        mask = np.ones((img_height, img_width), dtype=np.uint8) * 255
        
        # Process each object in the annotation
        objects = annotation.get('objects', [])
        
        for obj in objects:
            label = obj.get('label', '')
            polygon = obj.get('polygon', [])
            
            # Skip if deleted or invalid
            if obj.get('deleted', 0) == 1 or not polygon:
                continue
            
            # Get class ID using robust resolution to avoid mis-labeling
            canonical = resolve_label(label)
            class_id = IDD_CLASSES.get(canonical, 255)
 
             # Convert polygon to mask
            if len(polygon) >= 3:  # Need at least 3 points for a polygon
                obj_mask = polygon_to_mask(polygon, (img_height, img_width), class_id)

                # Update main mask: where obj_mask is set (==1), assign the intended
                # class_id. This ensures class_id == 0 (road) is correctly written
                # instead of being treated as background.
                mask[obj_mask > 0] = class_id
        
        # Save mask as PNG
        # Normalize json_path and remove trailing "_mask" from filename if present
        p = Path(json_path)
        if p.stem.endswith('_mask'):
            # Remove trailing "_mask" from filename stem while keeping suffix and parent
            json_path = str(p.with_name(p.stem[:-5] + p.suffix))
        output_path = Path(output_dir) / Path(json_path).with_suffix('.png').name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask_img.save(output_path)
        
        # Optionally save color-coded version for visualization
        if save_color:
            color_output_path = output_path.with_stem(output_path.stem + '_color')
            color_mask = create_color_mask(mask)
            color_img = Image.fromarray(color_mask.astype(np.uint8))
            color_img.save(color_output_path)
        
        return True, json_path
    
    except Exception as e:
        return False, f"Error processing {json_path}: {str(e)}"


def create_color_mask(mask):
    """
    Create color-coded visualization of segmentation mask.
    
    Args:
        mask: 2D numpy array with class IDs
    
    Returns:
        3D numpy array with RGB colors
    """
    # Define color palette (you can customize this)
    color_map = {
        0: [128, 64, 128],    # road - purple
        1: [244, 35, 232],    # sidewalk - pink
        2: [70, 70, 70],      # building - dark gray
        3: [102, 102, 156],   # wall - light purple
        4: [190, 153, 153],   # fence - light pink
        5: [153, 153, 153],   # pole - gray
        6: [250, 170, 30],    # traffic light - orange
        7: [220, 220, 0],     # traffic sign - yellow
        8: [107, 142, 35],    # vegetation - green
        9: [152, 251, 152],   # terrain - light green
        10: [70, 130, 180],   # sky - blue
        11: [220, 20, 60],    # person - red
        12: [255, 0, 0],      # rider - bright red
        13: [0, 0, 142],      # car - dark blue
        14: [0, 0, 70],       # truck - darker blue
        15: [0, 60, 100],     # bus - medium blue
        16: [0, 80, 100],     # train - blue-green
        17: [0, 0, 230],      # motorcycle - bright blue
        18: [119, 11, 32],    # bicycle - dark red
        19: [255, 127, 80],   # autorickshaw - coral
        20: [192, 192, 192],  # animal - silver
        255: [0, 0, 0],       # void - black
    }
    
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    for class_id, color in color_map.items():
        color_mask[mask == class_id] = color
    
    return color_mask


def find_json_files(data_dir, split='train'):
    """
    Find all JSON annotation files in the dataset directory.
    
    Args:
        data_dir: Root directory of the dataset
        split: Dataset split ('train', 'val', 'test')
    
    Returns:
        List of JSON file paths
    """
    json_dir = Path(data_dir) / split / 'labels'
    
    if not json_dir.exists():
        # Try alternative structure
        json_dir = Path(data_dir) / split
    
    if not json_dir.exists():
        raise ValueError(f"Could not find annotations directory at {json_dir}")
    
    json_files = list(json_dir.rglob('*.json'))
    return json_files


def main():
    parser = argparse.ArgumentParser(
        description='Convert IDD-AW JSON annotations to PNG segmentation masks'
    )
    parser.add_argument(
        '--datadir',
        type=str,
        required=True,
        help='Path to IDD-AW dataset root directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='',
        choices=['train', 'val', 'test'],
        help='Dataset split to process'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for PNG masks (default: datadir/split/masks)'
    )
    parser.add_argument(
        '--id-type',
        type=str,
        default='id',
        choices=['id', 'trainId'],
        help='Type of class ID to use'
    )
    parser.add_argument(
        '--color',
        action='store_true',
        help='Save color-coded visualization masks'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of parallel workers'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        output_dir = Path(args.datadir) / args.split / 'masks'
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files
    print(f"Searching for JSON files in {args.datadir}/{args.split}...")
    json_files = find_json_files(args.datadir, args.split)
    print(f"Found {len(json_files)} JSON annotation files")
    
    if len(json_files) == 0:
        print("No JSON files found. Please check your dataset path.")
        return
    
    # Process files in parallel
    print(f"Processing annotations with {args.num_workers} workers...")
    
    process_func = partial(
        process_json_file,
        output_dir=output_dir,
        id_type=args.id_type,
        save_color=args.color
    )
    
    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, json_files),
                total=len(json_files),
                desc="Converting annotations"
            ))
    else:
        results = []
        for json_file in tqdm(json_files, desc="Converting annotations"):
            results.append(process_func(json_file))
    
    # Report results
    successful = sum(1 for success, _ in results if success)
    failed = len(results) - successful
    
    print(f"\nConversion complete:")
    print(f"  ✓ Successful: {successful}/{len(json_files)}")
    if failed > 0:
        print(f"  ✗ Failed: {failed}")
        print("\nFailed files:")
        for success, msg in results:
            if not success:
                print(f"    {msg}")
    
    print(f"\nOutput masks saved to: {output_dir}")


if __name__ == '__main__':
    main()
