#!/usr/bin/env python3
"""
Segmentation Label Copier Script
================================

This script searches for JSON files in a directory, extracts the "original_image_path" field
from each JSON file, finds the corresponding segmentation label file, and copies it to a new
directory using the name of the JSON file.

Supports multiple dataset formats:
- ACDC: RGB images to multiple label types (gt_labelColor, gt_labelIds, gt_labelTrainIds)
- BDD10k: 10k folder to labels folder with train_id suffix

Usage:
    python copy_segmentation_labels.py <json_dir> <base_path> <labels_dir> <output_dir> [options]

Examples:
    # ACDC dataset
    python copy_segmentation_labels.py ./results ./ACDC ./ACDC ./copied_labels
    
    # BDD10k dataset
    python copy_segmentation_labels.py ./results ./BDD10k ./BDD10k ./copied_labels
"""

import os
import json
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_json_files(directory: str, recursive: bool = True) -> List[Path]:
    """
    Find all JSON files in the given directory and optionally subdirectories.
    
    Args:
        directory: Directory to search for JSON files
        recursive: Whether to search subdirectories recursively
        
    Returns:
        List of Path objects for found JSON files
    """
    json_dir = Path(directory)
    if not json_dir.exists():
        logger.error(f"JSON directory does not exist: {directory}")
        return []
    
    if recursive:
        json_files = list(json_dir.rglob("*.json"))
    else:
        json_files = list(json_dir.glob("*.json"))
        
    logger.info(f"Found {len(json_files)} JSON files in {directory} (recursive: {recursive})")
    return json_files


def extract_original_image_path(json_file: Path) -> Optional[str]:
    """
    Extract the "original_image_path" field from a JSON file.
    
    Args:
        json_file: Path to the JSON file
        
    Returns:
        The original image path string, or None if not found
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_path = data.get('original_image_path')
        if original_path:
            return original_path
        else:
            logger.warning(f"No 'original_image_path' field found in {json_file}")
            return None
            
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file {json_file}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading JSON file {json_file}: {e}")
        return None


def find_label_files(original_image_path: str, base_path: str, labels_dir: str, label_extensions: List[str] = None, label_types: List[str] = None) -> List[Path]:
    """
    Find the corresponding segmentation label files for an image.
    Supports ACDC dataset mapping from RGB images to multiple label types.
    Supports BDD10k dataset mapping from 10k folder to labels folder.
    
    Args:
        original_image_path: Path to the original image
        base_path: Base path for original images  
        labels_dir: Directory containing segmentation labels
        label_extensions: List of possible label file extensions
        label_types: List of label types (e.g., ['gt_labelColor', 'gt_labelIds', 'train_id'])
        
    Returns:
        List of paths to the label files
    """
    if label_extensions is None:
        label_extensions = ['.png', '.jpg', '.jpeg']
    
    if label_types is None:
        # Default label types for ACDC dataset and BDD10k dataset
        label_types = ['gt_labelColor', 'gt_labelIds', 'gt_labelTrainIds', 'train_id']
    
    # Extract information from the image path
    image_path = Path(base_path) / original_image_path
    image_name = image_path.stem

    logging.debug(f"Processing image: {image_path}")
    logging.debug(f"Image path parts: {image_path.parts}")
    
    labels_path = Path(labels_dir)
    if not labels_path.exists():
        logger.error(f"Labels directory does not exist: {labels_dir}")
        return []

    found_labels = []    # Handle BDD10k dataset mapping
    if '/10k/' in str(image_path):
        # BDD10k format: /path/BDD10k/10k/test/ceab7651-c77560d4.jpg -> /path/BDD10k/labels/test/ceab7651-c77560d4_train_id.png
        base_name = image_name
        logging.debug(f"Mapping BDD10k image name {image_name} to base name {base_name}")
        
        # Extract the relative path structure for BDD10k
        # Convert 10k/test/... to test/... (labels_dir already points to labels folder)
        image_parts = image_path.parts
        if '10k' in image_parts:
            # Find indices
            bdd10k_idx = image_parts.index('10k')
            # Reconstruct path: just use the subdirectory after 10k (e.g., test, train, val)
            label_parts = list(image_parts[bdd10k_idx+1:-1])
            label_subpath = Path(*label_parts) if label_parts else Path('.')
            
            logging.debug(f"BDD10k label subpath: {label_subpath}")
            
            # BDD10k uses train_id suffix
            bdd10k_label_types = ['train_id']
            for label_type in bdd10k_label_types:
                for ext in ['.png']:
                    label_filename = f"{base_name}_{label_type}{ext}"
                    potential_label = labels_path / label_subpath / label_filename
                    logging.debug(f"Checking for BDD10k label file: {potential_label}")
                    if potential_label.exists():
                        found_labels.append(potential_label)
                        logger.debug(f"Found BDD10k label: {potential_label}")
                    else: 
                        logger.debug(f"BDD10k label file not found: {potential_label}")
    
    # Handle ACDC dataset mapping
    elif 'rgb_ref_anon' in image_name or 'rgb_anon' in image_name:
        # ACDC format: GOPR0475_frame_000235_rgb_ref_anon -> GOPR0475_frame_000235_gt_labelType
        base_name = image_name.replace('_ref_anon', '_anon')
        base_name = base_name.replace('_rgb_anon', '')
        logging.debug(f"Mapping ACDC image name {image_name} to base name {base_name}")
        
        # Extract the relative path structure for ACDC
        # Convert rgb_anon_trainvaltest/rgb_anon/fog/train/... to gt_trainval/gt/fog/train/...
        image_parts = image_path.parts
        if 'rgb_anon' in image_parts:
            # Find indices
            rgb_anon_idx = image_parts.index('rgb_anon')
            if rgb_anon_idx > 0 and image_parts[rgb_anon_idx-1] == 'rgb_anon_trainvaltest':
                # Reconstruct path: ACDC/gt_trainval/gt/fog/train/GOPR0475/
                # Build the ACDC label subpath, removing any '_ref' occurrences from path parts
                parts = list(image_parts[rgb_anon_idx+1:-1])
                clean_parts = [p.replace('_ref', '') for p in parts]
                clean_parts = [p.replace('10k', 'labels') for p in clean_parts]

                label_subpath = Path(*clean_parts)
                logging.debug(f"ACDC label subpath: {label_subpath}")
                
                for label_type in label_types:
                    for ext in label_extensions:
                        label_filename = f"{base_name}_{label_type}{ext}"
                        potential_label = labels_path / label_subpath / label_filename
                        logging.debug(f"Checking for label file: {potential_label}")
                        if potential_label.exists():
                            found_labels.append(potential_label)
                            logger.debug(f"Found ACDC label: {potential_label}")
                        else: 
                            logger.debug(f"Label file not found: {potential_label}")
    # rgb_anon/fog/train_ref/GOPR0475/GOPR0475_frame_000148_rgb_ref_anon.png
    # gt_trainval/gt/fog/train/GOPR0475/GOPR0475_frame_000148_gt_invIds.png    

    # Fallback: try direct filename matching
    if not found_labels:
        base_name = image_name
        # Remove common image suffixes
        for suffix in ['_rgb_ref_anon', '_rgb', '_image', '_img']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        
        # Try to find label files with different extensions and types
        for label_type in [''] + label_types:  # Include empty string for direct matching
            search_name = f"{base_name}_{label_type}" if label_type else base_name
            for ext in label_extensions:
                # Direct search
                potential_label = labels_path / f"{search_name}{ext}"
                if potential_label.exists():
                    found_labels.append(potential_label)
                
                # Recursive search
                matches = list(labels_path.rglob(f"{search_name}{ext}"))
                found_labels.extend(matches)
    
    # Remove duplicates while preserving order
    unique_labels = []
    seen = set()
    for label in found_labels:
        if label not in seen:
            unique_labels.append(label)
            seen.add(label)
    
    if not unique_labels:
        logger.warning(f"No label files found for {image_path} in {labels_dir}")
    else:
        logger.info(f"Found {len(unique_labels)} label files for {image_name}")
    
    return unique_labels


def copy_label_files(label_files: List[Path], output_dir: str, base_name: str, relative_path: str = "", preserve_extension: bool = True) -> int:
    """
    Copy multiple label files to the output directory with new names, preserving directory structure.
    
    Args:
        label_files: List of paths to the source label files
        output_dir: Destination directory
        base_name: Base filename (from JSON file)
        relative_path: Relative path to preserve directory structure
        preserve_extension: Whether to preserve the original label file extension
        
    Returns:
        Number of files successfully copied
    """
    copied_count = 0
    
    try:
        # Create output path preserving directory structure
        output_path = Path(output_dir) / relative_path if relative_path else Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        json_stem = Path(base_name).stem
        
        for i, label_file in enumerate(label_files):
            try:
                # Determine the output filename
                if preserve_extension:
                    label_ext = label_file.suffix
                    # Extract label type from filename if present
                    label_stem = label_file.stem
                    if '_gt_' in label_stem:
                        # Extract the label type (e.g., 'gt_labelColor', 'gt_labelIds')
                        label_type = label_stem.split('_gt_')[-1]
                        output_filename = f"{json_stem}_gt_{label_type}{label_ext}"
                    elif '_train_id' in label_stem:
                        # Handle BDD10k train_id suffix
                        output_filename = f"{json_stem}_train_id{label_ext}"
                    else:
                        # Use index if no clear type
                        output_filename = f"{json_stem}_{i:02d}{label_ext}"
                else:
                    # Use the JSON filename with index
                    output_filename = f"{json_stem}_{i:02d}.png"  # Default to PNG
                
                output_file = output_path / output_filename
                
                # Copy the file
                shutil.copy2(label_file, output_file)
                logger.debug(f"Copied {label_file} -> {output_file}")
                copied_count += 1
                
            except Exception as e:
                logger.error(f"Error copying {label_file}: {e}")
        
    except Exception as e:
        logger.error(f"Error setting up output directory {output_path}: {e}")
    
    return copied_count


def process_json_files(json_dir: str, labels_dir: str, base_path: str, output_dir: str, 
                      label_extensions: List[str] = None, 
                      label_types: List[str] = None,
                      preserve_extension: bool = True,
                      preserve_structure: bool = True,
                      dry_run: bool = False) -> Dict[str, int]:
    """
    Process all JSON files and copy corresponding label files, preserving directory structure.
    
    Args:
        json_dir: Directory containing JSON files
        labels_dir: Directory containing segmentation labels
        base_path: Base path for original images
        output_dir: Directory to copy labels to
        label_extensions: List of possible label file extensions
        label_types: List of label types for ACDC dataset
        preserve_extension: Whether to preserve original label file extensions
        preserve_structure: Whether to preserve directory structure in output
        dry_run: If True, only simulate the process without copying files
        
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'total_json_files': 0,
        'json_files_processed': 0,
        'labels_found': 0,
        'labels_copied': 0,
        'errors': 0,
        'directories_processed': set()
    }
    
    # Find all JSON files recursively
    json_files = find_json_files(json_dir, recursive=True)
    stats['total_json_files'] = len(json_files)
    
    if not json_files:
        logger.warning("No JSON files found to process")
        return stats
    
    json_base_path = Path(json_dir)
    
    # Process each JSON file
    for json_file in json_files:
        try:
            # Calculate relative path to preserve directory structure
            relative_path = ""
            if preserve_structure:
                try:
                    relative_path = str(json_file.parent.relative_to(json_base_path))
                    if relative_path == ".":
                        relative_path = ""
                    stats['directories_processed'].add(relative_path if relative_path else "root")
                except ValueError:
                    # json_file is not under json_base_path
                    relative_path = ""
            
            # Extract original image path
            original_image_path = extract_original_image_path(json_file)
            if not original_image_path:
                stats['errors'] += 1
                continue
            
            stats['json_files_processed'] += 1
            
            # Find corresponding label files
            label_files = find_label_files(original_image_path, base_path, labels_dir, label_extensions, label_types)
            if not label_files:
                stats['errors'] += 1
                continue
            
            stats['labels_found'] += len(label_files)
            
            # Copy label files with JSON filename, preserving structure
            base_name = json_file.name
            if dry_run:
                output_path_display = Path(output_dir) / relative_path if relative_path else Path(output_dir)
                logger.info(f"[DRY RUN] Would copy {len(label_files)} label files for {base_name} to {output_path_display}")
                for label_file in label_files:
                    logger.info(f"  [DRY RUN] {label_file} -> {output_path_display}/{base_name}_*")
                stats['labels_copied'] += len(label_files)
            else:
                copied_count = copy_label_files(label_files, output_dir, base_name, relative_path, preserve_extension)
                stats['labels_copied'] += copied_count
                if copied_count != len(label_files):
                    stats['errors'] += len(label_files) - copied_count
                    
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            stats['errors'] += 1
    
    # Convert set to count for final stats
    stats['directories_processed'] = len(stats['directories_processed'])
    
    return stats


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='Copy segmentation labels based on JSON file references',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./results ./base_path ./sa1b_labels ./copied_labels
  %(prog)s ./json_files ./images ./labels ./output --extensions .png .npy
  %(prog)s ./data ./base ./masks ./out --no-preserve-ext --no-preserve-structure
  %(prog)s ./results ./base ./labels ./output --dry-run --verbose
        """
    )
    
    parser.add_argument('json_dir', help='Directory containing JSON files')
    parser.add_argument('base_path', help='Base path for original images')
    parser.add_argument('labels_dir', help='Directory containing segmentation labels')
    parser.add_argument('output_dir', help='Output directory for copied labels')
    
    parser.add_argument('--extensions', nargs='+', 
                       default=['.png', '.jpg', '.jpeg'],
                       help='Label file extensions to search for')
    parser.add_argument('--label-types', nargs='+',
                       default=['gt_labelColor', 'gt_labelIds', 'gt_labelTrainIds', 'train_id'],
                       help='Label types for ACDC/BDD10k datasets (e.g., gt_labelColor train_id)')
    parser.add_argument('--no-preserve-ext', action='store_true', 
                       help='Do not preserve original label file extension')
    parser.add_argument('--no-preserve-structure', action='store_true',
                       help='Do not preserve directory structure in output')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate the process without actually copying files')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directories
    if not Path(args.json_dir).exists():
        logger.error(f"JSON directory does not exist: {args.json_dir}")
        return 1
    
    if not Path(args.labels_dir).exists():
        logger.error(f"Labels directory does not exist: {args.labels_dir}")
        return 1
    
    # Process files
    logger.info("=" * 50)
    logger.info("SEGMENTATION LABEL COPIER")
    logger.info(f"JSON directory: {args.json_dir}")
    logger.info(f"Base path: {args.base_path}")
    logger.info(f"Labels directory: {args.labels_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Label extensions: {args.extensions}")
    logger.info(f"Label types: {args.label_types}")
    logger.info(f"Preserve extension: {not args.no_preserve_ext}")
    logger.info(f"Preserve structure: {not args.no_preserve_structure}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 50)
    
    try:
        stats = process_json_files(
            json_dir=args.json_dir,
            labels_dir=args.labels_dir,
            base_path=args.base_path,
            output_dir=args.output_dir,
            label_extensions=args.extensions,
            label_types=args.label_types,
            preserve_extension=not args.no_preserve_ext,
            preserve_structure=not args.no_preserve_structure,
            dry_run=args.dry_run
        )
        
        # Print final statistics
        logger.info("=" * 50)
        logger.info("PROCESSING COMPLETE")
        logger.info(f"Total JSON files found: {stats['total_json_files']}")
        logger.info(f"JSON files processed: {stats['json_files_processed']}")
        logger.info(f"Directories processed: {stats['directories_processed']}")
        logger.info(f"Label files found: {stats['labels_found']}")
        logger.info(f"Label files copied: {stats['labels_copied']}")
        logger.info(f"Errors encountered: {stats['errors']}")
        
        if stats['labels_copied'] > 0:
            success_rate = (stats['labels_copied'] / stats['total_json_files']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        logger.info("=" * 50)
        
        return 0
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())