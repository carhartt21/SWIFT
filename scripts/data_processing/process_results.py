#!/usr/bin/env python3
"""
Image Processing Script (Updated Version)
Processes folders containing image-JSON pairs based on confidence values.

This script:
1. Removes files matching numbers in subdirectory-specific .txt files
2. Reads JSON files to extract confidence values
3. Sorts files by confidence and copies top N images to destination

Updated to handle:
- Separate removal files for each subdirectory (e.g., cloudy.txt for cloudy/ folder)
- Text files contain only the number part (e.g., "000005" for cloudy_000005.jpg)
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

def load_files_to_remove_for_subdir(removal_folder: str, subdir_name: str) -> set:
    """
    Load file numbers to remove from a subdirectory-specific .txt file.

    Args:
        removal_folder: Folder containing the removal .txt files
        subdir_name: Name of subdirectory (used to find corresponding .txt file)

    Returns:
        Set of file numbers to remove
    """
    if not removal_folder:
        return set()

    txt_file_path = os.path.join(removal_folder, f"{subdir_name}.txt")

    if not os.path.exists(txt_file_path):
        print(f"Info: No removal file found at {txt_file_path}. No files will be removed for {subdir_name}.")
        return set()

    file_numbers_to_remove = set()
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            file_number = line.strip()
            if file_number:
                file_numbers_to_remove.add(file_number)

    print(f"Loaded {len(file_numbers_to_remove)} file numbers to remove from {txt_file_path}")
    return file_numbers_to_remove

def remove_specified_files_by_number(directory: str, subdir_name: str, file_numbers_to_remove: set) -> int:
    """
    Remove files (both .jpg and .json) that match the specified file numbers.
    Constructs filenames using pattern: {subdir_name}_{file_number}.{extension}

    Args:
        directory: Directory to process
        subdir_name: Name of subdirectory (used in filename pattern)
        file_numbers_to_remove: Set of file numbers to remove

    Returns:
        Number of files removed
    """
    removed_count = 0

    for file_number in file_numbers_to_remove:
        jpg_filename = f"{subdir_name}_{file_number}.jpg"
        json_filename = f"{subdir_name}_{file_number}.json"

        jpg_path = os.path.join(directory, jpg_filename)
        json_path = os.path.join(directory, json_filename)

        if os.path.exists(jpg_path):
            os.remove(jpg_path)
            removed_count += 1
            print(f"Removed: {jpg_path}")

        if os.path.exists(json_path):
            os.remove(json_path)
            removed_count += 1
            print(f"Removed: {json_path}")

    return removed_count

def read_json_files(directory: str) -> List[Tuple[str, float, Dict]]:
    """
    Read all JSON files in directory and extract confidence values.

    Args:
        directory: Directory containing JSON files

    Returns:
        List of tuples (base_filename, confidence, json_data)
    """
    json_files = []

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            json_path = os.path.join(directory, filename)
            base_filename = filename[:-5]  # Remove .json extension

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                if directory.endswith('foggy') or directory.endswith('fog'):
                    confidence = json_data.get('margin', 0.0) #only for fog images
                else:
                    confidence = json_data.get('confidence', 0.0) #for all other categories
                json_files.append((base_filename, confidence, json_data))

            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading {json_path}: {e}")
                continue

    print(f"Successfully read {len(json_files)} JSON files from {directory}")
    return json_files

def copy_top_confidence_images(json_files: List[Tuple[str, float, Dict]], 
                              source_dir: str, 
                              dest_dir: str, 
                              top_n: int) -> int:
    """
    Sort files by confidence and copy top N images to destination.

    Args:
        json_files: List of (base_filename, confidence, json_data) tuples
        source_dir: Source directory containing the files
        dest_dir: Destination directory for copied files
        top_n: Number of top files to copy

    Returns:
        Number of files successfully copied
    """
    # Sort by confidence in descending order
    sorted_files = sorted(json_files, key=lambda x: x[1], reverse=True)

    # Take top N files
    top_files = sorted_files[:top_n]

    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    copied_count = 0

    for base_filename, confidence, json_data in top_files:
        jpg_source = os.path.join(source_dir, f"{base_filename}.jpg")
        json_source = os.path.join(source_dir, f"{base_filename}.json")
        jpg_dest = os.path.join(dest_dir, f"{base_filename}.jpg")
        json_dest = os.path.join(dest_dir, f"{base_filename}.json")

        try:
            # Copy image file
            if os.path.exists(jpg_source):
                shutil.copy2(jpg_source, jpg_dest)
                copied_count += 1

            # Copy JSON file
            if os.path.exists(json_source):
                shutil.copy2(json_source, json_dest)
                copied_count += 1

            print(f"Copied {base_filename} (confidence: {confidence:.4f})")

        except IOError as e:
            print(f"Error copying {base_filename}: {e}")
            continue

    print(f"Successfully copied {len(top_files)} file pairs to {dest_dir}")
    return copied_count

def process_subdirectory(parent_dir: str, 
                        subdir_name: str, 
                        removal_folder: str,
                        dest_base_dir: str,
                        top_n: int) -> Dict[str, int]:
    """
    Process a single subdirectory.

    Args:
        parent_dir: Parent directory path
        subdir_name: Name of subdirectory to process
        removal_folder: Folder containing removal .txt files
        dest_base_dir: Base destination directory
        top_n: Number of top files to copy

    Returns:
        Dictionary with processing statistics
    """
    subdir_path = os.path.join(parent_dir, subdir_name)

    if not os.path.exists(subdir_path):
        print(f"Warning: Subdirectory {subdir_path} not found. Skipping.")
        return {"removed": 0, "copied": 0, "json_files": 0}

    print(f"\nProcessing subdirectory: {subdir_name}")

    # Step 1: Load file numbers to remove for this specific subdirectory
    file_numbers_to_remove = load_files_to_remove_for_subdir(removal_folder, subdir_name)

    # Step 2: Remove specified files
    removed_count = remove_specified_files_by_number(subdir_path, subdir_name, file_numbers_to_remove)

    # Step 3: Read JSON files and extract confidence values
    json_files = read_json_files(subdir_path)

    if not json_files:
        print(f"No JSON files found in {subdir_path}")
        return {"removed": removed_count, "copied": 0, "json_files": 0}

    # Step 4: Copy top N files to destination
    dest_dir = os.path.join(dest_base_dir, subdir_name)
    copied_count = copy_top_confidence_images(json_files, subdir_path, dest_dir, top_n)

    return {
        "removed": removed_count,
        "copied": copied_count,
        "json_files": len(json_files)
    }

def main():
    """Main function to coordinate the processing"""
    parser = argparse.ArgumentParser(
        description="Process image-JSON pairs based on confidence values (Updated Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_images.py /path/to/parent --subdirs cloudy rainy snowy \
                          --removal-folder /path/to/removal_lists \
                          --destination /path/to/dest \
                          --top-n 100

  python process_images.py /data/weather --subdirs clear_day cloudy foggy \
                          --removal-folder /data/blacklists \
                          --destination /data/processed \
                          --top-n 50

File naming convention:
  - Image files: {subdir_name}_{file_number}.jpg (e.g., cloudy_000005.jpg)
  - JSON files: {subdir_name}_{file_number}.json (e.g., cloudy_000005.json)
  - Removal files: {subdir_name}.txt (e.g., cloudy.txt containing "000005")
        """
    )

    parser.add_argument('parent_dir', 
                       help='Parent directory containing subdirectories with image-JSON pairs')

    parser.add_argument('--subdirs', 
                       nargs='+', 
                       required=True,
                       help='List of subdirectory names to process')

    parser.add_argument('--removal-folder', 
                       help='Folder containing .txt files with file numbers to remove (one file per subdirectory)')

    parser.add_argument('--destination', 
                       required=True,
                       help='Base destination directory for copied files')

    parser.add_argument('--top-n', 
                       type=int, 
                       default=100,
                       help='Number of top confidence files to copy (default: 100)')

    parser.add_argument('--dry-run', 
                       action='store_true',
                       help='Show what would be done without actually processing files')

    args = parser.parse_args()

    # Validate input arguments
    if not os.path.exists(args.parent_dir):
        print(f"Error: Parent directory {args.parent_dir} does not exist")
        return 1

    if args.removal_folder and not os.path.exists(args.removal_folder):
        print(f"Warning: Removal folder {args.removal_folder} does not exist")
        print("No files will be removed.")
        args.removal_folder = None

    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        print(f"Parent directory: {args.parent_dir}")
        print(f"Subdirectories to process: {args.subdirs}")
        print(f"Removal folder: {args.removal_folder or 'None'}")
        if args.removal_folder:
            for subdir in args.subdirs:
                removal_file = os.path.join(args.removal_folder, f"{subdir}.txt")
                exists = "✓" if os.path.exists(removal_file) else "✗"
                print(f"  {exists} {removal_file}")
        print(f"Destination: {args.destination}")
        print(f"Top N files to copy: {args.top_n}")
        return 0

    # Process each subdirectory
    total_stats = {"removed": 0, "copied": 0, "json_files": 0}

    for subdir_name in args.subdirs:
        stats = process_subdirectory(
            args.parent_dir, 
            subdir_name, 
            args.removal_folder,
            args.destination,
            args.top_n
        )

        # Update total statistics
        for key in total_stats:
            total_stats[key] += stats[key]

    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Subdirectories processed: {len(args.subdirs)}")
    print(f"Total JSON files processed: {total_stats['json_files']}")
    print(f"Total files removed: {total_stats['removed']}")
    print(f"Total files copied: {total_stats['copied']}")
    print(f"Destination directory: {args.destination}")

    return 0

if __name__ == "__main__":
    exit(main())
