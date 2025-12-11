#!/usr/bin/env python3
"""
Image Resizer Script
Resizes all images in subdirectories so that the short side becomes 1024 pixels
while maintaining aspect ratio and folder structure.
"""

import os
import sys
from PIL import Image
import argparse
from pathlib import Path

def resize_image_short_side(image_path, output_path, target_short_side=1024):
    """
    Resize image so that the shorter side becomes target_short_side pixels
    while maintaining aspect ratio.
    """
    try:
        with Image.open(image_path) as img:
            # Get current dimensions
            width, height = img.size

            # Determine which side is shorter
            if width < height:
                # Width is shorter
                new_width = target_short_side
                new_height = int((height / width) * target_short_side)
            else:
                # Height is shorter (or equal)
                new_height = target_short_side
                new_width = int((width / height) * target_short_side)

            # Resize image
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save resized image
            resized_img.save(output_path, quality=95, optimize=True)

            print(f"Resized: {image_path} -> {output_path} ({width}x{height} -> {new_width}x{new_height})")
            return True

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def is_image_file(filename):
    """Check if file is a supported image format"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    return Path(filename).suffix.lower() in image_extensions

def is_label_file(filename):
    """Check if file is a likely segmentation label image (PNG recommended)."""
    return Path(filename).suffix.lower() in {'.png'}

def resize_label_image(label_path, output_path, target_short_side=1024):
    """Resize a semantic segmentation label with nearest neighbor interpolation.

    Preserves class indices by using NEAREST interpolation and forces mode 'P' or 'L'
    depending on source. If the image is RGBA or RGB (unexpected for labels), it will
    be converted to 'P' if palette exists or 'L' otherwise.
    """
    try:
        with Image.open(label_path) as lbl:
            width, height = lbl.size

            # Determine scaling similar to images
            if width < height:
                new_width = target_short_side
                new_height = int((height / width) * target_short_side)
            else:
                new_height = target_short_side
                new_width = int((width / height) * target_short_side)

            # Convert mode for safety (labels should be single channel or palette)
            if lbl.mode not in ('P', 'L'):
                try:
                    lbl = lbl.convert('P')
                except Exception:
                    lbl = lbl.convert('L')

            resized = lbl.resize((new_width, new_height), Image.Resampling.NEAREST)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            resized.save(output_path)
            print(f"Label resized: {label_path} -> {output_path} ({width}x{height} -> {new_width}x{new_height})")
            return True
    except Exception as e:
        print(f"Error resizing label {label_path}: {e}")
        return False

def resize_labels_in_directory(labels_dir, output_dir, target_short_side=1024):
    """Resize all label images in a directory tree using nearest neighbor.

    Maintains folder structure inside output_dir.
    """
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)
    if not labels_path.exists():
        print(f"Error: Labels directory '{labels_dir}' does not exist.")
        return

    processed = 0
    errors = 0
    for root, _, files in os.walk(labels_path):
        root_path = Path(root)
        try:
            rel = root_path.relative_to(labels_path)
        except ValueError:
            rel = Path('.')
        out_sub = output_path / rel
        for fname in files:
            if is_label_file(fname):
                src = root_path / fname
                dest = out_sub / fname
                ok = resize_label_image(src, dest, target_short_side)
                processed += int(ok)
                errors += int(not ok)
    print(f"\nLabel resizing complete: {processed} succeeded, {errors} errors")

def resize_labels_for_all_datasets(datasets_root, output_folder_name='labels_resized', target_short_side=1024, dry_run=False):
    """Traverse datasets under datasets_root and resize labels in each dataset's 'labels' folder.

    For every immediate subdirectory of datasets_root (treated as a dataset), if a
    '<dataset>/labels' directory exists, resize all label images inside using nearest
    neighbor interpolation and write them to '<dataset>/<output_folder_name>', preserving
    the internal folder structure.

    Args:
        datasets_root (str|Path): Path containing dataset folders as immediate children.
        output_folder_name (str): Name of the sibling folder to write resized labels into.
        target_short_side (int): Target size for the shorter side.
        dry_run (bool): If True, only print what would be done.
    """
    root = Path(datasets_root)
    if not root.exists():
        print(f"Error: datasets_root does not exist: {datasets_root}")
        return

    datasets = [d for d in sorted(root.iterdir()) if d.is_dir()]
    print(f"Found {len(datasets)} candidate datasets under: {root}")
    totals = {"datasets": 0, "processed": 0}
    for ds in datasets:
        labels_dir = ds / 'labels'
        if not labels_dir.exists() or not labels_dir.is_dir():
            continue
        totals["datasets"] += 1
        out_dir = ds / output_folder_name
        print(f"Dataset: {ds.name}")
        print(f"  Labels: {labels_dir}")
        print(f"  Output: {out_dir}")
        print(f"  Target short side: {target_short_side}")
        if dry_run:
            # Count label files to report
            count = 0
            for _, _, files in os.walk(labels_dir):
                count += sum(1 for f in files if is_label_file(f))
            print(f"  [DRY-RUN] Would resize {count} label files")
        else:
            resize_labels_in_directory(str(labels_dir), str(out_dir), target_short_side)
            totals["processed"] += 1
    print(f"\nSummary: {totals['processed']}/{totals['datasets']} datasets with 'labels' processed")

def resize_images_in_directory(input_dir, output_dir, target_short_side=1024):
    """
    Process all images in input_dir and its subdirectories,
    saving resized versions to output_dir with same structure.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    processed_count = 0
    error_count = 0

    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_path):
        root_path = Path(root)

        # Calculate relative path from input directory
        try:
            relative_path = root_path.relative_to(input_path)
        except ValueError:
            # This shouldn't happen with os.walk, but just in case
            relative_path = Path(".")

        # Create corresponding output directory
        output_subdir = output_path / relative_path

        # Process all image files in current directory
        for filename in files:
            if is_image_file(filename):
                input_file = root_path / filename
                output_file = output_subdir / filename

                success = resize_image_short_side(input_file, output_file, target_short_side)

                if success:
                    processed_count += 1
                else:
                    error_count += 1

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Errors encountered: {error_count} images")

def main():
    parser = argparse.ArgumentParser(
        description="Resize images so short side becomes specified number of pixels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python resize_images.py /path/to/input /path/to/output
  python resize_images.py /path/to/input /path/to/output --size 512
  python resize_images.py ./dataset ./dataset_resized --size 2048
        """
    )
    # Add a convenience flag to process predefined AWARE datasets and exit
    AWARE_DATASETS = ["ACDC", "BDD10k", "BDD100k", "IDD", "IDD-AW", "MapillaryVistas", "OUTSIDE15k"]

    class ProcessAwareDataset(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            dataset = str(values)
            base = Path("/media/chge7185/HDD1/datasets/AWARE") / dataset / "categories"
            input_dir = base / "original_size"
            out_large = base / "large"
            out_medium = base / "medium"

            print(f"Dataset: {dataset}")
            print(f"Input: {input_dir}")
            print(f"Output (1024): {out_large}")
            print(f"Output (512): {out_medium}")
            print("-" * 50)

            # resize_images_in_directory(str(input_dir), str(out_large), 1024)
            resize_images_in_directory(str(input_dir), str(out_medium), 512)
            sys.exit(0)

    class ProcessAwareOrAll(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            v = str(values)
            datasets = AWARE_DATASETS if v == "ALL" else [v]
            for dataset in datasets:
                base = Path("/media/chge7185/HDD1/datasets/AWARE") / dataset / "categories"
                input_dir = base / "original_size"
                out_large = base / "large"
                out_medium = base / "medium"

                print(f"Dataset: {dataset}")
                print(f"Input: {input_dir}")
                print(f"Output (1024): {out_large}")
                print(f"Output (512): {out_medium}")
                print("-" * 50)

                # resize_images_in_directory(str(input_dir), str(out_large), 1024)
                resize_images_in_directory(str(input_dir), str(out_medium), 512)
            sys.exit(0)

    parser.add_argument(
        "--dataset",
        choices=AWARE_DATASETS + ["ALL"],
        action=ProcessAwareOrAll,
        help="Process one of the predefined AWARE datasets or ALL (creates 'large' 1024px and 'medium' 512px outputs) and exit",
    )
    parser.add_argument('--input_dir', help='Input directory containing images and subdirectories')
    parser.add_argument('--output_dir', help='Output directory for resized images')
    parser.add_argument('--size', '-s', type=int, default=1024, 
                       help='Target size for short side in pixels (default: 1024)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actually resizing')
    parser.add_argument('--labels_dir', type=str, default=None,
                        help='Optional labels directory to resize (semantic segmentation labels, nearest neighbor)')
    parser.add_argument('--labels_output_dir', type=str, default=None,
                        help='Output directory for resized labels (defaults to output_dir if not set)')
    parser.add_argument('--labels-all-datasets-root', type=str, default=None,
                        help='If set, traverse all immediate subdirectories and resize labels folders into <dataset>/labels_resized')

    args = parser.parse_args()

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target short side: {args.size} pixels")
    print(f"Dry run: {args.dry_run}")
    print("-" * 50)

    if args.dry_run and args.input_dir:
        # Just show what would be processed
        input_path = Path(args.input_dir)
        count = 0
        for root, dirs, files in os.walk(input_path):
            for filename in files:
                if is_image_file(filename):
                    print(f"Would process: {os.path.join(root, filename)}")
                    count += 1
        print(f"\nTotal images that would be processed: {count}")
    else:
        if args.input_dir and args.output_dir:
            resize_images_in_directory(args.input_dir, args.output_dir, args.size)
        if args.labels_dir:
            labels_out = args.labels_output_dir or args.output_dir
            if not labels_out:
                print("Error: --labels_dir provided but no --output_dir or --labels_output_dir specified.")
            else:
                resize_labels_in_directory(args.labels_dir, labels_out, args.size)
        if args.labels_all_datasets_root:
            resize_labels_for_all_datasets(
                args.labels_all_datasets_root,
                output_folder_name='labels_resized',
                target_short_side=args.size,
                dry_run=args.dry_run
            )

if __name__ == "__main__":
    main()