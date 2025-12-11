#!/usr/bin/env python3
"""
Dataset Color Stripe Generator - Final Version
=============================================

This script creates color stripe visualizations similar to "The Colors of Motion" 
but for image datasets instead of movies. Each image in the dataset is represented 
by a vertical stripe showing either:

1. The average color of the image
2. The most commonly occurring colors (using K-means clustering)

Updated to handle large datasets and JSON serialization issues.

Author: Generated for image dataset analysis
Inspired by: https://www.thecolorsofmotion.com/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from pathlib import Path
import argparse
import warnings
from typing import List, Tuple, Union
import json
import math

warnings.filterwarnings('ignore')

# PIL maximum image dimension (2^16 - 1)
PIL_MAX_DIMENSION = 65535


class DatasetColorExtractor:
    """
    A class to extract colors from image datasets and create vertical stripe visualizations.
    Updated to handle large datasets and JSON serialization issues.
    """

    def __init__(self, resize_dim: int = 256, verbose: bool = True):
        """
        Initialize the color extractor.

        Args:
            resize_dim (int): Dimension to resize images for faster processing (default: 256)
            verbose (bool): Whether to print progress messages (default: True)
        """
        self.resize_dim = resize_dim
        self.verbose = verbose

    def get_average_color(self, image_path: Union[str, Path]) -> Tuple[int, int, int]:
        """Extract the average color from an image."""
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img.thumbnail((self.resize_dim, self.resize_dim), Image.Resampling.LANCZOS)
                img_array = np.array(img)
                avg_color = np.mean(img_array, axis=(0, 1))
                # Convert numpy types to regular Python int for JSON serialization
                return tuple(int(c) for c in avg_color)
        except Exception as e:
            if self.verbose:
                print(f"Error processing {image_path}: {e}")
            return (128, 128, 128)

    def get_dominant_colors(self, image_path: Union[str, Path], 
                          n_colors: int = 3) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from an image using K-means clustering."""
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img.thumbnail((self.resize_dim, self.resize_dim), Image.Resampling.LANCZOS)
                img_array = np.array(img)
                pixels = img_array.reshape(-1, 3)

                if len(pixels) < n_colors:
                    if self.verbose:
                        print(f"Warning: {image_path} has fewer pixels than requested colors")
                    avg_color = tuple(int(c) for c in np.mean(pixels, axis=0))
                    return [avg_color] * n_colors

                kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
                kmeans.fit(pixels)
                colors = kmeans.cluster_centers_
                labels = kmeans.labels_
                label_counts = np.bincount(labels)
                sorted_indices = np.argsort(label_counts)[::-1]
                sorted_colors = colors[sorted_indices]

                # Convert numpy types to regular Python int for JSON serialization
                return [tuple(int(c) for c in color) for color in sorted_colors]
        except Exception as e:
            if self.verbose:
                print(f"Error processing {image_path}: {e}")
            return [(128, 128, 128)] * n_colors

    def process_dataset(self, dataset_path: Union[str, Path], 
                       method: str = 'average', 
                       n_colors: int = 3, 
                       max_images: int = None,
                       recursive: bool = True) -> Tuple[List[List[Tuple]], List[str]]:
        """Process all images in a dataset directory."""
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = []

        pattern = '**/*' if recursive else '*'
        for file_path in dataset_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)

        if not image_files:
            raise ValueError(f"No images found in {dataset_path}")

        image_files.sort()

        if max_images and max_images > 0:
            image_files = image_files[:max_images]

        if self.verbose:
            print(f"Found {len(image_files)} images")
            print(f"Processing with {method} color extraction...")

        colors = []

        for i, image_path in enumerate(image_files):
            if self.verbose and (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images...")

            if method == 'average':
                color = self.get_average_color(image_path)
                colors.append([color])
            elif method == 'dominant':
                dominant_colors = self.get_dominant_colors(image_path, n_colors)
                colors.append(dominant_colors)
            else:
                raise ValueError("Method must be 'average' or 'dominant'")

        if self.verbose:
            print(f"Completed processing {len(image_files)} images")

        filenames = [str(f.relative_to(dataset_path)) for f in image_files]
        return colors, filenames

    def _calculate_optimal_stripe_width(self, num_images: int, max_width: int = PIL_MAX_DIMENSION) -> int:
        """Calculate optimal stripe width to stay within PIL limits."""
        max_stripe_width = max_width // num_images
        return max(1, max_stripe_width)

    def _should_create_chunked_output(self, num_images: int, stripe_width: int) -> bool:
        """Determine if output should be chunked due to size constraints."""
        total_width = num_images * stripe_width
        return total_width > PIL_MAX_DIMENSION

    def create_stripe_visualization(self, 
                                   colors: List[List[Tuple]], 
                                   filenames: List[str] = None,
                                   output_path: str = 'dataset_colors.png', 
                                   stripe_width: int = 5, 
                                   height: int = 600, 
                                   title: str = "Dataset Color Visualization",
                                   save_metadata: bool = True,
                                   auto_adjust_width: bool = True,
                                   chunk_size: int = None) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Create a vertical stripe visualization from extracted colors.
        Updated to handle large datasets that exceed PIL's image size limits.
        """

        # Build new output file name in the same directory
        out_parent = Path(output_path).parent
        out_stem = Path(output_path).stem
        output_path = str(Path(output_path).with_suffix('.png'))
        if not colors:
            raise ValueError("No colors provided")

        num_images = len(colors)
        total_width = num_images * stripe_width

        # Check if we need to handle large dataset
        if total_width > PIL_MAX_DIMENSION:
            if auto_adjust_width:
                # Try to adjust stripe width first
                optimal_width = self._calculate_optimal_stripe_width(num_images)
                if optimal_width >= 1:
                    if self.verbose:
                        print(f"Warning: Output width ({total_width}) exceeds PIL limit ({PIL_MAX_DIMENSION})")
                        print(f"Auto-adjusting stripe width from {stripe_width} to {optimal_width}")
                    stripe_width = optimal_width
                    total_width = num_images * stripe_width

            # If still too large, create chunked output
            if total_width > PIL_MAX_DIMENSION:
                return self._create_chunked_visualization(
                    colors, filenames, output_path, stripe_width, height, 
                    title, save_metadata, chunk_size
                )

        # Create single visualization (fits within PIL limits)
        return self._create_single_visualization(
            colors, filenames, output_path, stripe_width, height, 
            title, save_metadata
        )

    def _create_single_visualization(self, colors, filenames, output_path, 
                                   stripe_width, height, title, save_metadata):
        """Create a single visualization image."""
        num_images = len(colors)
        single_color_mode = len(colors[0]) == 1
        total_width = num_images * stripe_width

        if single_color_mode:
            img_array = np.zeros((height, total_width, 3), dtype=np.uint8)
            for i, color_list in enumerate(colors):
                color = color_list[0]
                start_x = i * stripe_width
                end_x = start_x + stripe_width
                img_array[:, start_x:end_x] = color
        else:
            n_colors_per_image = len(colors[0])
            color_height = height // n_colors_per_image
            img_array = np.zeros((height, total_width, 3), dtype=np.uint8)

            for i, color_list in enumerate(colors):
                start_x = i * stripe_width
                end_x = start_x + stripe_width
                for j, color in enumerate(color_list):
                    start_y = j * color_height
                    end_y = min(start_y + color_height, height)
                    img_array[start_y:end_y, start_x:end_x] = color

        # Save visualization
        result_img = Image.fromarray(img_array)
        result_img.save(output_path, 'PNG', optimize=True)

        # Create matplotlib version
        self._create_matplotlib_version(img_array, output_path, title, 
                                      num_images, stripe_width, single_color_mode, colors)

        # Save metadata
        if save_metadata:
            self._save_metadata(colors, filenames, output_path, stripe_width, 
                              height, single_color_mode)

        if self.verbose:
            print(f"Single visualization saved: {output_path}")
            print(f"Dimensions: {total_width}x{height} pixels")

        return img_array

    def _create_chunked_visualization(self, colors, filenames, output_path, 
                                    stripe_width, height, title, save_metadata, chunk_size):
        """Create multiple visualization chunks for very large datasets."""
        num_images = len(colors)

        # Calculate chunk size if not provided
        if chunk_size is None:
            max_images_per_chunk = PIL_MAX_DIMENSION // stripe_width
            chunk_size = min(max_images_per_chunk, 1000)  # Cap at 1000 for manageable files

        num_chunks = math.ceil(num_images / chunk_size)

        if self.verbose:
            print(f"Creating {num_chunks} chunks with up to {chunk_size} images each")
            print(f"Total images: {num_images}, Stripe width: {stripe_width}")

        chunk_arrays = []
        base_name = Path(output_path).stem
        base_dir = Path(output_path).parent

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_images)

            chunk_colors = colors[start_idx:end_idx]
            chunk_filenames = filenames[start_idx:end_idx] if filenames else None

            # Create chunk output path
            chunk_output = base_dir / f"{base_name}_chunk_{chunk_idx+1:03d}.png"
            chunk_title = f"{title} - Part {chunk_idx+1}/{num_chunks}"

            # Create single chunk
            chunk_array = self._create_single_visualization(
                chunk_colors, chunk_filenames, str(chunk_output),
                stripe_width, height, chunk_title, False  # Don't save metadata for chunks
            )

            chunk_arrays.append(chunk_array)

            if self.verbose:
                print(f"Created chunk {chunk_idx+1}/{num_chunks}: {chunk_output}")

        # Save combined metadata
        if save_metadata:
            self._save_metadata(colors, filenames, output_path, stripe_width, 
                              height, len(colors[0]) == 1, is_chunked=True, 
                              num_chunks=num_chunks)

        # Create overview/summary image
        self._create_overview_image(chunk_arrays, colors, output_path, title, 
                                  stripe_width, height)

        if self.verbose:
            print(f"Chunked visualization complete: {num_chunks} files created")
            print(f"Overview image: {base_name}_overview.png")

        return chunk_arrays

    def _create_overview_image(self, chunk_arrays, colors, output_path, title, 
                             stripe_width, height):
        """Create a downsampled overview of the entire dataset."""
        base_name = Path(output_path).stem
        base_dir = Path(output_path).parent
        overview_path = base_dir / f"{base_name}_overview.png"

        # Downsample the dataset for overview
        num_images = len(colors)
        target_width = min(2000, PIL_MAX_DIMENSION // 2)  # Target overview width
        downsample_factor = max(1, num_images // target_width)

        downsampled_colors = colors[::downsample_factor]
        downsampled_stripe_width = max(1, stripe_width)

        if self.verbose:
            print(f"Creating overview with {len(downsampled_colors)} images "
                  f"(downsampling factor: {downsample_factor})")

        # Create overview visualization
        overview_array = self._create_single_visualization(
            downsampled_colors, None, str(overview_path),
            downsampled_stripe_width, height // 2,  # Smaller height for overview
            f"{title} - Overview ({len(downsampled_colors)} samples)", False
        )

        return overview_array

    def _create_matplotlib_version(self, img_array, output_path, title, 
                                 num_images, stripe_width, single_color_mode, colors):
        """Create matplotlib version with title and statistics."""
        plt.style.use('default')
        fig_width = max(12, min(20, img_array.shape[1] / 100))  # Cap figure width
        fig, ax = plt.subplots(figsize=(fig_width, 8))

        ax.imshow(img_array, aspect='auto')
        ax.set_title(title, fontsize=16, pad=20, weight='bold')
        ax.axis('off')

        # Add statistics
        if single_color_mode:
            method_text = "Average Color"
        else:
            n_colors_per_image = len(colors[0]) if colors else 0
            method_text = f"K-means ({n_colors_per_image} colors)"

        stats_text = f"Images: {num_images} | Method: {method_text} | Stripe Width: {stripe_width}px"
        ax.text(0.5, -0.05, stats_text, transform=ax.transAxes, 
                ha='center', va='top', fontsize=10, style='italic')

        plt.tight_layout()

        base_name = Path(output_path).stem
        plt_output = str(Path(output_path).parent / f"{base_name}_with_title.png")
        plt.savefig(plt_output, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

    def _save_metadata(self, colors, filenames, output_path, stripe_width, 
                      height, single_color_mode, is_chunked=False, num_chunks=1):
        """Save metadata JSON file with proper type conversion."""
        base_name = Path(output_path).stem
        metadata_path = str(Path(output_path).parent / f"{base_name}_metadata.json")

        # Ensure all color values are regular Python ints (not numpy types)
        def convert_colors(color_list):
            converted = []
            for color_sublist in color_list:
                converted_sublist = []
                for color in color_sublist:
                    # Convert each RGB tuple to regular Python ints
                    rgb_tuple = tuple(int(c) for c in color)
                    hex_color = f"#{rgb_tuple[0]:02x}{rgb_tuple[1]:02x}{rgb_tuple[2]:02x}"
                    converted_sublist.append({
                        'rgb': list(rgb_tuple),  # Convert tuple to list for JSON
                        'hex': hex_color
                    })
                converted.append(converted_sublist)
            return converted

        metadata = {
            'total_images': int(len(colors)),
            'method': 'average' if single_color_mode else 'dominant',
            'colors_per_image': 1 if single_color_mode else (len(colors[0]) if colors else 0),
            'stripe_width': int(stripe_width),
            'height': int(height),
            'is_chunked': bool(is_chunked),
            'num_chunks': int(num_chunks),
            'image_filenames': filenames or [f"image_{i:04d}" for i in range(len(colors))],
            'colors': convert_colors(colors)
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        if self.verbose:
            print(f"Metadata saved: {metadata_path}")


def main():
    """Main function with enhanced command-line interface for large datasets."""
    parser = argparse.ArgumentParser(
        description='Generate color stripe visualizations from image datasets (handles large datasets)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/images --method average
  %(prog)s /path/to/images --method dominant --n_colors 5 --auto_adjust
  %(prog)s /path/to/images --chunk_size 500 --stripe_width 2
        """
    )

    parser.add_argument('dataset_path', help='Path to the dataset directory')
    parser.add_argument('--method', choices=['average', 'dominant'], default='average')
    parser.add_argument('--n_colors', type=int, default=3)
    parser.add_argument('--stripe_width', type=int, default=5)
    parser.add_argument('--height', type=int, default=600)
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--output', default='output_colors/')
    parser.add_argument('--resize_dim', type=int, default=256)
    parser.add_argument('--no_recursive', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--no_metadata', action='store_true')
    parser.add_argument('--no_auto_adjust', action='store_true', 
                        help='Disable automatic stripe width adjustment')
    parser.add_argument('--chunk_size', type=int, default=None,
                        help='Number of images per chunk for large datasets')
    parser.add_argument('--loop_dir', action='store_true', help="Loop through subdirectories in source folder")

    args = parser.parse_args()

    extractor = DatasetColorExtractor(resize_dim=args.resize_dim, verbose=not args.quiet)

    if args.loop_dir:
        for subdir in Path(args.dataset_path).iterdir():
            if subdir.is_dir():
                print(f"\nProcessing subdirectory: {subdir}")
                try:
                    colors, filenames = extractor.process_dataset(
                        subdir, 
                        method=args.method, 
                        n_colors=args.n_colors,
                        max_images=args.max_images,
                        recursive=not args.no_recursive
                    )

                    title = f"Dataset Color Visualization - {args.method.title()} Method"
                    if args.method == 'dominant':
                        title += f" ({args.n_colors} colors per image)"

                    dataset_name = subdir.name or 'dataset'
                    output_subdir = os.path.join(args.output, dataset_name)
                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)
                    output_path = os.path.join(output_subdir, f"{dataset_name}_{args.method}.png")

                    result = extractor.create_stripe_visualization(
                        colors, 
                        filenames, 
                        output_path,
                        stripe_width=args.stripe_width,
                        height=args.height,
                        title=title,
                        save_metadata=not args.no_metadata,
                        auto_adjust_width=not args.no_auto_adjust,
                        chunk_size=args.chunk_size
                    )

                    if not args.quiet:
                        if isinstance(result, list):
                            print(f"\nLarge dataset processed as {len(result)} chunks")
                        else:
                            print(f"\nVisualization complete! Processed {len(colors)} images")

                except Exception as e:
                    print(f"Error processing {subdir}: {e}", file=sys.stderr)

    try:
        colors, filenames = extractor.process_dataset(
            args.dataset_path, 
            method=args.method, 
            n_colors=args.n_colors,
            max_images=args.max_images,
            recursive=not args.no_recursive
        )

        title = f"Dataset Color Visualization - {args.method.title()} Method"
        if args.method == 'dominant':
            title += f" ({args.n_colors} colors per image)"

        dataset_name = Path(args.dataset_path).name or 'dataset'
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        output_path = os.path.join(args.output, f"{dataset_name}_{args.method}.png")

        result = extractor.create_stripe_visualization(
            colors, 
            filenames, 
            output_path,
            stripe_width=args.stripe_width,
            height=args.height,
            title=title,
            save_metadata=not args.no_metadata,
            auto_adjust_width=not args.no_auto_adjust,
            chunk_size=args.chunk_size
        )

        if not args.quiet:
            if isinstance(result, list):
                print(f"\nLarge dataset processed as {len(result)} chunks")
            else:
                print(f"\nVisualization complete! Processed {len(colors)} images")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
