#!/usr/bin/env python3
"""
Perceptual Image Hash Search Tool

This script uses perceptual image hashing to find input image(s) in a large image dataset.
It supports multiple hash algorithms and optimization techniques for large-scale searches.

Features:
- Multiple hash algorithms (pHash, dHash, aHash, wHash)
- Optimized search using BK-trees for Hamming distance
- Parallel processing for hash computation
- Support for multiple input images
- Configurable similarity thresholds
- Progress tracking and detailed results

Requirements:
- pip install imagehash pillow numpy tqdm concurrent-futures pybktree
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
import json

try:
    import numpy as np
    from PIL import Image
    import imagehash
    from tqdm import tqdm
    import pybktree
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install imagehash pillow numpy tqdm pybktree")
    sys.exit(1)


class ImageHashSearcher:
    """
    A class for finding similar images in large datasets using perceptual hashing.
    """

    def __init__(self, hash_algorithm='dhash', hash_size=8, threshold=10, parallel=True):
        """
        Initialize the ImageHashSearcher.

        Args:
            hash_algorithm (str): Hash algorithm to use ('dhash', 'phash', 'ahash', 'whash')
            hash_size (int): Size of the hash (8 or 16 typically)
            threshold (int): Maximum Hamming distance for considering images similar
            parallel (bool): Whether to use parallel processing
        """
        self.hash_algorithm = hash_algorithm.lower()
        self.hash_size = hash_size
        self.threshold = threshold
        self.parallel = parallel
        self.hash_function = self._get_hash_function()

        # Storage for computed hashes
        self.image_hashes: Dict[str, imagehash.ImageHash] = {}
        self.bk_tree: Optional[pybktree.BKTree] = None

        # Supported image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}

    def _get_hash_function(self):
        """Get the hash function based on the algorithm name."""
        hash_functions = {
            'dhash': imagehash.dhash,
            'phash': imagehash.phash,  
            'ahash': imagehash.average_hash,
            'whash': imagehash.whash
        }

        if self.hash_algorithm not in hash_functions:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_algorithm}")

        return hash_functions[self.hash_algorithm]

    def _compute_hash(self, image_path: str) -> Optional[Tuple[str, imagehash.ImageHash]]:
        """
        Compute hash for a single image.

        Args:
            image_path (str): Path to the image file

        Returns:
            Tuple of (image_path, hash) or None if failed
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')

                # Use draft mode for faster loading when appropriate
                if hasattr(img, 'draft') and self.hash_size <= 32:
                    img.draft('L', (self.hash_size * 4, self.hash_size * 4))

                # Compute hash
                hash_value = self.hash_function(img, hash_size=self.hash_size)
                return (image_path, hash_value)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def compute_hashes(self, image_paths: List[str], show_progress: bool = True) -> Dict[str, imagehash.ImageHash]:
        """
        Compute hashes for a list of images.

        Args:
            image_paths (List[str]): List of image file paths
            show_progress (bool): Whether to show progress bar

        Returns:
            Dictionary mapping image paths to their hashes
        """
        print(f"Computing {self.hash_algorithm} hashes for {len(image_paths)} images...")

        if self.parallel and len(image_paths) > 100:
            # Use parallel processing for large datasets
            max_workers = min(cpu_count(), 8)  # Limit to avoid memory issues

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                if show_progress:
                    results = list(tqdm(
                        executor.map(self._compute_hash, image_paths),
                        total=len(image_paths),
                        desc="Computing hashes"
                    ))
                else:
                    results = list(executor.map(self._compute_hash, image_paths))
        else:
            # Sequential processing
            if show_progress:
                results = [self._compute_hash(path) for path in tqdm(image_paths, desc="Computing hashes")]
            else:
                results = [self._compute_hash(path) for path in image_paths]

        # Filter out failed computations and store results
        hashes = {}
        for result in results:
            if result is not None:
                path, hash_value = result
                hashes[path] = hash_value

        print(f"Successfully computed {len(hashes)} hashes")
        return hashes

    def build_search_index(self, hashes: Dict[str, imagehash.ImageHash]):
        """
        Build a BK-tree index for fast similarity search.

        Args:
            hashes (Dict[str, imagehash.ImageHash]): Dictionary of image hashes
        """
        print("Building search index...")

        # Convert hashes to integers for BK-tree
        def hash_to_int(hash_obj):
            """Convert ImageHash to integer for BK-tree storage."""
            return int(str(hash_obj), 16)

        def int_to_path(hash_int):
            """Convert hash integer back to file path."""
            hash_str = f"{hash_int:0{self.hash_size*self.hash_size//4}x}"
            for path, hash_obj in self.image_hashes.items():
                if str(hash_obj) == hash_str:
                    return path
            return None

        # Store hashes for later lookup
        self.image_hashes = hashes.copy()

        # Create BK-tree with hamming distance
        hash_ints = [hash_to_int(hash_obj) for hash_obj in hashes.values()]
        self.bk_tree = pybktree.BKTree(pybktree.hamming_distance, hash_ints)

        print("Search index built successfully")

    def find_similar_images(self, query_image_path: str) -> List[Tuple[str, int]]:
        """
        Find similar images to a query image.

        Args:
            query_image_path (str): Path to the query image

        Returns:
            List of tuples (image_path, hamming_distance) sorted by distance
        """
        if self.bk_tree is None:
            raise ValueError("Search index not built. Call build_search_index() first.")

        # Compute hash for query image
        query_result = self._compute_hash(query_image_path)
        if query_result is None:
            raise ValueError(f"Could not compute hash for query image: {query_image_path}")

        _, query_hash = query_result
        query_int = int(str(query_hash), 16)

        # Search in BK-tree
        matches = self.bk_tree.find(query_int, self.threshold)

        # Convert back to file paths and distances
        results = []
        for distance, hash_int in matches:
            # Find the file path corresponding to this hash
            hash_str = f"{hash_int:0{self.hash_size*self.hash_size//4}x}"
            for path, hash_obj in self.image_hashes.items():
                if str(hash_obj) == hash_str:
                    results.append((path, distance))
                    break

        # Sort by distance (most similar first)
        results.sort(key=lambda x: x[1])
        return results

    def find_similar_linear(self, query_image_path: str, target_hashes: Dict[str, imagehash.ImageHash]) -> List[Tuple[str, int]]:
        """
        Find similar images using linear search (for smaller datasets).

        Args:
            query_image_path (str): Path to the query image
            target_hashes (Dict[str, imagehash.ImageHash]): Target image hashes

        Returns:
            List of tuples (image_path, hamming_distance) sorted by distance
        """
        # Compute hash for query image
        query_result = self._compute_hash(query_image_path)
        if query_result is None:
            raise ValueError(f"Could not compute hash for query image: {query_image_path}")

        _, query_hash = query_result

        # Compare with all target hashes
        results = []
        for path, target_hash in target_hashes.items():
            distance = query_hash - target_hash  # Hamming distance
            if distance <= self.threshold:
                results.append((path, distance))

        # Sort by distance (most similar first)
        results.sort(key=lambda x: x[1])
        return results


def find_images_in_directory(directory: str, recursive: bool = True) -> List[str]:
    """
    Find all image files in a directory.

    Args:
        directory (str): Directory path to search
        recursive (bool): Whether to search recursively

    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    image_paths = []

    path_obj = Path(directory)
    if not path_obj.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    pattern = "**/*" if recursive else "*"

    for file_path in path_obj.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_paths.append(str(file_path))

    return sorted(image_paths)


def save_results(results: Dict[str, List[Tuple[str, int]]], output_file: str):
    """
    Save search results to a JSON file.

    Args:
        results (Dict): Dictionary of query -> matches
        output_file (str): Output file path
    """
    # Convert results to serializable format
    serializable_results = {}
    for query, matches in results.items():
        serializable_results[query] = [{"path": path, "distance": dist} for path, dist in matches]

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Find images in a large dataset using perceptual hashing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find similar images for a single query
  python image_hash_search.py /path/to/dataset --query /path/to/query.jpg

  # Use multiple query images
  python image_hash_search.py /path/to/dataset --query img1.jpg img2.jpg img3.jpg

  # Use pHash algorithm with higher threshold
  python image_hash_search.py /path/to/dataset --query query.jpg --algorithm phash --threshold 15

  # Save results to file
  python image_hash_search.py /path/to/dataset --query query.jpg --output results.json
        """
    )

    parser.add_argument('dataset_path', help='Path to the image dataset directory')
    parser.add_argument('--query', '-q', nargs='+', 
                        help='Path(s) to query image(s) or directories')
    parser.add_argument('--query-dir', '-Q', 
                        help='Path to a directory containing query images (will include all images inside)')
    parser.add_argument('--algorithm', '-a', default='dhash', 
                        choices=['dhash', 'phash', 'ahash', 'whash'],
                        help='Hash algorithm to use (default: dhash)')
    parser.add_argument('--hash-size', '-s', type=int, default=8,
                        help='Hash size (default: 8)')
    parser.add_argument('--threshold', '-t', type=int, default=10,
                        help='Maximum Hamming distance for similarity (default: 10)')
    parser.add_argument('--output', '-o', help='Output file for results (JSON format)')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing')
    parser.add_argument('--recursive', '-r', action='store_true', default=True,
                        help='Search dataset directory recursively (default: True)')
    parser.add_argument('--max-results', '-m', type=int, default=20,
                        help='Maximum number of results per query (default: 20)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path does not exist: {args.dataset_path}")
        return 1

    # Build list of query images from --query entries and/or --query-dir
    query_list: List[str] = []
    if args.query:
        for qp in args.query:
            if os.path.isdir(qp):
                # Expand directory to images
                try:
                    imgs = find_images_in_directory(qp, recursive=True)
                    if not imgs:
                        print(f"Warning: No images found in query directory: {qp}")
                    query_list.extend(imgs)
                except Exception as e:
                    print(f"Error reading query directory {qp}: {e}")
                    return 1
            elif os.path.exists(qp):
                query_list.append(qp)
            else:
                print(f"Error: Query path does not exist: {qp}")
                return 1

    if args.query_dir:
        if not os.path.isdir(args.query_dir):
            print(f"Error: --query-dir is not a directory: {args.query_dir}")
            return 1
        imgs = find_images_in_directory(args.query_dir, recursive=True)
        if not imgs:
            print(f"Warning: No images found in --query-dir: {args.query_dir}")
        query_list.extend(imgs)

    # Deduplicate and validate
    query_list = sorted(list(dict.fromkeys(query_list)))
    if not query_list:
        print("Error: No query images provided. Use --query and/or --query-dir.")
        return 1

    # Initialize searcher
    searcher = ImageHashSearcher(
        hash_algorithm=args.algorithm,
        hash_size=args.hash_size,
        threshold=args.threshold,
        parallel=not args.no_parallel
    )

    print(f"Image Hash Search Tool")
    print(f"Algorithm: {args.algorithm}")
    print(f"Hash size: {args.hash_size}")
    print(f"Threshold: {args.threshold}")
    print(f"Parallel processing: {not args.no_parallel}")
    print("-" * 50)

    # Find all images in dataset
    print(f"Scanning dataset directory: {args.dataset_path}")
    dataset_images = find_images_in_directory(args.dataset_path, args.recursive)
    print(f"Found {len(dataset_images)} images in dataset")

    if len(dataset_images) == 0:
        print("No images found in dataset. Exiting.")
        return 1

    # Compute hashes for dataset
    start_time = time.time()
    dataset_hashes = searcher.compute_hashes(dataset_images)

    # Choose search method based on dataset size
    use_bk_tree = len(dataset_hashes) > 1000

    if use_bk_tree:
        print(f"Large dataset detected. Building search index for faster lookup...")
        searcher.build_search_index(dataset_hashes)
    else:
        print(f"Using linear search for dataset of {len(dataset_hashes)} images")

    hash_time = time.time() - start_time
    print(f"Hash computation completed in {hash_time:.2f} seconds")
    print("-" * 50)

    # Process each query image
    all_results = {}

    for query_path in query_list:
        print(f"\nSearching for similar images to: {query_path}")

        try:
            if use_bk_tree:
                matches = searcher.find_similar_images(query_path)
            else:
                matches = searcher.find_similar_linear(query_path, dataset_hashes)

            # Limit results
            matches = matches[:args.max_results]

            if matches:
                print(f"Found {len(matches)} similar images:")
                for i, (match_path, distance) in enumerate(matches, 1):
                    similarity = max(0, 100 - (distance / args.threshold * 100))
                    print(f"  {i:2d}. {match_path} (distance: {distance}, similarity: {similarity:.1f}%)")

                all_results[query_path] = matches
            else:
                print("  No similar images found within threshold")
                all_results[query_path] = []

        except Exception as e:
            print(f"Error processing query {query_path}: {e}")
            all_results[query_path] = []

    # Save results if requested
    if args.output:
        save_results(all_results, args.output)

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

    return 0


if __name__ == "__main__":
    sys.exit(main())
