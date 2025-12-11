#!/usr/bin/env python3
"""
Advanced Dataset Splitter for Robust Reproducible Evaluation

This script splits weather/lighting condition image benchmarks for semantic 
segmentation/object detection tasks into train and test sets with:
- Statistically meaningful test set sample sizes
- Intelligent, condition-adapted split ratios
- Full reproducibility via random seeds
- Comprehensive logging and statistics

Author: Generated for Game Development Course
Date: 2025-11-06
"""

import os
import json
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetSplitter:
    """
    Handles splitting of hierarchical weather/lighting condition datasets.

    Attributes:
        input_root (Path): Root directory containing datasets
        output_root (Path): Root directory for split outputs
        train_ratio (float): Default train/test split ratio
        random_seed (int): Seed for reproducibility
        datasets (List[str]): List of datasets to process
        generate_lists (bool): Whether to generate image lists
        min_test_samples (int): Minimum recommended test samples per category
        ideal_test_samples (int): Ideal test samples per category
    """

    def __init__(
        self,
        input_root: str,
        output_root: str,
        train_ratio: float = 0.7,
        random_seed: int = 42,
        datasets: Optional[List[str]] = None,
        generate_lists: bool = False,
        min_test_samples: int = 50,
        ideal_test_samples: int = 100
    ):
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.datasets = datasets or []
        self.generate_lists = generate_lists
        self.min_test_samples = min_test_samples
        self.ideal_test_samples = ideal_test_samples

        # Set random seed for reproducibility
        random.seed(self.random_seed)

        # Statistics storage
        self.statistics = {
            'config': {
                'random_seed': self.random_seed,
                'train_ratio': self.train_ratio,
                'min_test_samples': self.min_test_samples,
                'ideal_test_samples': self.ideal_test_samples
            },
            'datasets': {},
            'summary': {}
        }

        # Image lists for reproducibility
        self.image_lists = {
            'train': defaultdict(lambda: defaultdict(list)),
            'test': defaultdict(lambda: defaultdict(list))
        }

        logger.info(f"Initialized DatasetSplitter with seed={random_seed}, "
                   f"train_ratio={train_ratio}")

    def compute_adaptive_split(
        self, 
        total_images: int, 
        category: str
    ) -> Tuple[int, int, float]:
        """
        Compute adaptive train/test split based on available data.

        Strategy:
        - For abundant categories (>500 images): use default ratio (70/30 or 80/20)
        - For moderate categories (200-500): prioritize test set (60/40)
        - For scarce categories (100-200): maximize test coverage (50/50 or 55/45)
        - For very scarce (<100): use most for test (40/60 up to 45/55)

        Args:
            total_images: Total number of images in category
            category: Category name for logging

        Returns:
            Tuple of (train_count, test_count, actual_train_ratio)
        """
        if total_images >= 500:
            # Abundant data: use default split
            train_count = int(total_images * self.train_ratio)
            test_count = total_images - train_count
            actual_ratio = self.train_ratio

        elif total_images >= 200:
            # Moderate data: prioritize larger test set
            train_ratio = 0.6
            train_count = int(total_images * train_ratio)
            test_count = total_images - train_count
            actual_ratio = train_ratio

        elif total_images >= 100:
            # Scarce data: balanced or test-heavy split
            # Aim for 50 test samples minimum
            test_count = max(self.min_test_samples, int(total_images * 0.5))
            test_count = min(test_count, total_images)  # Cap at total
            train_count = total_images - test_count
            actual_ratio = train_count / total_images if total_images > 0 else 0

        else:
            # Very scarce: maximize test set while keeping some training data
            # Try to get at least 40% for training (60% test)
            # But ensure we have training data
            test_count = min(
                int(total_images * 0.6), 
                total_images - max(10, int(total_images * 0.4))
            )
            test_count = max(test_count, total_images // 2)  # At least 50% to test
            train_count = total_images - test_count
            actual_ratio = train_count / total_images if total_images > 0 else 0

        # Log if test set is below recommendations
        if test_count < self.min_test_samples:
            logger.warning(
                f"Category '{category}': test set ({test_count}) below minimum "
                f"recommended ({self.min_test_samples}). This may limit statistical "
                f"significance."
            )
        elif test_count < self.ideal_test_samples:
            logger.info(
                f"Category '{category}': test set ({test_count}) below ideal "
                f"({self.ideal_test_samples}) but above minimum."
            )

        return train_count, test_count, actual_ratio

    def find_image_label_pairs(
        self,
        dataset_path: Path,
        dataset_name: str,
        category: str
    ) -> List[Tuple[Path, Path]]:
        """
        Find matching image-label pairs for a given category.

        Args:
            dataset_path: Path to dataset directory
            category: Category name

        Returns:
            List of (image_path, label_path) tuples
        """
        image_dir = dataset_path / "images" / category
        label_dir = dataset_path / "labels" 

        if not image_dir.exists():
            logger.warning(f"Image directory not found: {image_dir}")
            return []

        if not label_dir.exists():
            logger.warning(f"Label directory not found: {label_dir}")
            return []

        # Find all images
        image_extensions = ['.jpg', '.png', '.jpeg']
        images = []
        for ext in image_extensions:
            images.extend(image_dir.glob(f"*{ext}"))

        # Dataset-specific label matching
        pairs = []
        ds = dataset_name.strip().lower()

        def match_label_acdc(img: Path) -> Optional[Path]:
            stem = img.stem
            if stem.endswith('_rgb_ref_anon'):
                lbl_stem = stem[:-len('_rgb_ref_anon')] + '_gt_labelIds'
                cand = label_dir / f"{lbl_stem}.png"
                if cand.exists():
                    return cand
            if stem.endswith('_rgb_anon'):
                lbl_stem = stem[:-len('_rgb_anon')] + '_gt_labelIds'
                cand = label_dir / f"{lbl_stem}.png"
                if cand.exists():
                    return cand                
            # fallback
            cand = label_dir / f"{stem}.png"
            return cand if cand.exists() else None

        def match_label_bdd10k(img: Path) -> Optional[Path]:
            stem = img.stem
            cand = label_dir / f"{stem}_train_id.png"
            if cand.exists():
                return cand
            cand2 = label_dir / f"{stem}.png"
            return cand2 if cand2.exists() else None

        def match_label_idd_aw(img: Path) -> Optional[Path]:
            """IDD-AW: map '<id>_leftImg8bit.png' -> '<id>_gtFine_polygons.png'"""
            stem = img.stem  # e.g., '993075_leftImg8bit'
            suffix = '_leftImg8bit'
            if stem.endswith(suffix):
                base = stem[: -len(suffix)]
                cand = label_dir / f"{base}_gtFine_polygons.png"
                if cand.exists():
                    return cand
            # Fallbacks
            cand2 = label_dir / f"{stem}.png"
            return cand2 if cand2.exists() else None

        def match_label_default(img: Path) -> Optional[Path]:
            for ext in ['.png', '.json']:
                cand = label_dir / f"{img.stem}{ext}"
                if cand.exists():
                    return cand
            return None

        for img_path in images:
            label_path: Optional[Path] = None
            if ds == 'acdc':
                label_path = match_label_acdc(img_path)
            elif ds == 'bdd10k':
                label_path = match_label_bdd10k(img_path)
            elif ds == 'idd-aw':
                label_path = match_label_idd_aw(img_path)
            elif ds == 'bdd100k':
                # use combined annotations loaded in process_dataset
                lookup = getattr(self, 'bdd100k_lookup', {})
                ann = lookup.get(img_path.name)
                if ann is not None:
                    # virtual path for generated JSON
                    label_path = label_dir / f"{img_path.stem}.json"
                    if not hasattr(self, 'generated_label_contents'):
                        self.generated_label_contents = {}
                    self.generated_label_contents[str(label_path)] = ann
            else:
                label_path = match_label_default(img_path)

            if label_path is not None:
                pairs.append((img_path, label_path))
            else:
                logger.debug(f"No label found for image: {img_path}")

        return pairs

    def split_category(
        self,
        dataset_name: str,
        category: str,
        pairs: List[Tuple[Path, Path]]
    ) -> Dict:
        """
        Split a single category into train/test sets.

        Args:
            dataset_name: Name of the dataset
            category: Category name
            pairs: List of (image, label) path tuples

        Returns:
            Dictionary with split statistics
        """
        total = len(pairs)

        if total == 0:
            logger.warning(f"{dataset_name}/{category}: No image-label pairs found")
            return {
                'total': 0,
                'train': 0,
                'test': 0,
                'train_ratio': 0.0,
                'status': 'empty'
            }

        # Compute adaptive split
        train_count, test_count, actual_ratio = self.compute_adaptive_split(
            total, f"{dataset_name}/{category}"
        )

        # Shuffle pairs for random split
        shuffled_pairs = pairs.copy()
        random.shuffle(shuffled_pairs)

        # Split
        train_pairs = shuffled_pairs[:train_count]
        test_pairs = shuffled_pairs[train_count:]

        # Copy files to output directories
        for split_name, split_pairs in [('train', train_pairs), ('test', test_pairs)]:
            for img_path, label_path in split_pairs:
                # Create output directories
                out_img_dir = (self.output_root / split_name / 'images' / 
                              dataset_name / category)
                out_label_dir = (self.output_root / split_name / 'labels' / 
                                dataset_name / category)

                out_img_dir.mkdir(parents=True, exist_ok=True)
                out_label_dir.mkdir(parents=True, exist_ok=True)

                # Copy image (preserve image filename)
                shutil.copy2(img_path, out_img_dir / img_path.name)

                # Adapt label output name to match image stem exactly
                if label_path.suffix.lower() == '.json':
                    out_label_name = f"{img_path.stem}.json"
                else:
                    out_label_name = f"{img_path.stem}.png"
                dest_label_path = out_label_dir / out_label_name

                if label_path.exists():
                    shutil.copy2(label_path, dest_label_path)
                else:
                    # Handle generated BDD100k annotation content
                    gen_map = getattr(self, 'generated_label_contents', {})
                    content = gen_map.get(str(label_path))
                    if content is not None:
                        with open(dest_label_path, 'w') as f:
                            json.dump(content, f, indent=2)
                    else:
                        logger.warning(f"Missing label for {img_path.name}: expected {label_path}")

                # Record in image lists
                if self.generate_lists:
                    self.image_lists[split_name][dataset_name][category].append({
                        'image': img_path.name,
                        'label': dest_label_path.name
                    })

        # Compile statistics
        stats = {
            'total': total,
            'train': train_count,
            'test': test_count,
            'train_ratio': actual_ratio,
            'test_ratio': 1 - actual_ratio,
            'status': 'success'
        }

        # Add warnings if test set is suboptimal
        if test_count < self.min_test_samples:
            stats['warning'] = f'test_set_below_minimum ({test_count} < {self.min_test_samples})'
        elif test_count < self.ideal_test_samples:
            stats['note'] = f'test_set_below_ideal ({test_count} < {self.ideal_test_samples})'

        logger.info(
            f"{dataset_name}/{category}: {total} total -> "
            f"{train_count} train, {test_count} test "
            f"(ratio: {actual_ratio:.2f}/{1-actual_ratio:.2f})"
        )

        return stats

    def process_dataset(self, dataset_name: str) -> Dict:
        """
        Process all categories in a dataset.

        Args:
            dataset_name: Name of the dataset to process

        Returns:
            Dictionary with dataset statistics
        """
        dataset_path = self.input_root / dataset_name

        if not dataset_path.exists():
            logger.error(f"Dataset path not found: {dataset_path}")
            return {'status': 'not_found'}

        logger.info(f"Processing dataset: {dataset_name}")

        # Find all categories
        images_dir = dataset_path / "images"
        if not images_dir.exists():
            logger.error(f"Images directory not found: {images_dir}")
            return {'status': 'invalid_structure'}

        categories = [d.name for d in images_dir.iterdir() if d.is_dir()]

        dataset_stats = {
            'categories': {},
            'total_images': 0,
            'total_train': 0,
            'total_test': 0
        }

        # Preload BDD100k annotations if needed
        if dataset_name.strip().lower() == 'bdd100k':
            labels_dir = dataset_path / 'labels'
            lookup = {}
            if labels_dir.exists():
                for fname in ['bdd100k_labels_images_train.json', 'bdd100k_labels_images_val.json', 'bdd100k_labels_images_test.json']:
                    fpath = labels_dir / fname
                    if fpath.exists():
                        try:
                            with open(fpath, 'r') as f:
                                data = json.load(f)
                            for entry in data:
                                name = entry.get('name')
                                if name:
                                    lookup[name] = entry
                            logger.info(f"Loaded {fname} ({len(data)} entries)")
                        except Exception as e:
                            logger.warning(f"Failed reading {fname}: {e}")
            self.bdd100k_lookup = lookup

        # Process each category
        for category in categories:
            pairs = self.find_image_label_pairs(dataset_path, dataset_name, category)
            cat_stats = self.split_category(dataset_name, category, pairs)

            dataset_stats['categories'][category] = cat_stats
            dataset_stats['total_images'] += cat_stats.get('total', 0)
            dataset_stats['total_train'] += cat_stats.get('train', 0)
            dataset_stats['total_test'] += cat_stats.get('test', 0)

        # Compute overall ratio
        if dataset_stats['total_images'] > 0:
            dataset_stats['overall_train_ratio'] = (
                dataset_stats['total_train'] / dataset_stats['total_images']
            )
        else:
            dataset_stats['overall_train_ratio'] = 0.0

        logger.info(
            f"Dataset {dataset_name} complete: {dataset_stats['total_images']} images, "
            f"{dataset_stats['total_train']} train, {dataset_stats['total_test']} test"
        )

        return dataset_stats

    def run(self):
        """Execute the dataset splitting process."""
        logger.info("=" * 70)
        logger.info("Starting dataset splitting process")
        logger.info("=" * 70)

        # Process each dataset
        for dataset in self.datasets:
            dataset_stats = self.process_dataset(dataset)
            self.statistics['datasets'][dataset] = dataset_stats

        # Compute summary statistics
        self.compute_summary()

        # Save statistics
        self.save_statistics()

        # Save image lists if requested
        if self.generate_lists:
            self.save_image_lists()

        logger.info("=" * 70)
        logger.info("Dataset splitting complete!")
        logger.info("=" * 70)

    def compute_summary(self):
        """Compute summary statistics across all datasets."""
        summary = {
            'total_datasets': len(self.datasets),
            'total_images': 0,
            'total_train': 0,
            'total_test': 0,
            'categories': defaultdict(lambda: {'total': 0, 'train': 0, 'test': 0})
        }

        for dataset_name, dataset_stats in self.statistics['datasets'].items():
            if 'total_images' in dataset_stats:
                summary['total_images'] += dataset_stats['total_images']
                summary['total_train'] += dataset_stats['total_train']
                summary['total_test'] += dataset_stats['total_test']

                # Aggregate by category
                for cat, cat_stats in dataset_stats.get('categories', {}).items():
                    summary['categories'][cat]['total'] += cat_stats.get('total', 0)
                    summary['categories'][cat]['train'] += cat_stats.get('train', 0)
                    summary['categories'][cat]['test'] += cat_stats.get('test', 0)

        # Compute overall ratio
        if summary['total_images'] > 0:
            summary['overall_train_ratio'] = summary['total_train'] / summary['total_images']
            summary['overall_test_ratio'] = summary['total_test'] / summary['total_images']

        # Convert categories to regular dict for JSON serialization
        summary['categories'] = dict(summary['categories'])

        self.statistics['summary'] = summary

        logger.info(f"Summary: {summary['total_images']} total images across "
                   f"{summary['total_datasets']} datasets")

    def save_statistics(self):
        """Save split statistics to JSON file."""
        stats_file = self.output_root / 'split_statistics.json'

        with open(stats_file, 'w') as f:
            json.dump(self.statistics, f, indent=2)

        logger.info(f"Statistics saved to: {stats_file}")

    def save_image_lists(self):
        """Save image lists for reproducibility."""
        for split_name in ['train', 'test']:
            list_file = self.output_root / f'{split_name}_image_list.json'

            # Convert defaultdict to regular dict
            split_data = {
                dataset: dict(categories)
                for dataset, categories in self.image_lists[split_name].items()
            }

            with open(list_file, 'w') as f:
                json.dump(split_data, f, indent=2)

            logger.info(f"Image list saved to: {list_file}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Advanced Dataset Splitter for Robust Reproducible Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python dataset_splitter.py \
      --input /media/chge7185/HDD1/repositories \
      --output ./splits \
      --train_ratio 0.7 \
      --datasets ACDC BDD10k MapillaryVistas \
      --generate_lists \
      --seed 42
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input root directory containing datasets'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output root directory for splits'
    )

    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Default train/test split ratio (default: 0.7). '
             'Note: actual ratios are adaptive per category.'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['ACDC', 'BDD100k', 'BDD10k', 'OUTSIDE15k', 'IDD-AW', 'MapillaryVistas'],
        help='List of datasets to process (default: all available)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--generate_lists',
        action='store_true',
        help='Generate image lists for reproducibility'
    )

    parser.add_argument(
        '--min_test_samples',
        type=int,
        default=50,
        help='Minimum recommended test samples per category (default: 50)'
    )

    parser.add_argument(
        '--ideal_test_samples',
        type=int,
        default=100,
        help='Ideal test samples per category (default: 100)'
    )

    args = parser.parse_args()

    # Create splitter and run
    splitter = DatasetSplitter(
        input_root=args.input,
        output_root=args.output,
        train_ratio=args.train_ratio,
        random_seed=args.seed,
        datasets=args.datasets,
        generate_lists=args.generate_lists,
        min_test_samples=args.min_test_samples,
        ideal_test_samples=args.ideal_test_samples
    )

    splitter.run()


if __name__ == '__main__':
    main()
