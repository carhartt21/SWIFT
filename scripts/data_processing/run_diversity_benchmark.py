#!/usr/bin/env python3
"""
Layout Diversity Benchmark Script

This script runs the layout diversity analysis on multiple segmentation datasets
and generates a comparison report.

Usage:
    python run_diversity_benchmark.py --output-dir /path/to/output

The script expects datasets to be located at /media/chge7185/HDD1/datasets/AWARE/
and /media/chge7185/HDD1/datasets/Cityscapes/. Modify the DATASETS dict below
to customize paths for your environment.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from layout_diversity_score import analyze_dataset, visualize_similarity_matrix, visualize_similarity_distribution


# Dataset configurations
# Modify paths here to match your environment
DATASETS = {
    "ACDC": {
        "path": "/media/chge7185/HDD1/datasets/AWARE/ACDC/labels/",
        "num_classes": 34,
        "pattern": "*.png",
        "resize": None,
        "description": "ACDC dataset (Cityscapes format, adverse conditions)"
    },
    "BDD10k": {
        "path": "/media/chge7185/HDD1/datasets/AWARE/BDD10k/labels/",
        "num_classes": 19,
        "pattern": "*.png",
        "resize": None,
        "description": "BDD100k subset (train IDs format)"
    },
    "Cityscapes": {
        "path": "/media/chge7185/HDD1/datasets/Cityscapes/gtFine_trainvaltest/gtFine/",
        "num_classes": 34,
        "pattern": "**/*_labelIds.png",  # Recursive pattern for train/val subdirs
        "resize": None,
        "description": "Cityscapes train+val (exclude test - placeholder annotations)",
        "exclude_pattern": "test"  # Exclude test set
    },
    "Mapillary": {
        "path": "/media/chge7185/HDD1/datasets/AWARE/MapillaryVistas/labels/",
        "num_classes": 66,
        "pattern": "*.png",
        "resize": (512, 1024),  # Variable sizes - needs resize
        "description": "Mapillary Vistas (resized to 512x1024)"
    },
    "OUTSIDE15k": {
        "path": "/media/chge7185/HDD1/datasets/AWARE/OUTSIDE15k/labels/",
        "num_classes": 24,
        "pattern": "*.png",
        "resize": (512, 1024),  # Variable sizes - needs resize
        "description": "OUTSIDE15k composite dataset (resized to 512x1024)"
    },
    "IDD": {
        "path": "/media/chge7185/HDD1/datasets/AWARE/IDD/labels/",
        "num_classes": 27,
        "pattern": "*.png",
        "resize": (720, 1280),  # Variable sizes - needs resize
        "description": "IDD India Driving Dataset (resized to 720x1280)"
    },
}


def filter_files_by_split(files, exclude_pattern):
    """Filter out files matching exclude pattern (e.g., 'test' split in path)."""
    return [f for f in files if f'/{exclude_pattern}/' not in f]


def run_benchmark(datasets, output_dir, num_samples=100, levels=[0, 1, 2, 3], 
                  visualize=True, verbose=True):
    """
    Run layout diversity analysis on multiple datasets.
    
    Args:
        datasets: Dict of dataset configs (name -> config dict)
        output_dir: Directory to save results
        num_samples: Number of images to sample per dataset
        levels: Pyramid levels for SPM
        visualize: Generate visualization plots
        verbose: Print progress
        
    Returns:
        Dict of results for all datasets
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for name, config in datasets.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {name}")
            print(f"{'='*60}")
            print(f"Description: {config.get('description', 'N/A')}")
        
        path = Path(config["path"])
        
        # Check if path exists
        if not path.exists():
            print(f"  WARNING: Path not found: {path}")
            print(f"  Skipping {name}...")
            continue
        
        try:
            # Handle special case for Cityscapes (exclude test set)
            if "exclude_pattern" in config:
                # Custom handling for datasets with split exclusion
                import glob
                all_files = glob.glob(str(path / config["pattern"]), recursive=True)
                filtered_files = filter_files_by_split(all_files, config["exclude_pattern"])
                
                if verbose:
                    print(f"  Files after filtering: {len(filtered_files)} (excluded '{config['exclude_pattern']}')")
                
                # Load and analyze manually
                from PIL import Image
                step = max(1, len(filtered_files) // num_samples)
                selected = sorted(filtered_files)[::step][:num_samples]
                
                masks = []
                for f in selected:
                    img = Image.open(f)
                    if config.get("resize"):
                        img = img.resize((config["resize"][1], config["resize"][0]), Image.NEAREST)
                    masks.append(np.array(img))
                masks = np.array(masks)
                
                from layout_diversity_score import compute_layout_similarity
                sim_matrix, avg_sim, diversity = compute_layout_similarity(
                    masks, config["num_classes"], levels=levels
                )
                
                off_diag = sim_matrix[~np.eye(len(masks), dtype=bool)]
                results = {
                    "dataset_path": str(path),
                    "total_files": len(filtered_files),
                    "analyzed_samples": len(selected),
                    "mask_shape": masks.shape[1:],
                    "num_classes": config["num_classes"],
                    "diversity_score": float(diversity),
                    "avg_similarity": float(avg_sim),
                    "similarity_min": float(off_diag.min()),
                    "similarity_max": float(off_diag.max()),
                    "similarity_std": float(off_diag.std()),
                }
            else:
                # Standard processing
                results, sim_matrix = analyze_dataset(
                    labels_dir=config["path"],
                    num_classes=config["num_classes"],
                    num_samples=num_samples,
                    levels=levels,
                    pattern=config["pattern"],
                    resize=config.get("resize"),
                    verbose=verbose
                )
            
            # Store results
            all_results[name] = results
            
            # Save individual results
            np.savez(
                output_dir / f"{name.lower()}_diversity.npz",
                similarity_matrix=sim_matrix,
                **results
            )
            
            # Generate visualizations
            if visualize:
                import matplotlib
                matplotlib.use('Agg')
                
                visualize_similarity_matrix(
                    sim_matrix, results,
                    output_path=str(output_dir / f"{name.lower()}_heatmap.png")
                )
                visualize_similarity_distribution(
                    sim_matrix, results,
                    output_path=str(output_dir / f"{name.lower()}_distribution.png")
                )
                
        except Exception as e:
            print(f"  ERROR processing {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_results


def generate_comparison_chart(results, output_path):
    """Generate a comparison bar chart of all datasets."""
    import matplotlib.pyplot as plt
    
    # Sort by diversity score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['diversity_score'], reverse=True)
    
    names = [name for name, _ in sorted_results]
    scores = [r['diversity_score'] for _, r in sorted_results]
    counts = [r['total_files'] for _, r in sorted_results]
    ranges = [(r['similarity_min'], r['similarity_max']) for _, r in sorted_results]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Diversity scores bar chart
    ax1 = axes[0]
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
    bars = ax1.barh(names, scores, color=colors, edgecolor='black', linewidth=0.5)
    
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{score:.4f}', va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Layout Diversity Score', fontsize=12)
    ax1.set_title('Layout Diversity by Dataset\n(Higher = More Diverse)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline (0.5)')
    ax1.legend(loc='lower right')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # Similarity range chart
    ax2 = axes[1]
    for i, (name, (low, high), count) in enumerate(zip(names, ranges, counts)):
        ax2.barh(i, high - low, left=low, height=0.6, color=colors[i], 
                edgecolor='black', alpha=0.8)
        ax2.text(high + 0.02, i, f'n={count:,}', va='center', fontsize=9)
    
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names)
    ax2.set_xlabel('Pairwise Similarity Range', fontsize=12)
    ax2.set_title('Similarity Distribution Range\n[Min, Max]', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1.15)
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison chart saved to: {output_path}")
    
    return fig


def generate_summary_table(results):
    """Generate a markdown summary table."""
    sorted_results = sorted(results.items(), key=lambda x: x[1]['diversity_score'], reverse=True)
    
    print("\n" + "="*80)
    print("LAYOUT DIVERSITY BENCHMARK RESULTS")
    print("="*80)
    print("\n| Dataset | Images | Classes | Diversity Score | Similarity Range |")
    print("|---------|--------|---------|-----------------|------------------|")
    
    for name, r in sorted_results:
        sim_range = f"[{r['similarity_min']:.2f}, {r['similarity_max']:.2f}]"
        note = ""
        if name == sorted_results[0][0]:
            note = " 🥇"
        elif name == sorted_results[-1][0]:
            note = " 🥉"
        print(f"| {name} | {r['total_files']:,} | {r['num_classes']} | "
              f"**{r['diversity_score']:.4f}**{note} | {sim_range} |")
    
    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run layout diversity benchmark on multiple datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python run_diversity_benchmark.py --output-dir ./diversity_results

This will analyze all configured datasets and generate:
    - Individual .npz result files
    - Heatmap visualizations
    - Distribution plots
    - Comparison chart
    - Summary JSON
        """
    )
    
    parser.add_argument("--output-dir", "-o", type=str, default="./diversity_results",
                        help="Output directory for results (default: ./diversity_results)")
    parser.add_argument("--num-samples", "-n", type=int, default=100,
                        help="Number of images to sample per dataset (default: 100)")
    parser.add_argument("--levels", "-l", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="Pyramid levels (default: 0 1 2 3)")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Skip visualization generation")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Specific datasets to run (default: all)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # Filter datasets if specific ones requested
    datasets = DATASETS
    if args.datasets:
        datasets = {k: v for k, v in DATASETS.items() if k in args.datasets}
        if not datasets:
            print(f"ERROR: No matching datasets found. Available: {list(DATASETS.keys())}")
            sys.exit(1)
    
    print(f"Layout Diversity Benchmark")
    print(f"=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Samples per dataset: {args.num_samples}")
    print(f"Datasets to analyze: {list(datasets.keys())}")
    
    start_time = time.time()
    
    # Run benchmark
    results = run_benchmark(
        datasets=datasets,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        levels=args.levels,
        visualize=not args.no_visualize,
        verbose=not args.quiet
    )
    
    if not results:
        print("ERROR: No datasets were successfully processed!")
        sys.exit(1)
    
    # Generate comparison chart
    if not args.no_visualize and len(results) > 1:
        import matplotlib
        matplotlib.use('Agg')
        generate_comparison_chart(results, Path(args.output_dir) / "comparison_chart.png")
    
    # Print summary table
    generate_summary_table(results)
    
    # Save summary JSON
    summary = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": args.num_samples,
        "pyramid_levels": args.levels,
        "results": {name: {k: v for k, v in r.items() if k != 'mask_shape'} 
                   for name, r in results.items()}
    }
    
    summary_path = Path(args.output_dir) / "benchmark_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")
    
    elapsed = time.time() - start_time
    print(f"\nBenchmark completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
