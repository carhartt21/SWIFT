# SWIFT - Scalable Weather Image Filtering Toolkit

<p align="center">
  <strong>🌦️ High-performance weather classification for large-scale image datasets</strong>
</p>

SWIFT is a GPU-accelerated Python toolkit for filtering and classifying outdoor images from large-scale datasets based on weather conditions using OpenAI's CLIP model.

## 🌟 Features

- **Large-Scale Processing**: Handle millions of images with LMDB-backed storage
- **Weather Classification**: 7 standard + 3 severe weather categories
- **CLIP-Powered**: Vision-language model for accurate classification
- **GPU Acceleration**: Optimized batch processing with CUDA support
- **Fog Detection**: Advanced fog detection with counter-prompt margin scoring
- **CLI Interface**: Easy-to-use command-line tool for batch processing
- **Caching System**: Smart caching to avoid redundant processing
- **Dataset Analysis**: Comprehensive visualization with balance metrics (Imbalance Ratio & Shannon Entropy)

## 🌦️ Weather Categories

### Standard Categories
| Category | Description |
|----------|-------------|
| `clear_day` | Sunny, bright daylight conditions |
| `cloudy` | Overcast, cloudy weather |
| `rainy` | Rainy scenes with visible precipitation |
| `snowy` | Snow-covered scenes and winter conditions |
| `foggy` | Misty, low-visibility conditions |
| `night` | Nighttime scenes with artificial lighting |
| `dawn_dusk` | Golden hour, sunrise/sunset lighting |

### Severe Weather Categories
| Category | Description |
|----------|-------------|
| `severe_rain` | Heavy torrential rain with intense downpour |
| `severe_snow` | Heavy snowfall with thick accumulation |
| `severe_fog` | Dense fog with extremely low visibility |

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/carhartt21/SWIFT.git
cd SWIFT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM for large datasets

## 🚀 Quick Start

### Command Line (Recommended)

```bash
# Basic usage
python swift.py /path/to/images /path/to/output

# With custom parameters
python swift.py /path/to/images /path/to/output \
    --batch_size 128 \
    --max_per_category 100 \
    --confidence_threshold 0.2

# Severe weather only
python swift.py /path/to/images /path/to/output \
    --severe_weather_only \
    --confidence_threshold 0.15
```

### Python API

```python
from swift import SWIFTPipeline

# Initialize pipeline
pipeline = SWIFTPipeline(batch_size=64)

# Run classification
results = pipeline.run(
    source_folder="/path/to/images",
    output_dir="output",
    confidence_threshold=0.2
)
```

## 📁 Project Structure

```
SWIFT/
├── swift.py                    # Main SWIFT pipeline (CLI + API)
├── loader.py                   # LMDB dataset loader
├── utils.py                    # Utility functions
├── requirements.txt            # Dependencies
├── scripts/
│   ├── helpers/               # Helper utilities
│   ├── data_processing/       # Data processing scripts
│   │   └── layout_diversity_score.py  # Layout diversity analysis
│   └── visualization/         # Visualization and analysis tools
│       ├── weather_dataset_visualizer.py  # Main visualizer
│       └── test_balance_metrics.py        # Balance metrics tests
└── docs/                       # Documentation
```

## 📊 Dataset Analysis

SWIFT includes a powerful visualization pipeline for analyzing classified datasets:

```bash
python scripts/visualization/weather_dataset_visualizer.py /path/to/datasets
```

### Balance Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| **Imbalance Ratio (IR)** | N_max / N_min between largest and smallest categories | IR = 1 (perfect) |
| **Normalized Shannon Entropy (H_norm)** | Distribution uniformity on 0-1 scale | H_norm = 1 (uniform) |

See [Visualization Tools](docs/readme_visualizer.md) for full documentation.

## 🎯 Layout Diversity Score

SWIFT includes a **Layout Diversity Score** tool for analyzing the semantic layout diversity of segmentation datasets. This metric uses Spatial Pyramid Matching (SPM) to measure how varied the scene compositions are across a dataset.

### Quick Start

```bash
# Basic usage
python scripts/data_processing/layout_diversity_score.py /path/to/labels --num-classes 35

# With visualization
python scripts/data_processing/layout_diversity_score.py /path/to/labels \
    --num-classes 35 \
    --num-samples 100 \
    --output results.npz \
    --visualize

# For variable-sized images (e.g., Mapillary)
python scripts/data_processing/layout_diversity_score.py /path/to/labels \
    --num-classes 66 \
    --resize 512 1024
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-classes, -c` | 35 | Number of semantic classes |
| `--num-samples, -n` | 100 | Images to analyze (-1 for all) |
| `--levels, -l` | 0 1 2 3 | Pyramid levels (1×1, 2×2, 4×4, 8×8) |
| `--pattern, -p` | *.png | Glob pattern (supports ** for recursive) |
| `--resize, -r` | None | Resize to (H, W) for variable-sized datasets |
| `--output, -o` | None | Save results to .npz file |
| `--visualize, -v` | False | Generate heatmap and distribution plots |

### Python API

```python
from scripts.data_processing.layout_diversity_score import analyze_dataset, compute_layout_similarity

# High-level analysis
results, similarity_matrix = analyze_dataset(
    labels_dir="/path/to/labels",
    num_classes=35,
    num_samples=100
)
print(f"Diversity Score: {results['diversity_score']:.4f}")

# Low-level API for custom workflows
import numpy as np
masks = np.array([...])  # (N, H, W) array of segmentation masks
sim_matrix, avg_sim, diversity = compute_layout_similarity(masks, num_classes=35)
```

### Interpretation

| Diversity Score | Interpretation |
|-----------------|----------------|
| 0.0 - 0.3 | Low diversity (very similar layouts) |
| 0.3 - 0.5 | Moderate-low diversity |
| 0.5 - 0.7 | Moderate diversity |
| 0.7 - 1.0 | High diversity (varied layouts) |

### Benchmark Results

| Dataset | Images | Classes | Diversity Score |
|---------|--------|---------|-----------------|
| OUTSIDE15k | 14,884 | 24 | **0.8054** (Most Diverse) |
| BDD10k | 8,000 | 19 | **0.6605** |
| Mapillary | 20,000 | 66 | **0.6508** |
| ACDC | 2,006 | 34 | **0.5646** |
| IDD | 7,859 | 27 | **0.5572** |
| Cityscapes | 3,475 | 34 | **0.4736** (Least Diverse) |

> **Note**: Use only fully-annotated splits (train/val). Test sets may have placeholder annotations.

## ⚙️ CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--batch_size` | 64 | Images per batch (GPU memory dependent) |
| `--max_per_category` | 0 | Max images per category (0 = unlimited) |
| `--confidence_threshold` | 0.1 | Minimum confidence score |
| `--margin_threshold` | 0.01 | Margin between top-1 and top-2 |
| `--fog_margin_threshold` | 0.01 | Fog detection margin |
| `--severe_weather_only` | False | Only severe weather categories |
| `--force_convert` | False | Force LMDB rebuild |
| `--no_cache` | False | Disable folder scanning cache |

## 📊 Output

SWIFT generates organized output with metadata:

```
output/
├── sa1b_optimized.lmdb      # LMDB database
├── path_to_index.json       # Path mapping
├── dataset_stats.json       # Statistics
└── categories/
    ├── clear_day/
    │   ├── image_001.jpg
    │   └── image_001.json   # Metadata
    ├── cloudy/
    ├── rainy/
    └── ...
```

## 🚀 Performance

| Operation | Speed |
|-----------|-------|
| LMDB Conversion | 100-500+ img/s |
| Classification | 50-200+ img/s |

*Performance varies by hardware configuration*

## 📚 Documentation

- [Dataset Splitting Guide](docs/readme_dataset_split.md)
- [Visualization Tools](docs/readme_visualizer.md)

## 📄 License

This project is open source. Please ensure compliance with dataset licenses:
- SA-1B Dataset: Meta's Segment Anything Dataset license
- LAION-400M: Creative Commons licenses (varies by image)

## 🤝 Contributing

Contributions welcome! Please submit pull requests or open issues for:
- Performance improvements
- Additional weather categories
- Bug fixes and optimizations
- Documentation improvements

## 📚 References

- [CLIP: Learning Transferable Visual Representations](https://openai.com/blog/clip/)
- [SA-1B Dataset](https://segment-anything.com/dataset/index.html)
- [LAION-400M Dataset](https://laion.ai/blog/laion-400-open-dataset/)

---
