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
│   └── visualization/         # Visualization tools
└── docs/                       # Documentation
```

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
