# Dataset Splitter Script

This script organizes datasets with weather/lighting categories into reproducible **train/test splits** for tasks like semantic segmentation, object detection, or universal segmentation.  
Both image and label files are copied into an output folder preserving the hierarchy:  
`<OUTPUT>/<split>/images/<dataset>/<category>/` and `<OUTPUT>/<split>/labels/<dataset>/<category>/`.

## Directory Assumptions

Your input data must be organized like:

INPUT_ROOT/
DATASET1/
images/
rainy/
snowy/
...
labels/
rainy/
snowy/
...
DATASET2/
...


## Usage

python dataset_splitter.py
--input /path/to/your/repositories
--output /path/to/output_splits
--train_ratio 0.7
--generate_lists


- Use `--train_ratio 0.7` or your preferred split (e.g., 0.8)
- Use `--datasets DATASET1 DATASET2 ...` to restrict which datasets to process (default: all)

**Example with specific datasets:**

python dataset_splitter.py
--input /media/chge7185/HDD1/repositories
--output ./splits
--datasets ACDC BDD10k MapillaryVistas


## What It Does

- Finds image/label pairs in each `category` subfolder
- Maintains original filenames and folder layout (`dataset/category`) in output
- Shuffles and splits each category independently
- Supports most common image/label file extensions (`.jpg .jpeg .png .json`)
- Generates a `split_statistics.json` file with per-category counts
- (Optionally) generates `train_images.txt` and `test_images.txt` listing all images for each split

## Output Structure

output/
train/
images/
DATASET/
CATEGORY/
<original_filename>.jpg/png
labels/
DATASET/
CATEGORY/
<original_filename>.png/json
test/
...
split_statistics.json
train_images.txt (if --generate_lists)
test_images.txt (if --generate_lists)

text

## Requirements

- Python 3.7+
- Standard library only

## Notes

- Files are **not renamed** — original names are preserved in the output.
- Make sure that label files are uniquely named **within each dataset/category** directory.
- Warnings will be printed for missing label files or categories.
- The split is reproducible if you specify a `--seed`.

---

For advanced balancing or custom rules (as discussed in earlier research steps), you would tweak the main loop to select a subset of images per category accordingly.