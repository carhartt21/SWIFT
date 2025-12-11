import argparse
import json
import pickle
from pathlib import Path
import os
import shutil
from tqdm import tqdm
from PIL import Image, ImageDraw

def load_mapping(mapping_path):
    mapping = None
    if mapping_path.suffix == ".json":
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
    elif mapping_path.suffix == ".pkl":
        with open(mapping_path, "rb") as f:
            mapping = pickle.load(f)
    else:
        raise ValueError("Unsupported mapping file format.")
    # Invert mapping: index (as str) -> path
    index_to_path = {str(idx): path for path, idx in mapping.items()}
    return index_to_path


def _create_placeholder(text, size=(400, 300), bgcolor=(200, 200, 200)):
    img = Image.new('RGB', size, color=bgcolor)
    draw = ImageDraw.Draw(img)
    # Simple centering
    lines = text.split('\n')
    y = size[1] // 2 - (6 * len(lines))
    for line in lines:
        w, h = draw.textsize(line)
        draw.text(((size[0] - w) // 2, y), line, fill=(0, 0, 0))
        y += h + 2
    return img


def _debug_mapping_samples(json_files, index_to_path, json_dir, debug_n, debug_dir, offset=0):
    """Create side-by-side debug images for the first N JSON files.

    Args:
        json_files: List of JSON files to sample.
        index_to_path: Mapping from index (str) to image path.
        json_dir: Root directory of JSON files.
        debug_n: Number of samples to debug.
        debug_dir: Directory to save debug images.
        offset: Integer offset to apply to original_index for mapping (default: 0).
            This can be used to test mapping with a shifted index (e.g., for datasets with known offsets).
    For each JSON file sampled, try to locate an "input" image referenced in the JSON
    (common keys), and the mapped original image from index_to_path. Save a combined
    side-by-side PNG to debug_dir preserving the json subfolder structure.
    """
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    samples = list(json_files)[:debug_n]
    for idx, json_file in enumerate(samples):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            original_index = data.get('original_index')
            # Allow an optional per-JSON offset (keys: index_offset, offset, original_index_offset)
            # The offset parameter is now configurable for flexibility.
        
            # Apply offset to numeric original_index while keeping it within total number of mapped entries
            try:
                idx_int = int(original_index)
                total = max(1, len(index_to_path))
                # Wrap around to ensure the resulting index is within [0, total-1]
                new_idx = (idx_int + offset) % total
                original_index = str(new_idx)
            except Exception:
                # If original_index isn't an integer, leave it unchanged
                pass
            if original_index is None:
                # skip if missing
                print(f"[DEBUG] {json_file}: missing original_index, skipping")
                continue
            original_index = str(original_index)
            mapped_path = index_to_path.get(original_index)

            mapped_img_path = Path(mapped_path) if mapped_path else None

            # Attempt to find an "input" image referenced inside the JSON
            input_img_path = None
            candidate_keys = ['input_image', 'image_path', 'img_path', 'file_name', 'original_image_path', 'segmentation_path', 'file']
            key_attempts = []
            for k in candidate_keys:
                v = data.get(k)
                key_attempts.append((k, v))
                if not v:
                    continue
                p = Path(v)
                if not p.is_absolute():
                    p = (json_file.parent / p).resolve()
                key_attempts.append((f'resolved_{k}', str(p)))
                if p.exists():
                    input_img_path = p
                    key_attempts.append((f'found_{k}', str(p)))
                    break

            # If not found, try to locate by stem in the same folder as the JSON
            if input_img_path is None:
                stem = json_file.stem
                for ext in img_exts:
                    candidate = json_file.parent / (stem + ext)
                    if candidate.exists():
                        input_img_path = candidate
                        break

            # Build debug log text for this sample
            debug_lines = []
            debug_lines.append(f'json_file: {json_file}')
            debug_lines.append(f'original_index: {original_index}')
            debug_lines.append(f'mapped_path (from mapping): {mapped_path}')
            debug_lines.append('candidate_key_attempts:')
            for ka, kv in key_attempts:
                debug_lines.append(f'  - {ka}: {kv}')
            debug_lines.append(f'stem_fallback_checked: {stem}')
            debug_lines.append(f'input_img_resolved: {input_img_path}')
            debug_lines.append(f'mapped_img_resolved: {mapped_img_path}')

            # Print brief summary to console
            print(f"[DEBUG SAMPLE {idx}] json={json_file.name} index={original_index} mapped={'YES' if mapped_img_path and mapped_img_path.exists() else 'NO'} input={'YES' if input_img_path and input_img_path.exists() else 'NO'}")

            # Prepare images (open or placeholder)
            try:
                if input_img_path and input_img_path.exists():
                    im1 = Image.open(input_img_path).convert('RGB')
                else:
                    im1 = _create_placeholder('input image\nmissing', size=(400, 300))
            except Exception:
                im1 = _create_placeholder('input image\nerror', size=(400, 300))

            try:
                if mapped_img_path and mapped_img_path.exists():
                    im2 = Image.open(mapped_img_path).convert('RGB')
                else:
                    im2 = _create_placeholder('mapped image\nmissing', size=(400, 300))
            except Exception:
                im2 = _create_placeholder('mapped image\nerror', size=(400, 300))

            # Resize to same height (limit to 512 to avoid huge images)
            target_height = min(im1.height, im2.height, 512)
            def _resize_keep_aspect(im, h):
                w = int(im.width * (h / im.height))
                return im.resize((w, h), Image.LANCZOS)

            im1r = _resize_keep_aspect(im1, target_height)
            im2r = _resize_keep_aspect(im2, target_height)

            # Combine side by side with a small separator and a thin label area
            sep = 8
            combined = Image.new('RGB', (im1r.width + im2r.width + sep, target_height + 24), color=(255, 255, 255))
            combined.paste(im1r, (0, 0))
            combined.paste(im2r, (im1r.width + sep, 0))

            draw = ImageDraw.Draw(combined)
            # Overlay original_index and JSON filename at the top-left
            overlay_text = f'original_index: {original_index} | json: {json_file.name}'
            draw.text((4, 4), overlay_text, fill=(0, 0, 0))

            label1 = f'input: {input_img_path.name if input_img_path else "(none)"}'
            label2 = f'mapped: {mapped_img_path.name if mapped_img_path else "(none)"}'
            draw.text((4, target_height + 4), label1, fill=(0, 0, 0))
            draw.text((im1r.width + sep + 4, target_height + 4), label2, fill=(0, 0, 0))

            # Save preserving json relative folder structure
            try:
                json_rel_path = json_file.parent.relative_to(json_dir)
            except Exception:
                json_rel_path = Path(json_file.parent.name)

            out_dir = Path(debug_dir) / json_rel_path
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f'debug_{idx:03d}_{json_file.stem}.png'
            combined.save(out_path)
            # Save debug text log alongside the image for detailed inspection
            log_path = out_dir / f'debug_{idx:03d}_{json_file.stem}.txt'
            try:
                with open(log_path, 'w') as lf:
                    lf.write('\n'.join(debug_lines))
            except Exception as e:
                print(f"Failed to write debug log for {json_file}: {e}")

        except Exception as e:
            print(f"Debug sample error for {json_file}: {e}")

def process_json_files(json_dir, mapping_path, dest_dir,
                       iterate_parent=False, overwrite=True,
                       write_metadata=True,
                       debug_n=0, debug_dir=None):
    """Process JSON files, add original image path, and optionally copy images and metadata.

    Args:
        json_dir: directory containing JSON files (or parent of multiple subfolders)
        mapping_path: path to mapping file (json or pkl)
        dest_dir: destination root where original images (and metadata) will be copied
        iterate_parent: if True, iterate over immediate subdirectories of json_dir and process their JSONs
        overwrite: whether to overwrite existing copied images
        write_metadata: whether to write a copy of the metadata alongside the copied image
        source_root: optional path to treat as the root of the original image paths; if omitted the commonpath of mapping values is used
    """

    index_to_path = load_mapping(Path(mapping_path))

    json_dir = Path(json_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Collect list of JSON files to process
    json_files = []
    if iterate_parent:
        for child in sorted(json_dir.iterdir()):
            if child.is_dir():
                json_files.extend(list(child.rglob('*.json')))
    else:
        json_files = list(json_dir.rglob('*.json'))

    # Report how many JSON files were found for processing
    print(f"Found {len(json_files)} JSON files in {json_dir} (iterate_parent={iterate_parent})")

    if not json_files:
        print(f"No JSON files found in {json_dir} (iterate_parent={iterate_parent})")
        return

    # If debugging requested, run a small sample test to validate mapping and exit
    if debug_n and debug_n > 0:
        if debug_dir is None:
            debug_dir = dest_dir / 'debug_samples'
        else:
            debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running debug samples (n={debug_n}) and exiting main processing loop. Debug output: {debug_dir}")
        # You can change the offset value here if needed for your dataset.
        _debug_mapping_samples(json_files, index_to_path, json_dir, debug_n, debug_dir, offset=0)
        print("Debug sampling complete. Exiting without copying files (debug mode).")
        return

    for json_file in tqdm(json_files, desc=f"Processing {len(json_files)} JSON files"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            original_index = data.get('original_index')
            if original_index is None:
                print(f"original_index missing in {json_file}")
                continue

            original_index = str(original_index)
            image_path = index_to_path.get(original_index)
            if not image_path:
                print(f"Index {original_index} not found in mapping for {json_file}.")
                continue

            src_img = Path(image_path)
            if not src_img.exists():
                print(f"Source image not found: {src_img} (from {json_file})")
                continue

            # Preserve JSON directory structure in destination
            try:
                json_rel_path = json_file.parent.relative_to(json_dir)
            except ValueError:
                # If json_file is not under json_dir, use its parent name
                json_rel_path = Path(json_file.parent.name)

            # Compute image filename (preserve original extension)
            image_filename = src_img.name
            # Compute JSON filename based on original image name
            json_filename = src_img.stem + '.json'

            # Create destination path preserving JSON folder structure
            dst_img_path = dest_dir / json_rel_path / image_filename
            dst_img_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy image
            if overwrite or not dst_img_path.exists():
                shutil.copy2(src_img, dst_img_path)

            # Update metadata in original JSON (add path) and optionally write a copy next to image
            data['original_image_path'] = str(src_img)

            if write_metadata:
                # Use original image name for the JSON metadata file
                meta_dst = dst_img_path.parent / json_filename
                with open(meta_dst, 'w') as mf:
                    json.dump(data, mf, indent=2)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Add original image path to JSON files based on mapping and copy images/metadata.")
    parser.add_argument("json_dir", help="Directory to scan for JSON files.")
    parser.add_argument("mapping_file", help="Path to mapping file (path_to_index.json or .pkl).")
    parser.add_argument("dest_dir", help="Destination root to copy original images and metadata to.")
    parser.add_argument("--scan-parent", action='store_true', dest='scan_parent',
                        help="Iterate immediate subdirectories of json_dir and process JSONs inside them")
    parser.add_argument("--no-overwrite", action='store_true', dest='no_overwrite',
                        help="Do not overwrite existing files in destination")
    parser.add_argument("--no-metadata", action='store_true', dest='no_metadata',
                        help="Do not write metadata JSON files next to copied images")
    parser.add_argument("--source-root", type=str, default=None,
                        help="Optional source root to preserve folder structure relative to this path")
    parser.add_argument("--debug", type=int, default=0,
                        help="Number of sample JSONs to create debug side-by-side images for")
    parser.add_argument("--debug-dir", type=str, default=None,
                        help="Optional directory to write debug images into (defaults to <dest_dir>/debug_samples)")

    args = parser.parse_args()

    process_json_files(
        args.json_dir,
        args.mapping_file,
        args.dest_dir,
        iterate_parent=args.scan_parent,
        overwrite=not args.no_overwrite,
        write_metadata=not args.no_metadata,
        debug_n=args.debug,
        debug_dir=args.debug_dir
    )

if __name__ == "__main__":
    main()
