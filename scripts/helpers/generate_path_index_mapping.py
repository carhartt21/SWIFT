import argparse
import json
import pickle
from pathlib import Path
import os

def scan_files(source_folder, extensions='.jpg', exclude_dirs=None):
    """Recursively scan for all files in the source folder, adding files in the parent directory first.

    Args:
        source_folder: root folder to scan
        extensions: optional iterable of extensions to include (e.g. ['.jpg', '.png']). If None all files included.
        exclude_dirs: optional iterable of directory names to skip (e.g. ['.git', '__pycache__'])
    """
    all_files = []
    src = Path(source_folder)
    if not src.exists():
        return all_files

    print(f"Starting scan of: {source_folder}")

    if exclude_dirs is None:
        exclude_dirs = {'.git', '__pycache__'}
    else:
        exclude_dirs = set(exclude_dirs)

    # Normalize extensions to a set of lower-case strings starting with a dot
    if extensions:
        extensions = {e.lower() if e.startswith('.') else f'.{e.lower()}' for e in extensions}
    else:
        extensions = None

    # Add files directly in the source (parent) directory first, in sorted order for determinism
    try:
        entries = list(os.scandir(src))
        entries.sort(key=lambda e: e.name)
        for entry in entries:
            try:
                if entry.is_file(follow_symlinks=False):
                    p = Path(entry.path)
                    if extensions is None or p.suffix.lower() in extensions:
                        all_files.append(str(p.resolve()))
                        # Print progress every 1000 files for the parent directory pass
                        if len(all_files) % 1000 == 0:
                            print(f"Scanned {len(all_files)} files...")
            except PermissionError:
                # Skip unreadable entries
                continue
    except PermissionError:
        pass

    print(f"Top-level files found: {len(all_files)}")

    # Helper: fast recursive scandir yielding files (includes all subdirectories)
    def scandir_recursive(root_path):
        stack = [str(root_path)]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    entries = list(it)
            except PermissionError:
                continue
            entries.sort(key=lambda e: e.name)
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name in exclude_dirs:
                            continue
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        p = Path(entry.path)
                        if extensions is None or p.suffix.lower() in extensions:
                            yield entry.path
                except Exception:
                    continue

    # Then walk subdirectories using fast scandir_recursive (skip root files since already processed)
    count = len(all_files)
    print("Walking subdirectories...")
    files_added = 0
    for file_path in scandir_recursive(src):
        parent = Path(file_path).parent
        # Skip files directly in the root since they were already added above
        if parent == src:
            continue
        all_files.append(str(Path(file_path).resolve()))
        files_added += 1
        count += 1
        if count % 1000 == 0:
            print(f"Scanned {count} files...")

    # Final progress print
    print(f"Total files scanned: {len(all_files)} (added {files_added} from subdirectories)")
    return all_files


def generate_mapping(source_folder, output_dir, extensions=None, exclude_dirs=None):
    all_files = scan_files(source_folder, extensions=extensions, exclude_dirs=exclude_dirs)
    if not all_files:
        print("No files found in the source folder.")
        return
    print(f"Generating mapping for {len(all_files)} files...")
    path_index_map = {str(Path(p).resolve()): idx for idx, p in enumerate(all_files)}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    map_json_path = output_path / "path_to_index.json"
    map_pickle_path = output_path / "path_to_index.pkl"
    print(f"Writing mapping files to: {output_path}")
    with open(map_json_path, "w") as jf:
        json.dump(path_index_map, jf, indent=2)
    with open(map_pickle_path, "wb") as pf:
        pickle.dump(path_index_map, pf)
    print(f"Wrote path->index mapping ({len(path_index_map)}) entries to {map_json_path} and {map_pickle_path}")
    # Print a small sample of the mapping for verification
    sample_items = list(path_index_map.items())[:5]
    if sample_items:
        print("Sample mappings (first 5):")
        for p, i in sample_items:
            print(f"  {i}: {p}")


def main():
    parser = argparse.ArgumentParser(description="Generate path-to-index mapping files for a directory.")
    parser.add_argument("input_dir", help="Input directory to scan for files.")
    parser.add_argument("output_dir", help="Directory to write mapping files.")
    parser.add_argument('--extensions', type=str, default=None,
                        help='Comma-separated list of extensions to include (e.g. .jpg,.png). If omitted all files are included.')
    parser.add_argument('--exclude-dirs', type=str, default=None,
                        help='Comma-separated list of directory names to exclude (e.g. .git,__pycache__).')
    args = parser.parse_args()
    exts = [e.strip() for e in args.extensions.split(',')] if args.extensions else None
    excludes = [d.strip() for d in args.exclude_dirs.split(',')] if args.exclude_dirs else None
    generate_mapping(args.input_dir, args.output_dir, extensions=exts, exclude_dirs=excludes)

if __name__ == "__main__":
    main()
