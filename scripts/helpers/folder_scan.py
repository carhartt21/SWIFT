
import pickle
import time
import os
import json
from collections import defaultdict
from pathlib import Path
import logging
import argparse

logger = logging.getLogger(__name__)

class FolderScanCache:
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_path(self, source_folder):
        """Generate cache filename based on source folder"""
        folder_hash = abs(hash(str(source_folder)))
        return self.cache_dir / f"scan_cache_{folder_hash}.pkl"
    
    def save_scan_results(self, source_folder, file_list):
        """Save scanning results to cache"""
        cache_path = self.get_cache_path(source_folder)
        
        cache_data = {
            'source_folder': str(source_folder),
            'scan_timestamp': time.time(),
            'file_count': len(file_list),
            'file_list': [str(f) for f in file_list]  # Convert Path objects to strings
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Saved scan cache: {len(file_list):,} files to {cache_path}")
        return cache_path
    
    def load_scan_results(self, source_folder, max_age_hours=24):
        """Load scanning results from cache if valid"""
        cache_path = self.get_cache_path(source_folder)
        
        if not cache_path.exists():
            logger.info("No cache file found")
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check cache age
            cache_age = time.time() - cache_data['scan_timestamp']
            if cache_age > (max_age_hours * 3600):
                logger.info(f"Cache expired ({cache_age/3600:.1f} hours old)")
                return None
            
            # Convert strings back to Path objects
            file_list = [Path(f) for f in cache_data['file_list']]
            
            logger.info(f"Loaded cached scan: {len(file_list):,} files "
                       f"(age: {cache_age/3600:.1f} hours)")
            return file_list
            
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None

# Integration with your pipeline
def scan_flat_directory_with_cache(self, source_folder: str, use_cache: bool = True,
                                  max_cache_age_hours: int = 24):
    """Enhanced scanning with caching support"""
    
    cache_manager = FolderScanCache()
    
    # Try to load from cache first
    if use_cache:
        cached_files = cache_manager.load_scan_results(source_folder, max_cache_age_hours)
        if cached_files is not None:
            return cached_files
    
    # Perform actual scan if cache miss
    logger.info("Performing fresh directory scan...")
    start_time = time.time()
    
    root_path = Path(source_folder)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    
    all_files = [f for f in root_path.iterdir() 
                if f.is_file() and f.suffix in extensions]
    
    scan_time = time.time() - start_time
    logger.info(f"Fresh scan complete: {len(all_files):,} files in {scan_time:.1f} seconds")
    
    # Save to cache
    if use_cache:
        cache_manager.save_scan_results(source_folder, all_files)
    
    return all_files


def scan_recursive_and_write_mapping(source_folder: str, output_dir: str,
                                     use_cache: bool = True,
                                     max_cache_age_hours: int = 24,
                                     progress_interval: int = 10000):
    """Scan folder recursively using os.scandir (original implementation) and
    write a mapping file path->idx (absolute resolved paths).

    Returns the mapping dict (path -> index).
    """
    cache_manager = FolderScanCache()

    # Try cache first
    if use_cache:
        cached_files = cache_manager.load_scan_results(source_folder, max_cache_age_hours)
        if cached_files is not None:
            all_files = cached_files
        else:
            all_files = []
    else:
        all_files = []

    if not all_files:
        logger.info(f"Starting optimized scan of {source_folder}")
        start_time = time.time()

        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
        all_files = []
        processed_count = 0
        folder_counts = defaultdict(int)

        # Use os.scandir() - original implementation with recursion
        def _scan_dir(folder):
            nonlocal all_files, processed_count, folder_counts
            try:
                with os.scandir(folder) as entries:
                    for entry in entries:
                        try:
                            if entry.is_file():
                                if any(entry.name.endswith(ext) for ext in extensions):
                                    all_files.append(Path(entry.path))
                                    processed_count += 1
                                    # count under top-level directory for an overview
                                    try:
                                        rel = Path(entry.path).resolve().relative_to(Path(source_folder).resolve())
                                        top = rel.parts[0] if len(rel.parts) > 0 else ''
                                    except Exception:
                                        top = ''
                                    folder_counts[top] += 1

                                    if processed_count % progress_interval == 0:
                                        elapsed = time.time() - start_time
                                        rate = processed_count / elapsed if elapsed > 0 else 0
                                        logger.info(f"Scanned {processed_count:,} files in {elapsed:.1f}s ({rate:.0f} files/sec)")
                            elif entry.is_dir():
                                _scan_dir(entry.path)
                        except Exception:
                            # ignore problematic entries
                            continue
            except Exception as e:
                logger.error(f"Error scanning {folder}: {e}")

        _scan_dir(source_folder)

        total_time = time.time() - start_time
        final_rate = processed_count / total_time if total_time > 0 else 0
        logger.info(f"Scan complete: {processed_count:,} image files in {total_time:.1f} seconds")
        logger.info(f"Final scan rate: {final_rate:.0f} files/second")

        # Print per-top-level-folder counts for overview
        try:
            logger.info("Top-level folder counts:")
            for k, v in sorted(folder_counts.items(), key=lambda x: (-x[1], x[0])):
                name = k if k else '(root)'
                logger.info(f"  {name}: {v}")
        except Exception:
            pass

        # Save to cache
        if use_cache:
            cache_manager.save_scan_results(source_folder, all_files)

    # Build mapping path -> index
    path_index_map = {str(Path(p).resolve()): idx for idx, p in enumerate(all_files)}

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    map_json_path = out_path / 'path_to_index.json'
    map_pkl_path = out_path / 'path_to_index.pkl'

    try:
        with open(map_json_path, 'w') as jf:
            json.dump(path_index_map, jf, indent=2)
        with open(map_pkl_path, 'wb') as pf:
            pickle.dump(path_index_map, pf)
        logger.info(f"Wrote path->index mapping ({len(path_index_map):,} entries) to {map_json_path} and {map_pkl_path}")
    except Exception as e:
        logger.error(f"Failed to write mapping files: {e}")

    return path_index_map


def main():
    parser = argparse.ArgumentParser(description="Scan a folder recursively and write path->index mapping")
    parser.add_argument('source_folder', help='Root folder to scan for images')
    parser.add_argument('output_dir', help='Directory where mapping files will be written')
    parser.add_argument('--no-cache', action='store_true', dest='no_cache', help='Disable using the scan cache')
    parser.add_argument('--sort', action='store_true', dest='sort', help='Sort file paths before assigning indices (deterministic)')
    parser.add_argument('--write-index', action='store_true', dest='write_index', help='Also write index->path JSON alongside path->index')

    args = parser.parse_args()

    use_cache = not args.no_cache
    mapping = scan_recursive_and_write_mapping(args.source_folder, args.output_dir, use_cache=use_cache)

    if not mapping:
        logger.error('No files found or mapping empty')
        return

    if args.sort:
        # Rebuild mapping deterministically by sorting resolved paths
        sorted_paths = sorted(mapping.keys())
        mapping = {p: idx for idx, p in enumerate(sorted_paths)}
        # overwrite the mapping files with the sorted mapping
        out_path = Path(args.output_dir)
        try:
            with open(out_path / 'path_to_index.json', 'w') as jf:
                json.dump(mapping, jf, indent=2)
            with open(out_path / 'path_to_index.pkl', 'wb') as pf:
                pickle.dump(mapping, pf)
            logger.info('Rewrote mapping files using sorted paths')
        except Exception as e:
            logger.error(f'Failed to rewrite mapping files: {e}')

    if args.write_index:
        # write index->path mapping as JSON (string keys)
        index_to_path = {str(v): k for k, v in mapping.items()}
        out_path = Path(args.output_dir)
        try:
            with open(out_path / 'index_to_path.json', 'w') as jf:
                json.dump(index_to_path, jf, indent=2)
            logger.info('Wrote index_to_path.json')
        except Exception as e:
            logger.error(f'Failed to write index_to_path.json: {e}')


if __name__ == '__main__':
    main()
