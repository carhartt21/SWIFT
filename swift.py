#!/usr/bin/env python3
"""
SWIFT - Scalable Weather Image Filtering Toolkit

A high-performance pipeline for filtering and classifying outdoor images from 
large-scale datasets based on weather conditions using OpenAI's CLIP model.

Features:
- GPU-accelerated batch processing with CUDA support
- LMDB integration for fast database operations
- 7 standard weather categories + 3 severe weather categories
- Advanced fog detection using counter-prompt margin scoring
- Efficient caching and parallel processing
"""

import os
import sys
import time
import json
import pickle
import lmdb
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Optional, Any
import argparse
import logging
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Module-level functions for parallel processing
def process_file_chunk(file_chunk):
    """Process a chunk of files for parallel processing"""
    import cv2
    import numpy as np
    from pathlib import Path
    
    processed_results = []
    
    for file_path in file_chunk:
        try:
            # Load and process image
            img = cv2.imread(str(file_path))
            if img is None:
                continue
            
            # Resize for efficiency
            height, width = img.shape[:2]
            if max(height, width) > 512:
                scale = 512 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height), 
                               interpolation=cv2.INTER_AREA)
            
            # Encode image with optimized settings
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85,
                           cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            img_encoded = cv2.imencode('.jpg', img, encode_params)[1].tobytes()
            
            processed_results.append({
                'path': str(file_path),
                'data': img_encoded,
                'size': len(img_encoded)
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return processed_results

class FolderScanCache:
    """Caching system for folder scanning results"""
    
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_path(self, source_folder):
        folder_hash = abs(hash(str(source_folder)))
        return self.cache_dir / f"scan_cache_{folder_hash}.pkl"
    
    def save_scan_results(self, source_folder, file_list):
        cache_path = self.get_cache_path(source_folder)
        cache_data = {
            'source_folder': str(source_folder),
            'scan_timestamp': time.time(),
            'file_count': len(file_list),
            'file_list': [str(f) for f in file_list]
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Saved scan cache: {len(file_list):,} files to {cache_path}")
        return cache_path
    
    def load_scan_results(self, source_folder, max_age_hours=24):
        cache_path = self.get_cache_path(source_folder)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            cache_age = time.time() - cache_data['scan_timestamp']
            if cache_age > (max_age_hours * 3600):
                logger.info(f"Cache expired ({cache_age/3600:.1f} hours old)")
                return None
            
            file_list = [Path(f) for f in cache_data['file_list']]
            logger.info(f"Loaded cached scan: {len(file_list):,} files "
                       f"(age: {cache_age/3600:.1f} hours)")
            return file_list
            
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None

class SA1BLMDBDataset(Dataset):
    """Optimized LMDB dataset for SA-1B images with proper CLIP preprocessing"""
    
    def __init__(self, lmdb_path: str, preprocess_fn=None, cache_size: int = 1000):
        self.lmdb_path = lmdb_path
        self.preprocess_fn = preprocess_fn
        self.cache_size = cache_size
        self.cache = {}
        self.cache_order = []
        
        # Get dataset size
        with lmdb.open(str(lmdb_path), readonly=True, lock=False) as env:
            with env.begin() as txn:
                self.length = env.stat()['entries']
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            return self.cache[idx], idx
        
        # Load from LMDB
        with lmdb.open(str(self.lmdb_path), readonly=True, lock=False, 
                      readahead=False, meminit=False) as env:
            with env.begin() as txn:
                key = f"{idx:08d}".encode('ascii')
                img_bytes = txn.get(key)
                
                if img_bytes is None:
                    return None, idx
                
                # Decode image
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                
                # Use CLIP's preprocessing if available
                if self.preprocess_fn:
                    img = self.preprocess_fn(img)
                else:
                    # Fallback preprocessing
                    if max(img.size) > 512:
                        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                    img = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                    ])(img)
                
                # Cache management
                if len(self.cache) >= self.cache_size:
                    oldest = self.cache_order.pop(0)
                    del self.cache[oldest]
                
                self.cache[idx] = img
                self.cache_order.append(idx)
                
                return img, idx

class SWIFTPipeline:
    """SWIFT - Scalable Weather Image Filtering Toolkit
    
    Main pipeline class for weather classification of outdoor images using CLIP.
    Supports both standard and severe weather categories with advanced fog detection.
    """
    
    def __init__(self, batch_size: int = 64, num_workers: int = 0, fog_margin_threshold: float = 0.15):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fog_margin_threshold = fog_margin_threshold
        
        # Load CLIP model
        logger.info(f"Loading CLIP model on {self.device}")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Define improved prompts - more distinct and specific
        self.outdoor_prompts = [
            "an outdoor scene with trees and sky",
            "a street view with buildings and roads",
            "an open landscape with horizon visible",
            "an exterior view of buildings and nature",
            "an outdoor environment with natural lighting"
        ]
        
        self.indoor_prompts = [
            "an indoor room with walls and ceiling",
            "an interior space with artificial lighting",
            "inside a building with closed environment",
            "a room interior with furniture and walls",
            "an enclosed indoor space"
        ]
        
        # Improved weather categories with more distinct prompts
        self.weather_categories = {
            'clear_day': [
                "bright sunny day with blue sky and strong shadows",
                "clear weather with bright sunlight and vivid colors",
                "sunny outdoor scene with sharp contrast and blue sky",
                "daylight scene with bright illumination and clear visibility",
                "clear day with strong natural lighting and shadows"
            ],
            'cloudy': [
                "overcast sky with gray clouds covering the sun",
                "cloudy weather with diffused lighting and gray sky",
                "cloudy day with muted colors and soft shadows",
                "overcast conditions with cloud cover and dim lighting",
                "gray cloudy sky with uniform diffused light"
            ],
            'rainy': [
                "wet streets with rain puddles and reflections",
                "rainy weather with water drops and wet surfaces",
                "rain scene with umbrellas and wet pavement",
                "stormy weather with rain and dark clouds",
                "wet conditions with rain and water on surfaces"
            ],
            'snowy': [
                "snow covered ground with white precipitation",
                "winter scene with snow falling and white landscape",
                "snowy weather with snowflakes and winter conditions",
                "white snow on trees and ground in winter",
                "cold snowy scene with snow accumulation"
            ],
            'night': [
                "nighttime scene with artificial street lighting",
                "dark evening with illuminated buildings and street lights",
                "night photography with artificial lighting and darkness",
                "after dark scene with electric lights and dark sky",
                "nighttime urban scene with bright lights against dark background"
            ],
            'dawn_dusk': [
                "golden hour lighting with warm orange and pink sky",
                "sunrise or sunset with dramatic sky colors",
                "twilight scene with gradient sky from light to dark",
                "dawn or dusk with low sun angle and warm lighting",
                "golden hour photography with soft warm light"
            ]
        }

        self.severe_weather_categories = {
            'severe_rain': [
                "heavy torrential rain with water pouring down intensely",
                "severe rainstorm with visible raindrops falling rapidly",
                "extreme rainfall with rain drops covering the camera lens",
                "monsoon-like conditions with heavy downpour and flooding",
                "intense rain with water streaming down surfaces and puddles forming quickly"
            ],
            'severe_snow': [
                "heavy snowfall with thick snow accumulating on the ground",
                "blizzard conditions with snow falling heavily and wind blowing",
                "severe snowstorm with snow piling up rapidly on surfaces",
                "extreme winter weather with dense snow falling continuously",
                "intense snow with whiteout conditions and deep snow drifts"
            ],
            'severe_fog': [
                "dense fog with visibility less than 5 meters obscuring everything",
                "thick fog with zero visibility and objects barely visible at close range",
                "severe haze with extremely low visibility under 10 meters",
                "impenetrable fog covering the landscape with minimal sight distance",
                "heavy mist and fog with visibility reduced to just a few meters"
            ],
        }
        
        # Fog detection prompts for counter-prompt margin scoring
        self.fog_prompts = [
            "a photo of a dense ground-level fog obscuring distant objects in an outdoor scene",
            "a photo of heavy haze over a landscape with very low visibility",
            "a photo of thick fog covering the ground in a countryside scene",
        ]
        
        self.cloudy_prompts = [
            "a photo of an overcast cloudy sky with no fog near the ground",
            "a photo of thick clouds in the sky above a clear landscape",
        ]
        
        self.blurred_portrait_prompts = [
            "a photo of a blurred portrait of a person",
            "a photo of a defocused close-up of a face",
            "a photo of a blurred object"
        ]
        
        self.clear_prompts = [
            "a photo of a clear day with high visibility",
            "a photo of a crisp landscape with no haze or fog",
            "a photo of a clear blue sky"
        ]
        
        # Define counter-prompt classes for fog margin scoring
        self.fog_classes = {
            "fog": self.fog_prompts,
            "cloudy": self.cloudy_prompts,
            "blurred": self.blurred_portrait_prompts,
            "clear": self.clear_prompts,
        }
        
        # Multi-GPU setup
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)
            self.batch_size = batch_size * torch.cuda.device_count()
        
        self.model.eval()
        
        # Pre-compute text features - but store individual prompt features
        self._prepare_text_features()
    
    def _prepare_text_features(self):
        """Pre-compute text features for individual prompts (not averaged)"""
        logger.info("Pre-computing individual text features...")
        
        with torch.no_grad():
            # Outdoor/Indoor features - use best performing prompt
            outdoor_features = []
            for prompt in self.outdoor_prompts:
                tokens = clip.tokenize([prompt]).to(self.device)
                features = self.model.encode_text(tokens)
                # Normalize features to unit vectors
                features = F.normalize(features, p=2, dim=1)
                outdoor_features.append(features)
            
            indoor_features = []
            for prompt in self.indoor_prompts:
                tokens = clip.tokenize([prompt]).to(self.device)
                features = self.model.encode_text(tokens)
                features = F.normalize(features, p=2, dim=1)
                indoor_features.append(features)
            
            # Store all individual features
            self.outdoor_text_features = torch.cat(outdoor_features, dim=0)
            self.indoor_text_features = torch.cat(indoor_features, dim=0)
            
            # Weather category features - store individual prompts
            self.weather_text_features = {}
            for category, prompts in self.weather_categories.items():
                category_features = []
                for prompt in prompts:
                    tokens = clip.tokenize([prompt]).to(self.device)
                    features = self.model.encode_text(tokens)
                    features = F.normalize(features, p=2, dim=1)
                    category_features.append(features)
                
                # Store all prompt features for this category
                self.weather_text_features[category] = torch.cat(category_features, dim=0)
            
            # Severe weather category features - store individual prompts
            self.severe_weather_text_features = {}
            for category, prompts in self.severe_weather_categories.items():
                category_features = []
                for prompt in prompts:
                    tokens = clip.tokenize([prompt]).to(self.device)
                    features = self.model.encode_text(tokens)
                    features = F.normalize(features, p=2, dim=1)
                    category_features.append(features)
                
                # Store all prompt features for this category
                self.severe_weather_text_features[category] = torch.cat(category_features, dim=0)
            
            # Fog detection features for counter-prompt margin scoring
            self.fog_text_features = {}
            for cls, prompts in self.fog_classes.items():
                tokens = clip.tokenize(prompts).to(self.device)
                feats = self.model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                self.fog_text_features[cls] = feats.mean(dim=0, keepdim=True)
        
        logger.info("Individual text features prepared and normalized")
    
    def optimized_scan_with_scandir(self, source_folder: str, use_cache: bool = True,
                                   max_cache_age_hours: int = 24):
        """Ultra-fast scanning using os.scandir() with caching"""
        
        cache_manager = FolderScanCache()
        
        # Try cache first
        if use_cache:
            cached_files = cache_manager.load_scan_results(source_folder, max_cache_age_hours)
            if cached_files is not None:
                return cached_files
        
        # Perform optimized scan
        logger.info(f"Starting optimized scan of {source_folder}")
        start_time = time.time()
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
        image_files = []
        processed_count = 0
        
        # Use os.scandir() - much faster than iterdir()
        with os.scandir(source_folder) as entries:
            for entry in entries:
                if entry.is_file():
                    if any(entry.name.endswith(ext) for ext in extensions):
                        image_files.append(Path(entry.path))
                        processed_count += 1
                        
                        # Progress reporting every 100K files
                        if processed_count % 100000 == 0:
                            elapsed = time.time() - start_time
                            rate = processed_count / elapsed
                            logger.info(f"Scanned {processed_count:,} files in {elapsed:.1f}s ({rate:.0f} files/sec)")
                elif entry.is_dir():
                    # Recursively scan subdirectories
                    sub_files = self.optimized_scan_with_scandir(entry.path, use_cache=False)
                    image_files.extend(sub_files)
                    processed_count += len(sub_files)
                    
                    if processed_count % 100000 == 0:
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed
                        logger.info(f"Scanned {processed_count:,} files in {elapsed:.1f}s ({rate:.0f} files/sec)")
                        
        total_time = time.time() - start_time
        final_rate = len(image_files) / total_time if total_time > 0 else 0
        
        logger.info(f"Scan complete: {len(image_files):,} image files in {total_time:.1f} seconds")
        logger.info(f"Final scan rate: {final_rate:.0f} files/second")
        
        # Save to cache
        if use_cache:
            cache_manager.save_scan_results(source_folder, image_files)
        
        return image_files
    

    def parallel_process_to_lmdb(self, source_folder: str, output_dir: str,
                                batch_size: int = 10000, max_workers: int = None):
        """Parallel processing with optimized LMDB creation"""

        if max_workers is None:
            max_workers = min(mp.cpu_count(), 8)

        logger.info("Starting parallel LMDB creation...")

        # Get file list with optimized scanning
        all_files = self.optimized_scan_with_scandir(source_folder)

        if not all_files:
            logger.error("No image files found")
            return 0, None
        
        # Map file paths to sequential LMDB indices and persist mapping for later lookup
        logger.info("Mapping file paths to LMDB indices and saving mapping file...")
        path_index_map = {}
        for idx, p in enumerate(all_files):
            # Normalize to absolute path for robust lookup later
            path_index_map[str(Path(p).resolve())] = idx

        # Ensure output directory exists and write both JSON and pickle for convenience
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        map_json_path = output_path / "path_to_index.json"
        map_pickle_path = output_path / "path_to_index.pkl"

        try:
            with open(map_json_path, "w") as jf:
                json.dump(path_index_map, jf, indent=2)
            with open(map_pickle_path, "wb") as pf:
                pickle.dump(path_index_map, pf)
            logger.info(f"Wrote path->index mapping ({len(path_index_map):,} entries) to {map_json_path} and {map_pickle_path}")
        except Exception as e:
            logger.error(f"Failed to write path->index mapping: {e}")

        # Setup LMDB with realistic sizing
        lmdb_path = output_path / "sa1b_optimized.lmdb"

        # Realistic map_size: 64KB per image
        db_size = len(all_files) * 64 * 1024 * 2
        logger.info(f"Creating LMDB with map_size: {db_size / 1e9:.1f} GB")

        env = lmdb.open(str(lmdb_path), map_size=db_size)
        total_processed = 0

        # Process in parallel batches
        for i in range(0, len(all_files), batch_size):
            batch_files = all_files[i:i+batch_size]

            # Split batch among workers
            chunk_size = len(batch_files) // max_workers
            if chunk_size == 0:
                chunk_size = 1

            file_chunks = [batch_files[j:j+chunk_size] 
                          for j in range(0, len(batch_files), chunk_size)]

            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_files)} images")
            start_time = time.time()

            # Parallel processing
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                chunk_results = list(executor.map(process_file_chunk, file_chunks))

            # Flatten results and write to LMDB
            with env.begin(write=True) as txn:
                for chunk_result in chunk_results:
                    for item in chunk_result:
                        key = f"{total_processed:08d}".encode('ascii')
                        txn.put(key, item['data'])
                        total_processed += 1

            batch_time = time.time() - start_time
            speed = len(batch_files) / batch_time if batch_time > 0 else 0
            logger.info(f"Batch completed: {len(batch_files)} images in {batch_time:.1f}s ({speed:.1f} img/s)")

        env.close()

        # Verify database
        with lmdb.open(str(lmdb_path), readonly=True) as verify_env:
            with verify_env.begin() as txn:
                actual_entries = verify_env.stat()['entries']

        logger.info(f"LMDB creation complete: {actual_entries:,} entries verified")
        return actual_entries, str(lmdb_path)
    
    def classify_batch_gpu_improved(self, image_batch: torch.Tensor) -> List[Dict[str, Any]]:
        """Improved GPU-accelerated batch classification with fog detection and proper scoring"""
        with torch.no_grad():
            image_batch = image_batch.float().to(self.device)
            
            # Encode images and normalize
            image_features = self.model.encode_image(image_batch)
            image_features = F.normalize(image_features, p=2, dim=1)
            
            # Outdoor vs Indoor classification using max similarity across all prompts
            outdoor_similarities = torch.mm(image_features, self.outdoor_text_features.t())
            indoor_similarities = torch.mm(image_features, self.indoor_text_features.t())
            
            # Take maximum similarity across all prompts for each category
            outdoor_max_sim = outdoor_similarities.max(dim=1)[0]
            indoor_max_sim = indoor_similarities.max(dim=1)[0]
            
            is_outdoor = outdoor_max_sim > indoor_max_sim
            outdoor_confidence = torch.softmax(torch.stack([outdoor_max_sim, indoor_max_sim], dim=1), dim=1)[:, 0]
            
            # Weather classification for outdoor images
            results = []
            for i, img_features in enumerate(image_features):
                if not is_outdoor[i]:
                    results.append({
                        'category': 'indoor',
                        'confidence': float(outdoor_confidence[i]),
                        'all_scores': {'indoor': float(1 - outdoor_confidence[i])},
                        'is_outdoor': False,
                        'outdoor_confidence': float(outdoor_confidence[i]),
                        'fog_score': 0.0,
                        'fog_margin': -1.0,
                        'fog_confounder_score': 0.0,
                        'is_foggy': False
                    })
                    continue
                
                # First, perform fog detection using counter-prompt margin scoring
                fog_scores = {}
                for cls in ["fog", "cloudy", "blurred", "clear"]:
                    txt_feat = self.fog_text_features[cls]
                    score = torch.cosine_similarity(img_features.unsqueeze(0), txt_feat).item()
                    fog_scores[cls] = score
                
                # Margin logic: fog_score - max(confounder_scores)
                fog_score = fog_scores["fog"]
                confounder_score = max(fog_scores["cloudy"], fog_scores["blurred"], fog_scores["clear"])
                fog_margin = fog_score - confounder_score
                is_foggy = fog_margin >= self.fog_margin_threshold
                
                # Otherwise, perform regular weather classification
                category_max_similarities = {}
                category_raw_scores = {}
                
                # Choose weather features based on severe_weather_only flag
                weather_features = self.severe_weather_text_features if self.severe_weather_only else self.weather_text_features
                
                for category, category_features in weather_features.items():
                    # Compute similarity with all prompts in this category
                    similarities = torch.mm(img_features.unsqueeze(0), category_features.t())
                    # Take maximum similarity across prompts
                    max_sim = similarities.max().item()
                    category_max_similarities[category] = max_sim
                    category_raw_scores[category] = similarities.cpu().numpy().flatten()
                
                # Convert to probabilities using softmax for better calibration
                similarity_values = torch.tensor(list(category_max_similarities.values()))
                probabilities = torch.softmax(similarity_values * 10, dim=0)  # Temperature scaling
                
                category_names = list(category_max_similarities.keys())
                prob_scores = {cat: float(prob) for cat, prob in zip(category_names, probabilities)}
                
                # Find best category
                best_idx = probabilities.argmax().item()
                best_category = category_names[best_idx]
                best_confidence = float(probabilities[best_idx])
                
                # Calculate margin between top-1 and top-2
                sorted_probs = torch.sort(probabilities, descending=True)[0]
                margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

                results.append({
                    'category': best_category,
                    'confidence': best_confidence,
                    'is_foggy': is_foggy,
                    'all_scores': prob_scores,
                    'raw_similarities': fog_scores,
                    'margin': margin,
                    'fog_score': fog_score,
                    'fog_margin': fog_margin,
                    'fog_confounder_score': confounder_score,
                    'is_outdoor': True,
                    'outdoor_confidence': float(outdoor_confidence[i])
                })


            return results
    
    def setup_dataloader(self, lmdb_path: str) -> DataLoader:
        """Setup optimized DataLoader with CLIP preprocessing"""
        dataset = SA1BLMDBDataset(lmdb_path, preprocess_fn=self.preprocess)
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device == "cuda" else False,
            drop_last=False
        )
    
    def organize_and_export_results(self, all_results: List[Dict[str, Any]], 
                                   lmdb_path: str, output_dir: str,
                                   max_per_category: int = 50,
                                   confidence_threshold: float = 0.3,
                                   margin_threshold: float = 0.1):
        """Organize results with improved confidence and margin thresholds"""
        # Try to load a path->index mapping (index is value) so we can include original filenames
        mapping = {}
        try:
            map_json = Path(output_dir) / "path_to_index.json"
            if map_json.exists():
                with open(map_json, 'r') as mf:
                    mapping = json.load(mf)
                    logger.info(f"Loaded path->index mapping from {map_json} ({len(mapping):,} entries)")
            else:
                logger.warning(f"No mapping file found at {map_json}")
            # invert mapping to index->path for fast lookup
            index_to_path = {str(v): k for k, v in mapping.items()} if mapping else {}
        except Exception as e:
            logger.warning(f"Could not load path->index mapping from {output_dir}: {e}")
            index_to_path = {}
        
        # Organize by category with improved filtering
        categorized_images = defaultdict(list)
        uncategorized = []
        for result in all_results:
            if (result['is_outdoor'] and
                (result['confidence'] > confidence_threshold and
                 result.get('margin', 0) > margin_threshold)):
                category = result['category']
                categorized_images[category].append(result)
            else:
                uncategorized.append(result)

        # Log distribution
        logger.info("Images per category (after filtering):")
        total_fog_count = 0
        for category, items in categorized_images.items():
            if items:
                avg_confidence = np.mean([item['confidence'] for item in items])
                avg_margin = np.mean([item.get('margin', 0) for item in items])
                fog_count_in_category = sum(1 for item in items if item.get('is_foggy', False))
                total_fog_count += fog_count_in_category
                logger.info(f"  {category}: {len(items)} images (avg conf: {avg_confidence:.3f}, avg margin: {avg_margin:.3f}, fog: {fog_count_in_category})")
        logger.info(f"Total fog images across all categories: {total_fog_count}")
        
        # Balance and export
        output_path = Path(f'{output_dir}/categories')
        output_path.mkdir(exist_ok=True)
        dataset_stats = {}
        
        env = lmdb.open(lmdb_path, readonly=True)
        
        for category, items in categorized_images.items():
            if len(items) > 0:
                # Sort by confidence * margin for best quality examples
                items.sort(key=lambda x: x['confidence'] * x.get('margin', 0), reverse=True)
                if max_per_category > 0:
                    selected_items = items[:min(len(items), max_per_category)]
                else:
                    selected_items = items

                category_dir = output_path / category
                category_dir.mkdir(exist_ok=True)
                fog_dir = output_path / 'foggy'
                fog_dir.mkdir(exist_ok=True)

                valid_count = 0
                for i, item in enumerate(selected_items):
                    try:
                        # Load and save image
                        with env.begin() as txn:
                            key = f"{item['original_index']:08d}".encode('ascii')
                            img_bytes = txn.get(key)
                            
                            if img_bytes:
                                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                                # Prefer original filename if available in mapping, otherwise fallback
                                # try:
                                resolved_orig_path = index_to_path.get(str(item['original_index']))
                                # except Exception:
                                #     resolved_orig_path = None

                                if resolved_orig_path:
                                    orig_name = Path(resolved_orig_path).name
                                    target_path = category_dir / orig_name
                                    # Avoid overwriting if filename already exists in this category
                                    if target_path.exists():
                                        base = Path(orig_name).stem
                                        ext = Path(orig_name).suffix or ".jpg"
                                        target_path = category_dir / f"{base}_{item['original_index']:08d}{ext}"
                                else:
                                    logger.warning(f"No original path found for index {item['original_index']}, using default naming")
                                    target_path = category_dir / f"{category}_{i:06d}.jpg"

                                cv2.imwrite(str(target_path), img)

                                # Save enhanced metadata with fog detection info
                                # Resolve original path and filename if mapping exists
                                resolved_path = None
                                resolved_fname = None
                                try:
                                    resolved_path = index_to_path.get(str(item['original_index']))
                                    if resolved_path:
                                        resolved_fname = Path(resolved_path).name
                                except Exception:
                                    resolved_path = None
                                    resolved_fname = None

                                metadata = {
                                    'confidence': item['confidence'],
                                    'margin': item.get('margin', 0),
                                    'all_scores': item['all_scores'],
                                    'raw_similarities': item.get('raw_similarities', {}),
                                    'category': category,
                                    'is_outdoor': item['is_outdoor'],
                                    'outdoor_confidence': item['outdoor_confidence'],
                                    'fog_score': item.get('fog_score', 0.0),
                                    'fog_margin': item.get('fog_margin', -1.0),
                                    'fog_confounder_score': item.get('fog_confounder_score', 0.0),
                                    'is_foggy': item.get('is_foggy', False),
                                    'original_index': item['original_index']
                                }

                                # if resolved_fname:
                                    # metadata['original_filename'] = resolved_fname


                                metadata_path = target_path.with_suffix('.json')
                                with open(metadata_path, 'w') as f:
                                    json.dump(metadata, f, indent=2)

                                if item.get('is_foggy', False):
                                    # Save copy in fog directory with unique filename
                                    fog_count = len([p for p in fog_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
                                    fog_img_path = fog_dir / resolved_fname if resolved_fname else fog_dir / f"foggy_{fog_count:06d}.jpg"
                                    cv2.imwrite(str(fog_img_path), img)

                                    # Save fog metadata with unique filename
                                    # Save fog metadata using the saved fog image's filename
                                    try:
                                        fog_metadata = metadata.copy()
                                        fog_metadata['original_category'] = category
                                        fog_metadata['fog_category'] = 'foggy'
                                        # Use the image filename (fog_img_path) and replace its extension with .json
                                        fog_metadata_path = fog_img_path.with_suffix('.json')
                                        with open(fog_metadata_path, 'w') as f:
                                            json.dump(fog_metadata, f, indent=2)
                                    except Exception as e:
                                        logger.error(f"Failed to write fog metadata for {fog_img_path}: {e}")

                                valid_count += 1
                    except Exception as e:
                        logger.error(f"Error saving {category}_{i}: {e}")
                
                dataset_stats[category] = valid_count
                dataset_stats['foggy'] = dataset_stats.get('foggy', 0) + sum(1 for item in selected_items if item.get('is_foggy', False))
                logger.info(f"Exported {valid_count} high-quality images for {category}")
                
        store_uncategorized = True
        if store_uncategorized:
            # Save uncategorized images in a separate folder
            uncategorized_dir = output_path / 'uncategorized'
            uncategorized_dir.mkdir(exist_ok=True)
            for j, item in enumerate(uncategorized):
                try:
                    with env.begin() as txn:
                        key = f"{item['original_index']:08d}".encode('ascii')
                        img_bytes = txn.get(key)
                        
                        if img_bytes:
                            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                            resolved_orig_path = index_to_path.get(str(item['original_index']))
                            if resolved_orig_path:
                                orig_name = Path(resolved_orig_path).name
                                target_path = uncategorized_dir / orig_name
                                if target_path.exists():
                                    base = Path(orig_name).stem
                                    ext = Path(orig_name).suffix or ".jpg"
                                    target_path = uncategorized_dir / f"{base}_{item['original_index']:08d}{ext}"
                            else:
                                target_path = uncategorized_dir / f"uncategorized_{j:06d}.jpg"

                            cv2.imwrite(str(target_path), img)

                            # Save metadata
                            metadata = {
                                'confidence': item['confidence'],
                                'margin': item.get('margin', 0),
                                'all_scores': item['all_scores'],
                                'category': 'uncategorized',
                                'is_outdoor': item['is_outdoor'],
                                'outdoor_confidence': item['outdoor_confidence'],
                                'original_index': item['original_index']
                            }

                            if resolved_orig_path:
                                metadata['original_filename'] = Path(resolved_orig_path).name

                            metadata_path = target_path.with_suffix('.json')
                            with open(metadata_path, 'w') as f:
                                json.dump(metadata, f, indent=2)
                except Exception as e:
                    logger.error(f"Error saving uncategorized_{j}: {e}")
            logger.info(f"Exported {len(uncategorized)} uncategorized images")  
        
        env.close()
        
        # Count fog images across all categories
        fog_count = len([p for p in (output_path / 'foggy').iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]) if (output_path / 'foggy').exists() else 0
        
        # Save enhanced statistics
        stats_summary = {
            'total_categories': len(dataset_stats),
            'total_images': sum(dataset_stats.values()),
            'total_fog_images': fog_count,
            'confidence_threshold': confidence_threshold,
            'margin_threshold': margin_threshold,
            'category_counts': dataset_stats,
            'fog_margin_threshold': self.fog_margin_threshold,
            'creation_date': (f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'),
        }

        with open(output_path / "dataset_stats.json", 'w') as f:
            json.dump(stats_summary, f, indent=2)
        
        return dataset_stats
    
    def run_complete_pipeline(self, source_folder: str, output_dir: str,
                             max_per_category: int = 50,
                             confidence_threshold: float = 0.3,
                             margin_threshold: float = 0.1,
                             force_convert: bool = False,
                             use_cache: bool = True,
                             severe_weather_only: bool = False) -> Dict[str, int]:
        """Run the complete improved pipeline with fog detection"""
        
        self.severe_weather_only = severe_weather_only
        
        logger.info("=== IMPROVED SA-1B PIPELINE WITH FOG DETECTION ===")
        logger.info(f"Source: {source_folder}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        logger.info(f"Margin threshold: {margin_threshold}")
        logger.info(f"Fog margin threshold: {self.fog_margin_threshold}")
        logger.info(f"Severe weather only: {severe_weather_only}")
        
        try:
            # Step 1: Create or verify LMDB
            lmdb_path = Path(output_dir) / "sa1b_optimized.lmdb"
            
            if not lmdb_path.exists() or force_convert:
                logger.info("Step 1: Creating LMDB with parallel processing...")
                processed_count, lmdb_path_str = self.parallel_process_to_lmdb(
                    source_folder, output_dir
                )
                # Ensure lmdb_path_str is a string for downstream callers
                lmdb_path_str = str(lmdb_path_str) if lmdb_path_str is not None else str(lmdb_path)
                
                if processed_count == 0:
                    logger.error("No images were processed")
                    return {}
                
                logger.info(f"Successfully processed {processed_count:,} images to LMDB")
            else:
                lmdb_path_str = str(lmdb_path)
                logger.info("Using existing LMDB database")
            lmdb_only = False
            if lmdb_only:
                logger.info("LMDB creation complete. Exiting as per lmdb_only flag.")
                return {}
            
            # Step 2: Setup DataLoader with CLIP preprocessing
            logger.info("Step 2: Setting up DataLoader with CLIP preprocessing...")
            dataloader = self.setup_dataloader(lmdb_path_str)
            
            logger.info("Step 3: Starting improved classification pipeline...")
            all_results = []
            start_time = time.time()
            
            for batch_idx, (images, indices) in enumerate(dataloader):
                valid_mask = [img is not None for img in images]
                if not any(valid_mask):
                    continue
                
                valid_images = torch.stack([img for img, valid in zip(images, valid_mask) if valid])
                valid_indices = [idx for idx, valid in zip(indices, valid_mask) if valid]
                
                # Use improved classification method
                batch_results = self.classify_batch_gpu_improved(valid_images)
                
                for result, idx in zip(batch_results, valid_indices):
                    result['original_index'] = idx.item()
                    all_results.append(result)
                
                if batch_idx % 50 == 0:
                    elapsed = time.time() - start_time
                    images_processed = len(all_results)
                    speed = images_processed / elapsed if elapsed > 0 else 0
                    
                    # Log confidence statistics including fog detection
                    outdoor_results = [r for r in all_results if r['is_outdoor']]
                    fog_results = [r for r in all_results if r['category'] == 'foggy']
                    if outdoor_results:
                        confidences = [r['confidence'] for r in outdoor_results]
                        margins = [r.get('margin', 0) for r in outdoor_results]
                        fog_margins = [r.get('fog_margin', -1) for r in outdoor_results]
                        avg_conf = np.mean(confidences)
                        avg_margin = np.mean(margins)
                        avg_fog_margin = np.mean([m for m in fog_margins if m > -1])
                        logger.info(f"Batch {batch_idx}: {images_processed} images processed, "
                                  f"{speed:.1f} img/sec, avg_conf: {avg_conf:.3f}, avg_margin: {avg_margin:.3f}")
            
            # Step 3: Organize and export results with improved thresholds
            logger.info("Step 4: Organizing and exporting results with quality filtering...")
            final_stats = self.organize_and_export_results(
                all_results, lmdb_path_str, output_dir, max_per_category, 
                confidence_threshold, margin_threshold
            )
            
            # Final summary with quality metrics including fog detection
            total_time = time.time() - start_time
            total_images = len(all_results)
            outdoor_images = len([r for r in all_results if r['is_outdoor']])
            fog_images = len([r for r in all_results if r['category'] == 'foggy'])
            high_quality_images = sum(final_stats.values()) if final_stats else 0
            
            # Calculate average confidence and margin for exported images
            exported_results = [r for r in all_results 
                              if r['is_outdoor'] and r['confidence'] > confidence_threshold 
                              and r.get('margin', 0) > margin_threshold or r['category'] == 'foggy' and r.get('fog_margin', -1) >= self.fog_margin_threshold]
            
            if exported_results:
                avg_confidence = np.mean([r['confidence'] for r in exported_results])
                avg_margin = np.mean([r.get('margin', 0) for r in exported_results])
                fog_exported = len([r for r in exported_results if r['category'] == 'foggy'])
            else:
                avg_confidence = 0
                avg_margin = 0
                fog_exported = 0
            
            logger.info("=" * 50)
            logger.info("IMPROVED PIPELINE WITH FOG DETECTION COMPLETE")
            logger.info(f"Total processing time: {total_time:.1f} seconds")
            logger.info(f"Total images processed: {total_images:,}")
            logger.info(f"Outdoor images detected: {outdoor_images:,}")
            logger.info(f"High-quality images exported: {high_quality_images:,}")
            logger.info(f"Processing speed: {total_images/total_time:.1f} images/second")
            logger.info(f"Average confidence of exported images: {avg_confidence:.3f}")
            logger.info(f"Average margin of exported images: {avg_margin:.3f}")
            logger.info(f"Fog margin threshold used: {self.fog_margin_threshold}")
            logger.info("Category distribution:")
            for category, count in final_stats.items():
                logger.info(f"  {category}: {count} images")
            logger.info("=" * 50)
            
            return final_stats
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

def main():
    """SWIFT CLI - Scalable Weather Image Filtering Toolkit"""
    parser = argparse.ArgumentParser(
        description="SWIFT - Scalable Weather Image Filtering Toolkit",
        epilog="Example: python swift.py /path/to/images /path/to/output --batch_size 128"
    )
    parser.add_argument("source_folder", help="Path to source folder containing images")
    parser.add_argument("output_dir", help="Output directory for classified images")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for GPU processing (default: 64)")
    parser.add_argument("--max_per_category", type=int, default=0, help="Maximum images per weather category (0 = unlimited)")
    parser.add_argument("--confidence_threshold", type=float, default=0.2, 
                       help="Minimum confidence threshold for filtering (default: 0.2)")
    parser.add_argument("--margin_threshold", type=float, default=0.03, 
                       help="Minimum margin between top-1 and top-2 categories for filtering (default: 0.03)")
    parser.add_argument("--fog_margin_threshold", type=float, default=0.01, 
                       help="Margin threshold for fog detection vs confounders (default: 0.01)")
    parser.add_argument("--force_convert", action="store_true", help="Force LMDB conversion even if exists")
    parser.add_argument("--no_cache", action="store_true", help="Disable scanning cache")
    parser.add_argument("--severe_weather_only", action="store_true", 
                        help="Only classify into severe weather categories (fog, rain, snow, clear)")
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.source_folder).exists():
        logger.error(f"Source folder does not exist: {args.source_folder}")
        sys.exit(1)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize SWIFT pipeline
    pipeline = SWIFTPipeline(
        batch_size=args.batch_size, 
        fog_margin_threshold=args.fog_margin_threshold
    )
    
    # Run pipeline
    try:
        results = pipeline.run_complete_pipeline(
            source_folder=args.source_folder,
            output_dir=args.output_dir,
            max_per_category=args.max_per_category,
            confidence_threshold=args.confidence_threshold,
            margin_threshold=args.margin_threshold,
            force_convert=args.force_convert,
            use_cache=not args.no_cache,
            severe_weather_only=args.severe_weather_only
        )
        
        logger.info("SWIFT pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# Alias for backward compatibility
ImprovedSA1BPipeline = SWIFTPipeline

if __name__ == "__main__":
    main()