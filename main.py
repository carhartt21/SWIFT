# Combined Pipeline for LAION and SA-1B Dataset Filtering
# Supports both text-based (LAION) and image-only (SA-1B) filtering

import requests
import pandas as pd
import numpy as np
import torch
import clip
import json
import os
from PIL import Image
import io
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
import random

class StreamingSA1BDataset(Dataset):
    def __init__(self, lmdb_path, cache_size=1000):
        self.lmdb_path = lmdb_path
        self.cache_size = cache_size
        self.cache = {}
        self.cache_order = []
        
    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
        
        # Load from LMDB
        with lmdb.open(str(self.lmdb_path), readonly=True) as env:
            with env.begin() as txn:
                key = f"{idx:08d}".encode('ascii')
                img_bytes = txn.get(key)
                
                if img_bytes:
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    
                    # Cache management
                    if len(self.cache) >= self.cache_size:
                        oldest = self.cache_order.pop(0)
                        del self.cache[oldest]
                    
                    self.cache[idx] = img
                    self.cache_order.append(idx)
                    
                    return img
        
        return None


class OptimizedSA1BPipeline:
    def __init__(self, batch_size=32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.batch_size = batch_size
        
        # Enable mixed precision for faster inference
        self.model.half()  # Use FP16
        
        # Prepare text features once (cache them)
        self.outdoor_text_features = self._prepare_text_features(self.outdoor_prompts)
        self.indoor_text_features = self._prepare_text_features(self.indoor_prompts)
        self.weather_text_features = self._prepare_weather_features()
    
    def _prepare_text_features(self, prompts):
        """Pre-compute and cache text features"""
        with torch.no_grad():
            text_tokens = clip.tokenize(prompts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            return text_features.mean(dim=0, keepdim=True)
    
    def _prepare_weather_features(self):
        """Pre-compute weather category features"""
        weather_features = {}
        for category, prompts in self.weather_categories.items():
            text_tokens = clip.tokenize(prompts).to(self.device)
            features = self.model.encode_text(text_tokens)
            weather_features[category] = features.mean(dim=0, keepdim=True)
        return weather_features
    
    def classify_batch_gpu(self, image_batch):
        """GPU-accelerated batch classification"""
        with torch.no_grad():
            # Move batch to GPU
            if isinstance(image_batch, list):
                image_tensors = torch.stack([
                    self.preprocess(img).half() for img in image_batch
                ]).to(self.device)
            else:
                image_tensors = image_batch.half().to(self.device)
            
            # Batch encode images
            image_features = self.model.encode_image(image_tensors)
            
            # Outdoor classification
            outdoor_similarities = torch.cosine_similarity(
                image_features, self.outdoor_text_features
            )
            indoor_similarities = torch.cosine_similarity(
                image_features, self.indoor_text_features
            )
            
            is_outdoor = outdoor_similarities > indoor_similarities
            
            # Weather classification for outdoor images
            weather_results = []
            for i, (img_features, outdoor_flag) in enumerate(zip(image_features, is_outdoor)):
                if outdoor_flag:
                    best_category = None
                    best_score = -1
                    
                    for category, category_features in self.weather_text_features.items():
                        similarity = torch.cosine_similarity(
                            img_features.unsqueeze(0), category_features
                        ).item()
                        
                        if similarity > best_score:
                            best_score = similarity
                            best_category = category
                    
                    weather_results.append({
                        'category': best_category,
                        'confidence': float(best_score),
                        'is_outdoor': True
                    })
                else:
                    weather_results.append({
                        'category': None,
                        'confidence': 0.0,
                        'is_outdoor': False
                    })
            
            return weather_results

    def process_dataloader(self, dataloader):
        """Process entire dataset using DataLoader"""
        all_results = []
        
        for batch_idx, (images, indices) in enumerate(dataloader):
            # Process batch on GPU
            batch_results = self.classify_batch_gpu(images)
            
            # Add metadata
            for i, result in enumerate(batch_results):
                result['original_index'] = indices[i].item()
                all_results.append(result)
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(dataloader)}")
        
        return all_results


class UltraFastSA1BPipeline:
    def __init__(self, lmdb_path, batch_size=64, num_workers=8):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.half()  # FP16 optimization
        
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            batch_size *= torch.cuda.device_count()
        
        # Setup optimized DataLoader
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        dataset = SA1BLMDBDataset(lmdb_path, transform=transform)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        # Pre-compute text features
        self._prepare_text_features()
    
    def run_optimized_pipeline(self, output_dir="ultra_fast_sa1b"):
        """Run the complete optimized pipeline"""
        print("Starting ultra-fast SA-1B processing...")
        
        all_results = []
        start_time = time.time()
        
        for batch_idx, (images, indices) in enumerate(self.dataloader):
            batch_results = self.classify_batch_gpu(images)
            
            for i, result in enumerate(batch_results):
                result['original_index'] = indices[i].item()
                all_results.append(result)
            
            # Progress tracking
            if batch_idx % 50 == 0:
                elapsed = time.time() - start_time
                images_processed = (batch_idx + 1) * self.dataloader.batch_size
                speed = images_processed / elapsed
                print(f"Batch {batch_idx}: {images_processed} images, {speed:.1f} img/sec")
        
        # Organize and export results
        organized_results = self.organize_results(all_results)
        self.export_results(organized_results, output_dir)
        
        total_time = time.time() - start_time
        total_images = len(all_results)
        print(f"Processing complete: {total_images} images in {total_time:.1f}s ({total_images/total_time:.1f} img/sec)")
        
        return organized_results

# Usage
if __name__ == "__main__":
    # Step 1: Convert SA-1B to LMDB (one-time setup)
    # parallel_lmdb_conversion("/path/to/sa1b", "sa1b_optimized.lmdb")
    
    # Step 2: Run optimized pipeline
    pipeline = UltraFastSA1BPipeline(
        lmdb_path="sa1b_optimized.lmdb",
        batch_size=128,  # Adjust based on GPU memory
        num_workers=12   # Adjust based on CPU cores
    )
    
    results = pipeline.run_optimized_pipeline()


class UnifiedOutdoorWeatherPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Outdoor classification prompts
        self.outdoor_prompts = [
            "outdoor scene", "street scene", "road", "highway", "parking lot",
            "building exterior", "urban scene", "city street", "outdoor view",
            "natural outdoor landscape", "exterior view", "traffic scene"
        ]
        
        self.indoor_prompts = [
            "indoor scene", "inside building", "interior room", "indoor space",
            "office interior", "home interior", "indoor environment", "enclosed space"
        ]
        
        # Weather/lighting categories
        self.weather_categories = {
            'clear_day': [
                "clear sunny day", "bright sunny weather", "clear blue sky",
                "sunny outdoor scene", "bright daylight"
            ],
            'cloudy': [
                "cloudy weather", "overcast sky", "cloudy outdoor scene",
                "gray cloudy day", "partly cloudy"
            ],
            'rainy': [
                "rainy weather", "rain on street", "wet rainy scene",
                "rainy day outdoor", "rain puddles"
            ],
            'snowy': [
                "snowy weather", "snow covered scene", "winter snow",
                "snowy outdoor scene", "snow on ground"
            ],
            'foggy': [
                "foggy weather", "misty scene", "fog outdoor",
                "hazy weather", "low visibility fog"
            ],
            'night': [
                "night scene", "nighttime outdoor", "dark evening",
                "night street lighting", "after dark"
            ],
            'dawn_dusk': [
                "dawn lighting", "dusk scene", "golden hour",
                "sunrise outdoor", "sunset lighting"
            ]
        }
        
        # Outdoor keywords for text filtering (LAION only)
        self.outdoor_keywords = [
            'outdoor', 'street', 'road', 'highway', 'traffic', 'parking',
            'landscape', 'building', 'city', 'urban', 'rural', 'nature',
            'sky', 'weather', 'rain', 'snow', 'fog', 'sunny', 'cloudy'
        ]

    # LAION-specific methods
    def load_laion_dataset(self, dataset_name="laion400m"):
        """Load LAION dataset metadata"""
        if dataset_name == "laion400m":
            return load_dataset("laion/laion400m", streaming=True, split='train')
        else:
            raise ValueError(f"Unsupported LAION dataset: {dataset_name}")

    def prefilter_by_text(self, df, outdoor_keywords):
        """Filter using text captions for outdoor relevance (LAION only)"""
        pattern = '|'.join(self.outdoor_keywords)
        
        # Map common caption column names
        caption_mapping = {
            'TEXT': 'TEXT',           # LAION datasets
            'caption': 'caption',     # Common format
            'text': 'text',           # Alternative format
            'description': 'description',
            'alt_text': 'alt_text'
        }
        
        caption_col = None
        for col_name in caption_mapping.keys():
            if col_name in df.columns:
                caption_col = col_name
                break
        
        if caption_col is None:
            available_cols = df.columns.tolist()
            print(f"Warning: No caption column found. Available columns: {available_cols}")
            return df
        
        print(f"Filtering using column: {caption_col}")
        outdoor_mask = df[caption_col].fillna('').str.contains(pattern, case=False, na=False)
        filtered_df = df[outdoor_mask]
        
        print(f"Filtered from {len(df)} to {len(filtered_df)} images ({len(filtered_df)/len(df)*100:.1f}%)")
        return filtered_df

    def download_and_validate_image(self, url, min_size=(256, 256)):
        """Download image with quality checks (LAION only)"""
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(io.BytesIO(response.content))
            
            if img.size[0] < min_size[0] or img.size[1] < min_size[1]:
                return None
            if img.mode not in ['RGB', 'RGBA']:
                return None
                
            return img.convert('RGB')
        except:
            return None

    def batch_download(self, url_list, max_workers=32):
        """Parallel download with quality filtering (LAION only)"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.download_and_validate_image, url_list))
        return [img for img in results if img is not None]

    # SA-1B specific methods
    def load_sa1b_dataset(self, data_folder, max_images=1000):
        """Load SA-1B dataset from local folder"""
        data_folder = Path(data_folder)
        
        # Get image files
        # image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        print(f"Loading images from {data_folder}...")
        # Use os.scandir for faster directory traversal (especially with many files)
        # Parallelize file discovery for large directories
        def is_image_file(entry):
            return entry.is_file() and entry.name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

        with ThreadPoolExecutor() as executor:
            entries = list(os.scandir(data_folder))
            image_entries = list(executor.map(is_image_file, entries))
            image_files = [Path(entry.path) for entry, is_img in zip(entries, image_entries) if is_img]
        
        print(f"Found {len(image_files)} images in SA-1B dataset")
        print(f"Loading {max_images} images...")
        image_files = image_files[:max_images]
        
        images_data = []
        for img_path in tqdm(image_files, desc="Loading SA-1B images"):
            try:
                img = Image.open(img_path).convert('RGB')
                # Resize large SA-1B images for processing efficiency
                if max(img.size) > 1024:
                    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                
                images_data.append({
                    'image': img,
                    'path': str(img_path),
                    'filename': img_path.name
                })
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        return images_data

    # Unified classification methods
    def classify_outdoor_batch(self, images, threshold=0.3, dataset_type="laion"):
        """Classify images as outdoor/indoor using CLIP"""
        outdoor_scores = []
        
        # Handle different input formats
        if dataset_type == "sa1b":
            actual_images = [item['image'] for item in images]
        else:
            actual_images = images
        
        with torch.no_grad():
            outdoor_text = clip.tokenize(self.outdoor_prompts).to(self.device)
            indoor_text = clip.tokenize(self.indoor_prompts).to(self.device)
            
            for i, img in enumerate(actual_images):
                img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(img_tensor)
                
                outdoor_features = self.model.encode_text(outdoor_text)
                indoor_features = self.model.encode_text(indoor_text)
                
                outdoor_sim = torch.cosine_similarity(
                    image_features, outdoor_features.mean(dim=0, keepdim=True)
                ).item()
                indoor_sim = torch.cosine_similarity(
                    image_features, indoor_features.mean(dim=0, keepdim=True)
                ).item()
                
                is_outdoor = outdoor_sim > indoor_sim
                outdoor_scores.append((is_outdoor, outdoor_sim, indoor_sim))
                
                if i < 10:  # Debug first 10
                    print(f"Image {i}: Outdoor={outdoor_sim:.3f}, Indoor={indoor_sim:.3f}, Classified={'Outdoor' if is_outdoor else 'Indoor'}")
        
        outdoor_count = sum(1 for score in outdoor_scores if score[0])
        print(f"Classification summary: {outdoor_count}/{len(outdoor_scores)} images classified as outdoor")
        
        return outdoor_scores

    def classify_weather_batch(self, images, dataset_type="laion"):
        """Classify weather conditions for batch of images"""
        results = []
        
        # Handle different input formats
        if dataset_type == "sa1b":
            actual_images = [item['image'] for item in images]
        else:
            actual_images = images
        
        with torch.no_grad():
            # Prepare all text prompts
            all_prompts = []
            category_mapping = []
            
            for category, prompts in self.weather_categories.items():
                for prompt in prompts:
                    all_prompts.append(prompt)
                    category_mapping.append(category)
            
            text_tokens = clip.tokenize(all_prompts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            
            for i, img in enumerate(actual_images):
                img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(img_tensor)
                
                similarities = torch.cosine_similarity(
                    image_features, text_features
                ).cpu().numpy()
                
                # Group by category and take max
                category_scores = {}
                for j, category in enumerate(category_mapping):
                    if category not in category_scores:
                        category_scores[category] = []
                    category_scores[category].append(similarities[j])
                
                final_scores = {cat: max(scores) 
                              for cat, scores in category_scores.items()}
                best_category = max(final_scores, key=final_scores.get)
                confidence = final_scores[best_category]
                
                results.append({
                    'category': best_category,
                    'confidence': float(confidence),
                    'all_scores': {k: float(v) for k, v in final_scores.items()}
                })
                
                if i < 10:  # Debug first 10
                    print(f"Image {i}: Best={best_category} (conf={confidence:.3f})")
        
        # Summary statistics
        confidence_stats = [r['confidence'] for r in results]
        print(f"Weather classification summary:")
        print(f"  Average confidence: {np.mean(confidence_stats):.3f}")
        print(f"  Min confidence: {min(confidence_stats):.3f}")
        print(f"  Max confidence: {max(confidence_stats):.3f}")
        
        return results

    def organize_by_weather(self, images, weather_results, min_per_category=50, 
                           confidence_threshold=0.2, dataset_type="laion"):
        """Organize images by weather category and balance dataset"""
        categorized_images = defaultdict(list)
        
        print(f"Organizing {len(images)} images with confidence threshold {confidence_threshold}")
        
        for i, (img, result) in enumerate(zip(images, weather_results)):
            if result['confidence'] > confidence_threshold:
                category = result['category']
                
                if dataset_type == "sa1b":
                    categorized_images[category].append({
                        'image': img['image'],
                        'confidence': result['confidence'],
                        'scores': result['all_scores'],
                        'path': img['path'],
                        'filename': img['filename']
                    })
                else:
                    categorized_images[category].append({
                        'image': img,
                        'confidence': result['confidence'],
                        'scores': result['all_scores']
                    })
        
        # Debug: Print how many images per category
        print("Images per category after confidence filtering:")
        total_filtered = 0
        for category, items in categorized_images.items():
            print(f"  {category}: {len(items)} images")
            total_filtered += len(items)
        
        print(f"Total images after confidence filtering: {total_filtered}/{len(images)}")
        
        if total_filtered == 0:
            print("WARNING: No images passed confidence threshold!")
            return {}
        
        # Sort by confidence within each category
        for category in categorized_images:
            categorized_images[category].sort(
                key=lambda x: x['confidence'], reverse=True
            )
        
        # Balance dataset
        balanced_dataset = {}
        for category, items in categorized_images.items():
            if len(items) > 0:
                take_count = min(len(items), min_per_category)
                balanced_dataset[category] = items[:take_count]
                print(f"Taking {take_count} images for {category}")
        
        return balanced_dataset

    def validate_and_export(self, organized_dataset, output_dir="filtered_dataset", dataset_type="laion"):
        """Final validation and export of organized dataset"""
        Path(output_dir).mkdir(exist_ok=True)
        dataset_stats = {}
        
        for category, items in organized_dataset.items():
            category_dir = Path(output_dir) / category
            category_dir.mkdir(exist_ok=True)
            
            valid_count = 0
            for i, item in enumerate(items):
                try:
                    # Save image
                    img_path = category_dir / f"{category}_{i:06d}.jpg"
                    item['image'].save(img_path, 'JPEG', quality=95)
                    
                    # Prepare metadata based on dataset type
                    if dataset_type == "sa1b":
                        metadata = {
                            'confidence': float(item['confidence']),
                            'all_scores': {k: float(v) for k, v in item['scores'].items()},
                            'category': category,
                            'original_path': item['path'],
                            'original_filename': item['filename'],
                            'dataset_type': 'sa1b'
                        }
                    else:
                        metadata = {
                            'confidence': float(item['confidence']),
                            'all_scores': {k: float(v) for k, v in item['scores'].items()},
                            'category': category,
                            'dataset_type': 'laion'
                        }
                    
                    metadata_path = category_dir / f"{category}_{i:06d}.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    valid_count += 1
                    
                except Exception as e:
                    print(f"Error saving {category}_{i}: {e}")
            
            dataset_stats[category] = valid_count
            print(f"Successfully saved {valid_count} images for {category}")
        
        # Save overall statistics
        with open(Path(output_dir) / "dataset_stats.json", 'w') as f:
            json.dump(dataset_stats, f, indent=2)
        
        return dataset_stats

    # Unified pipeline methods
    def run_laion_pipeline(self, dataset_name="laion400m", batch_size=1000, 
                          target_per_category=50, output_dir="laion_filtered"):
        """Execute LAION filtering pipeline"""
        
        print("=== RUNNING LAION PIPELINE ===")
        print("Stage 1: Loading LAION metadata...")
        dataset = self.load_laion_dataset(dataset_name)
        
        print("Stage 2: Text-based pre-filtering...")
        sample_data = []
        for i, item in enumerate(dataset):
            if i >= batch_size * 10:
                break
            sample_data.append(item)
        
        if not sample_data:
            print("No data loaded from LAION dataset")
            return {}, {}
        
        df = pd.DataFrame(sample_data)
        filtered_metadata = self.prefilter_by_text(df, self.outdoor_keywords)
        
        print("Stage 3: Downloading images...")
        urls = filtered_metadata['URL'].tolist()[:batch_size]
        images = self.batch_download(urls)
        print(f"Downloaded {len(images)} valid images")
        
        if len(images) == 0:
            return {}, {}
        
        print("Stage 4: Outdoor classification...")
        outdoor_results = self.classify_outdoor_batch(images, dataset_type="laion")
        outdoor_images = [images[i] for i, (is_outdoor, _, _) in enumerate(outdoor_results) if is_outdoor]
        print(f"Found {len(outdoor_images)} outdoor images")
        
        if len(outdoor_images) == 0:
            return {}, {}
        
        print("Stage 5: Weather classification...")
        weather_results = self.classify_weather_batch(outdoor_images, dataset_type="laion")
        
        print("Stage 6: Organization and balancing...")
        organized_dataset = self.organize_by_weather(
            outdoor_images, weather_results, target_per_category, dataset_type="laion"
        )
        
        print("Stage 7: Export...")
        final_stats = self.validate_and_export(organized_dataset, output_dir, dataset_type="laion")
        
        return organized_dataset, final_stats

    def run_sa1b_pipeline(self, data_folder, max_images=1000, 
                         target_per_category=50, output_dir="sa1b_filtered"):
        """Execute SA-1B filtering pipeline"""
        
        print("=== RUNNING SA-1B PIPELINE ===")
        print("Stage 1: Loading SA-1B images...")
        images_data = self.load_sa1b_dataset(data_folder, max_images)
        print(f"Loaded {len(images_data)} images from SA-1B")
        
        if len(images_data) == 0:
            return {}, {}
        
        print("Stage 2: Skipping text filtering (no captions in SA-1B)...")
        
        print("Stage 3: Outdoor classification...")
        outdoor_results = self.classify_outdoor_batch(images_data, dataset_type="sa1b")
        outdoor_data = [images_data[i] for i, (is_outdoor, _, _) in enumerate(outdoor_results) if is_outdoor]
        print(f"Found {len(outdoor_data)} outdoor images")
        
        if len(outdoor_data) == 0:
            return {}, {}
        
        print("Stage 4: Weather classification...")
        weather_results = self.classify_weather_batch(outdoor_data, dataset_type="sa1b")
        
        print("Stage 5: Organization and balancing...")
        organized_dataset = self.organize_by_weather(
            outdoor_data, weather_results, target_per_category, dataset_type="sa1b"
        )
        
        print("Stage 6: Export...")
        final_stats = self.validate_and_export(organized_dataset, output_dir, dataset_type="sa1b")
        
        return organized_dataset, final_stats

    def run_combined_pipeline(self, laion_params=None, sa1b_params=None, 
                             combined_output_dir="combined_filtered"):
        """Run both pipelines and combine results"""
        
        print("=== RUNNING COMBINED PIPELINE ===")
        
        all_results = {}
        combined_stats = {}
        
        # Run LAION pipeline if parameters provided
        if laion_params:
            print("\n--- Processing LAION Dataset ---")
            laion_dataset, laion_stats = self.run_laion_pipeline(**laion_params)
            all_results['laion'] = laion_dataset
            combined_stats['laion'] = laion_stats
        
        # Run SA-1B pipeline if parameters provided
        if sa1b_params:
            print("\n--- Processing SA-1B Dataset ---")
            sa1b_dataset, sa1b_stats = self.run_sa1b_pipeline(**sa1b_params)
            all_results['sa1b'] = sa1b_dataset
            combined_stats['sa1b'] = sa1b_stats
        
        # Combine and save overall statistics
        Path(combined_output_dir).mkdir(exist_ok=True)
        with open(Path(combined_output_dir) / "combined_stats.json", 'w') as f:
            json.dump(combined_stats, f, indent=2)
        
        # Print summary
        print("\n=== COMBINED PIPELINE SUMMARY ===")
        for dataset_name, stats in combined_stats.items():
            total_images = sum(stats.values()) if stats else 0
            print(f"{dataset_name.upper()} Dataset: {total_images} total images")
            for category, count in stats.items():
                print(f"  {category}: {count} images")
        
        return all_results, combined_stats

# Usage Examples
if __name__ == "__main__":
    pipeline = UnifiedOutdoorWeatherPipeline()
    
    # Option 1: Run only LAION
    # laion_dataset, laion_stats = pipeline.run_laion_pipeline(
    #     dataset_name="laion400m",
    #     batch_size=1000,
    #     target_per_category=50,
    #     output_dir="laion_outdoor_weather"
    # )
    
    # Option 2: Run only SA-1B
    sa1b_dataset, sa1b_stats = pipeline.run_sa1b_pipeline(
        data_folder="/media/franck/SA-1B",
        max_images=1000,
        target_per_category=50,
        output_dir="sa1b_outdoor_weather"
    )
    
    # Option 3: Run both pipelines combined
    # combined_results, combined_stats = pipeline.run_combined_pipeline(
    #     laion_params={
    #         "dataset_name": "laion400m",
    #         "batch_size": 500,
    #         "target_per_category": 25,
    #         "output_dir": "combined_laion"
    #     },
    #     sa1b_params={
    #         "data_folder": "/path/to/sa1b/images",
    #         "max_images": 500,
    #         "target_per_category": 25,
    #         "output_dir": "combined_sa1b"
    #     },
    #     combined_output_dir="combined_outdoor_weather"
    # )
    
    print("Combined pipeline completed successfully!")
