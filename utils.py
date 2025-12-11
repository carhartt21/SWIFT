import lmdb
import pickle
from pathlib import Path
import cv2
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

def parallel_folder_scan(root_folder, max_workers=None):
    """Parallel folder scanning for large datasets"""
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    def scan_subfolder(subfolder):
        return list(Path(subfolder).glob("*.jpg"))
    
    # Get all subfolders
    subfolders = [f for f in Path(root_folder).iterdir() if f.is_dir()]
    
    # Parallel scan
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(scan_subfolder, subfolders))
    
    # Flatten results
    all_files = []
    for file_list in results:
        all_files.extend(file_list)
    
    return all_files

def parallel_lmdb_conversion(source_folder, lmdb_path, max_workers=4):
    """Convert SA-1B to LMDB using parallel processing"""
    
    # Scan files in parallel
    print("Scanning folders...")
    image_files = parallel_folder_scan(source_folder)
    print(f"Found {len(image_files)} images")
    
    # Convert in chunks
    chunk_size = len(image_files) // max_workers
    chunks = [image_files[i:i+chunk_size] for i in range(0, len(image_files), chunk_size)]
    
    def convert_chunk(chunk_data):
        chunk_files, chunk_id = chunk_data
        chunk_lmdb = f"{lmdb_path}_chunk_{chunk_id}"
        
        env = lmdb.open(chunk_lmdb, map_size=len(chunk_files) * 1024 * 1024 * 3 * 2)
        
        with env.begin(write=True) as txn:
            for i, img_path in enumerate(chunk_files):
                try:
                    img = cv2.imread(str(img_path))
                    img_encoded = cv2.imencode('.jpg', img)[1].tobytes()
                    key = f"{chunk_id}_{i:08d}".encode('ascii')
                    txn.put(key, img_encoded)
                except Exception as e:
                    print(f"Error in chunk {chunk_id}: {e}")
        
        env.close()
        return len(chunk_files)
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        chunk_data = [(chunk, i) for i, chunk in enumerate(chunks)]
        results = list(executor.map(convert_chunk, chunk_data))
    
    print(f"Converted {sum(results)} images across {len(chunks)} chunks")


def convert_sa1b_to_lmdb(source_folder, lmdb_path, max_images=None):
    """Convert SA-1B folder to LMDB for faster loading"""
    
    # Calculate database size (estimate)
    img_size_estimate = 1024 * 1024 * 3  # Assuming max 1024x1024x3
    db_size = (max_images or 1000000) * img_size_estimate * 2  # 2x buffer
    
    env = lmdb.open(str(lmdb_path), map_size=db_size)
    
    image_files = list(Path(source_folder).glob("*.jpg"))[:max_images]
    
    with env.begin(write=True) as txn:
        for i, img_path in enumerate(image_files):
            try:
                # Load and encode image
                img = cv2.imread(str(img_path))
                img_encoded = cv2.imencode('.jpg', img)[1].tobytes()
                
                # Store with key-value pair
                key = f"{i:08d}".encode('ascii')
                txn.put(key, img_encoded)
                
                if i % 10000 == 0:
                    print(f"Processed {i} images")
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    env.close()
    print(f"LMDB conversion complete: {len(image_files)} images")

def load_from_lmdb(lmdb_path, max_images):
    """Fast loading from LMDB"""
    env = lmdb.open(str(lmdb_path), readonly=True)
    images_data = []
    
    with env.begin() as txn:
        cursor = txn.cursor()
        for i, (key, value) in enumerate(cursor):
            if i >= max_images:
                break
                
            # Decode image
            img_array = np.frombuffer(value, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            
            # Resize for efficiency
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            images_data.append({
                'image': img,
                'key': key.decode('ascii'),
                'index': i
            })
    
    env.close()
    return images_data
