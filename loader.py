from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class SA1BLMDBDataset(Dataset):
    def __init__(self, lmdb_path, transform=None):
        self.env = lmdb.open(str(lmdb_path), readonly=True)
        self.transform = transform
        
        # Get dataset size
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with self.env.begin() as txn:
            key = f"{idx:08d}".encode('ascii')
            img_bytes = txn.get(key)
            
            if img_bytes is None:
                return None
            
            # Decode image
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            
            if self.transform:
                img = self.transform(img)
            
            return img, idx

# Create optimized DataLoader
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

dataset = SA1BLMDBDataset(lmdb_path="sa1b.lmdb", transform=transform)
dataloader = DataLoader(
    dataset, 
    batch_size=32,  # Process in batches
    shuffle=False,
    num_workers=8,  # Parallel loading
    pin_memory=True,  # GPU optimization
    prefetch_factor=2
)
