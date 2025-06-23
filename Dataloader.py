import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import gc

class MemoryEfficientDataset(Dataset):
    def __init__(self, file_path='/home/zalert_rig305/Desktop/EE/Programs/Movies_and_TV.json', 
                 chunk_size=1000, preload_chunks=2):
        """
        Memory-efficient dataset that loads data in chunks
        
        Args:
            file_path: Path to JSON file
            chunk_size: Number of records to load per chunk
            preload_chunks: Number of chunks to keep in memory
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.preload_chunks = preload_chunks
        
        # Get total number of samples without loading all data
        self.num_samples = self._count_samples()
        
        # Cache for loaded chunks
        self.chunk_cache = {}
        self.cache_order = []
        
    def _count_samples(self):
        """Count total samples without loading entire file"""
        count = 0
        with open(self.file_path, 'r') as f:
            # If it's a JSON Lines file (one JSON object per line)
            try:
                for line in f:
                    if line.strip():
                        count += 1
                return count
            except:
                # If it's a single JSON array, load just to count
                f.seek(0)
                data = json.load(f)
                return len(data)
    
    def _load_chunk(self, chunk_idx):
        """Load a specific chunk of data"""
        if chunk_idx in self.chunk_cache:
            return self.chunk_cache[chunk_idx]
        
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.num_samples)
        
        # Load chunk from file
        with open(self.file_path, 'r') as f:
            try:
                # Try JSON Lines format first
                lines = []
                for i, line in enumerate(f):
                    if start_idx <= i < end_idx:
                        lines.append(json.loads(line.strip()))
                    elif i >= end_idx:
                        break
                chunk_data = pd.DataFrame(lines)
            except:
                # Fall back to single JSON array
                f.seek(0)
                full_data = json.load(f)
                chunk_data = pd.DataFrame(full_data[start_idx:end_idx])
        
        # Manage cache size
        if len(self.chunk_cache) >= self.preload_chunks:
            # Remove oldest chunk
            oldest_chunk = self.cache_order.pop(0)
            del self.chunk_cache[oldest_chunk]
            gc.collect()  # Force garbage collection
        
        # Add to cache
        self.chunk_cache[chunk_idx] = chunk_data
        self.cache_order.append(chunk_idx)
        
        return chunk_data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size
        
        chunk_data = self._load_chunk(chunk_idx)
        
        if local_idx >= len(chunk_data):
            raise IndexError(f"Index {idx} out of range")
        
        sample = chunk_data.iloc[local_idx]
        
        # Handle different data types more carefully
        try:
            # Convert to tensor, handling mixed data types
            if sample.dtype == 'object':
                # For mixed types, convert to strings first, then encode
                sample_values = sample.astype(str).values
                # Simple string encoding (you might want to use proper tokenization)
                encoded = [hash(str(val)) % 10000 for val in sample_values]
                return torch.tensor(encoded, dtype=torch.float32)
            else:
                return torch.tensor(sample.values, dtype=torch.float32)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return a zero tensor as fallback
            return torch.zeros(len(sample), dtype=torch.float32)

# Memory-efficient dataset creation
def create_efficient_dataloader(file_path, batch_size=8, chunk_size=500, 
                               preload_chunks=2, num_workers=0):
    """
    Create a memory-efficient dataloader
    
    Args:
        file_path: Path to data file
        batch_size: Smaller batch size for weak hardware
        chunk_size: Size of data chunks to load
        preload_chunks: Number of chunks to keep in memory
        num_workers: Set to 0 for weak hardware to avoid multiprocessing overhead
    """
    dataset = MemoryEfficientDataset(
        file_path=file_path,
        chunk_size=chunk_size,
        preload_chunks=preload_chunks
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,  # Disable shuffle to maintain chunk locality
        num_workers=num_workers,
        pin_memory=False,  # Disable for weak hardware
        drop_last=False
    )
    
    return dataloader

# Usage example
if __name__ == "__main__":
    # Create memory-efficient dataloader
    file_path = '/home/zalert_rig305/Desktop/EE/Programs/Movies_and_TV.json'
    
    # Adjust these parameters based on your hardware limitations
    dataloader = create_efficient_dataloader(
        file_path=file_path,
        batch_size=4,        # Very small batch size
        chunk_size=200,      # Small chunks
        preload_chunks=1,    # Keep only 1 chunk in memory
        num_workers=0        # No multiprocessing
    )
    
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Process data in small batches
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: Shape {batch.shape}")
        
        # Process your batch here
        
        
        # Optional: Force garbage collection after each batch
        if i % 10 == 0:
            gc.collect()
        
        # For demonstration, only process first few batches
        if i >= 2:
            print("Stopping early for demonstration...")
            break
