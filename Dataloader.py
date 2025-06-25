import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import gzip
import gc
import os
from typing import Union, Optional

class MemoryEfficientDataset(Dataset):
    def __init__(self, file_path: str = '/home/zalert_rig305/Desktop/EE/Programs/Movies_and_TV.json.gz', 
                 chunk_size: int = 1000, preload_chunks: int = 2):
        """
        Memory-efficient dataset that loads data in chunks with proper file format handling
        
        Args:
            file_path: Path to JSON or JSON.gz file
            chunk_size: Number of records to load per chunk
            preload_chunks: Number of chunks to keep in memory
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.preload_chunks = preload_chunks
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type
        self.is_gzipped = file_path.endswith('.gz')
        self.is_jsonl = self._is_json_lines_format()
        
        # Get total number of samples without loading all data
        self.num_samples = self._count_samples()
        
        if self.num_samples == 0:
            raise ValueError(f"No valid samples found in {file_path}")
        
        # Cache for loaded chunks
        self.chunk_cache = {}
        self.cache_order = []
        
        print(f"Initialized dataset: {self.num_samples} samples, "
              f"{'gzipped' if self.is_gzipped else 'plain'} "
              f"{'JSON Lines' if self.is_jsonl else 'JSON array'}")
    
    def _get_file_opener(self):
        """Get appropriate file opener based on file type"""
        if self.is_gzipped:
            return lambda path, mode='rt': gzip.open(path, mode, encoding='utf-8')
        else:
            return lambda path, mode='r': open(path, mode, encoding='utf-8')
    
    def _is_json_lines_format(self) -> bool:
        """Detect if file is in JSON Lines format"""
        opener = self._get_file_opener()
        try:
            with opener(self.file_path) as f:
                first_line = f.readline().strip()
                if not first_line:
                    return False
                
                # Try to parse as single JSON object
                json.loads(first_line)
                
                # Check if second line exists and is also valid JSON
                second_line = f.readline().strip()
                if second_line:
                    try:
                        json.loads(second_line)
                        return True  # Multiple JSON objects = JSON Lines
                    except json.JSONDecodeError:
                        return False
                return False  # Only one line = single JSON object
        except (json.JSONDecodeError, Exception):
            return False
    
    def _count_samples(self) -> int:
        """Count total samples without loading entire file"""
        opener = self._get_file_opener()
        
        try:
            with opener(self.file_path) as f:
                if self.is_jsonl:
                    # Count lines for JSON Lines format
                    count = 0
                    for line in f:
                        if line.strip():
                            count += 1
                    return count
                else:
                    # Load and count array elements for single JSON
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            return len(data)
                        elif isinstance(data, dict):
                            return 1
                        else:
                            return 0
                    except json.JSONDecodeError:
                        return 0
        except Exception as e:
            print(f"Error counting samples: {e}")
            return 0
    
    def _load_chunk(self, chunk_idx: int) -> pd.DataFrame:
        """Load a specific chunk of data with improved error handling"""
        if chunk_idx in self.chunk_cache:
            return self.chunk_cache[chunk_idx]
        
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.num_samples)
        
        if start_idx >= self.num_samples:
            return pd.DataFrame()
        
        opener = self._get_file_opener()
        chunk_data = None
        
        try:
            with opener(self.file_path) as f:
                if self.is_jsonl:
                    # Handle JSON Lines format
                    lines = []
                    for i, line in enumerate(f):
                        if start_idx <= i < end_idx:
                            try:
                                lines.append(json.loads(line.strip()))
                            except json.JSONDecodeError as e:
                                print(f"Warning: Skipping malformed JSON at line {i}: {e}")
                                continue
                        elif i >= end_idx:
                            break
                    chunk_data = pd.DataFrame(lines)
                else:
                    # Handle single JSON array
                    full_data = json.load(f)
                    if isinstance(full_data, list):
                        chunk_data = pd.DataFrame(full_data[start_idx:end_idx])
                    else:
                        chunk_data = pd.DataFrame([full_data] if start_idx == 0 else [])
        
        except Exception as e:
            print(f"Error loading chunk {chunk_idx}: {e}")
            chunk_data = pd.DataFrame()
        
        # Manage cache size
        if len(self.chunk_cache) >= self.preload_chunks:
            # Remove oldest chunk
            if self.cache_order:
                oldest_chunk = self.cache_order.pop(0)
                if oldest_chunk in self.chunk_cache:
                    del self.chunk_cache[oldest_chunk]
                gc.collect()  # Force garbage collection
        
        # Add to cache
        self.chunk_cache[chunk_idx] = chunk_data
        self.cache_order.append(chunk_idx)
        
        return chunk_data
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx >= self.num_samples or idx < 0:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")

        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size

        chunk_data = self._load_chunk(chunk_idx)

        if chunk_data.empty or local_idx >= len(chunk_data):
            print(f"Warning: Empty or invalid chunk at index {idx}")
            return torch.zeros(self.default_tensor_length, dtype=torch.float32)

        try:
            sample = chunk_data.iloc[local_idx]
            return self._convert_sample_to_tensor(sample)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return torch.zeros(self.default_tensor_length, dtype=torch.float32)

    def _convert_sample_to_tensor(self, sample: pd.Series) -> torch.Tensor:
        try:
            encoded_values = []
            for val in sample.values:
                if isinstance(val, (int, float)):
                    encoded_values.append(float(val))
                elif isinstance(val, str):
                    encoded_values.append(float(hash(val) % 10000) / 10000.0)
                elif val is None or pd.isna(val):
                    encoded_values.append(0.0)
                else:
                    encoded_values.append(float(hash(str(val)) % 10000) / 10000.0)

            return torch.tensor(encoded_values, dtype=torch.float32)
        except Exception as e:
            print(f"Tensor conversion error: {e}")
            return torch.zeros(self.default_tensor_length, dtype=torch.float32)



def create_efficient_dataloader(file_path: str, batch_size: int = 8, 
                               chunk_size: int = 500, preload_chunks: int = 2, 
                               num_workers: int = 0, shuffle: bool = False) -> DataLoader:
    """
    Create a memory-efficient dataloader with improved configuration
    
    Args:
        file_path: Path to data file (supports .json and .json.gz)
        batch_size: Batch size for training
        chunk_size: Size of data chunks to load
        preload_chunks: Number of chunks to keep in memory
        num_workers: Number of worker processes (0 for single-threaded)
        shuffle: Whether to shuffle data (disabled by default for chunk locality)
    
    Returns:
        DataLoader: Configured DataLoader instance
    """
    try:
        dataset = MemoryEfficientDataset(
            file_path=file_path,
            chunk_size=chunk_size,
            preload_chunks=preload_chunks
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,  # Disable for memory efficiency
            drop_last=False,
            persistent_workers=False
        )
        
        return dataloader
        
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        raise


def validate_dataloader(dataloader: DataLoader, max_batches: int = 5) -> bool:
    """
    Validate that the dataloader works correctly
    
    Args:
        dataloader: DataLoader to validate
        max_batches: Maximum number of batches to test
    
    Returns:
        bool: True if validation passes
    """
    try:
        print(f"Validating dataloader...")
        print(f"Dataset size: {len(dataloader.dataset)}")
        print(f"Number of batches: {len(dataloader)}")
        
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}: Shape {batch.shape}, dtype {batch.dtype}")
            
            # Basic validation checks
            if torch.isnan(batch).any():
                print(f"Warning: NaN values found in batch {i}")
            
            if torch.isinf(batch).any():
                print(f"Warning: Infinite values found in batch {i}")
            
            # Stop after max_batches
            if i >= max_batches - 1:
                break
        
        print("Dataloader validation completed successfully")
        return True
        
    except Exception as e:
        print(f"Dataloader validation failed: {e}")
        return False


# Usage example
if __name__ == "__main__":
    # Configuration for different scenarios
    configs = {
        'weak_hardware': {
            'batch_size': 2,
            'chunk_size': 200,
            'preload_chunks': 1,
            'num_workers': 0
        },
        'medium_hardware': {
            'batch_size': 8,
            'chunk_size': 500,
            'preload_chunks': 2,
            'num_workers': 0
        },
        'strong_hardware': {
            'batch_size': 32,
            'chunk_size': 1000,
            'preload_chunks': 4,
            'num_workers': 2
        }
    }
    
    # Default file path (update as needed)
    file_path = '/home/zalert_rig305/Desktop/EE/Programs/Movies_and_TV.json.gz'
    
    # Choose configuration based on your hardware
    config = configs['medium_hardware']  # Change as needed
    
    try:
        # Create dataloader
        dataloader = create_efficient_dataloader(
            file_path=file_path,
            **config
        )
        
        # Validate dataloader
        if validate_dataloader(dataloader, max_batches=3):
            print("\nDataloader is ready for use!")
        else:
            print("\nDataloader validation failed!")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your file path and ensure the data file exists.")