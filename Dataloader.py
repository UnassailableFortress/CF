import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import numpy as np
import json
import gzip
import gc
import os
from typing import Union, Optional, List, Dict
from collections import defaultdict
import random

class CFDataset(Dataset):
    """Specialized dataset for Collaborative Filtering with user-item interactions"""
    
    def __init__(self, df: pd.DataFrame, negative_sampling_ratio: int = 4):
        """
        Initialize CF dataset with positive and negative sampling
        
        Args:
            df: DataFrame with columns ['user_id', 'item_id', 'rating', 'timestamp']
            negative_sampling_ratio: Number of negative samples per positive sample
        """
        self.df = df
        self.negative_sampling_ratio = negative_sampling_ratio
        
        # Create mappings
        self.user_to_idx = {user: idx for idx, user in enumerate(df['user_id'].unique())}
        self.item_to_idx = {item: idx for idx, item in enumerate(df['item_id'].unique())}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # Create user-item interaction sets for negative sampling
        self.user_items = defaultdict(set)
        for _, row in df.iterrows():
            user_idx = self.user_to_idx[row['user_id']]
            item_idx = self.item_to_idx[row['item_id']]
            self.user_items[user_idx].add(item_idx)
        
        self.n_users = len(self.user_to_idx)
        self.n_items = len(self.item_to_idx)
        
        # Prepare positive samples
        self.positive_samples = []
        for _, row in df.iterrows():
            user_idx = self.user_to_idx[row['user_id']]
            item_idx = self.item_to_idx[row['item_id']]
            rating = row.get('rating', 1.0)
            timestamp = row.get('timestamp', 0)
            self.positive_samples.append((user_idx, item_idx, rating, timestamp))
        
        print(f"Initialized CF Dataset: {self.n_users} users, {self.n_items} items, "
              f"{len(self.positive_samples)} interactions")
    
    def __len__(self):
        return len(self.positive_samples) * (1 + self.negative_sampling_ratio)
    
    def __getitem__(self, idx):
        # Determine if this is a positive or negative sample
        n_positive = len(self.positive_samples)
        
        if idx < n_positive:
            # Positive sample
            user_idx, item_idx, rating, timestamp = self.positive_samples[idx]
            label = 1.0
        else:
            # Negative sample
            pos_idx = (idx - n_positive) % n_positive
            user_idx, _, _, timestamp = self.positive_samples[pos_idx]
            
            # Sample a negative item (not in user's history)
            user_items = self.user_items[user_idx]
            negative_items = list(set(range(self.n_items)) - user_items)
            
            if negative_items:
                item_idx = random.choice(negative_items)
            else:
                item_idx = random.randint(0, self.n_items - 1)
            
            rating = 0.0
            label = 0.0
        
        # Return as tensors
        return {
            'user': torch.tensor(user_idx, dtype=torch.long),
            'item': torch.tensor(item_idx, dtype=torch.long),
            'rating': torch.tensor(rating, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32),
            'timestamp': torch.tensor(timestamp, dtype=torch.long)
        }
    
    def get_user_items(self, user_id):
        """Get all items interacted by a user"""
        if user_id in self.user_to_idx:
            user_idx = self.user_to_idx[user_id]
            return [self.idx_to_item[item_idx] for item_idx in self.user_items[user_idx]]
        return []
    
    def get_item_users(self, item_id):
        """Get all users who interacted with an item"""
        if item_id not in self.item_to_idx:
            return []
        
        item_idx = self.item_to_idx[item_id]
        users = []
        for user_idx, items in self.user_items.items():
            if item_idx in items:
                users.append(self.idx_to_user[user_idx])
        return users


class MemoryEfficientDataset(Dataset):
    """Enhanced memory-efficient dataset with CF-specific features"""
    
    def __init__(self, file_path: str, chunk_size: int = 1000, 
                 preload_chunks: int = 2, transform=None):
        """
        Memory-efficient dataset that loads data in chunks
        
        Args:
            file_path: Path to JSON or JSON.gz file
            chunk_size: Number of records to load per chunk
            preload_chunks: Number of chunks to keep in memory
            transform: Optional transform to apply to samples
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.preload_chunks = preload_chunks
        self.transform = transform
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type
        self.is_gzipped = file_path.endswith('.gz')
        self.is_jsonl = self._is_json_lines_format()
        
        # Get total number of samples
        self.num_samples = self._count_samples()
        
        if self.num_samples == 0:
            raise ValueError(f"No valid samples found in {file_path}")
        
        # Cache for loaded chunks
        self.chunk_cache = {}
        self.cache_order = []
        
        # Extract schema from first sample
        self._extract_schema()
        
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
                
                json.loads(first_line)
                second_line = f.readline().strip()
                if second_line:
                    try:
                        json.loads(second_line)
                        return True
                    except json.JSONDecodeError:
                        return False
                return False
        except:
            return False
    
    def _count_samples(self) -> int:
        """Count total samples without loading entire file"""
        opener = self._get_file_opener()
        
        try:
            with opener(self.file_path) as f:
                if self.is_jsonl:
                    count = sum(1 for line in f if line.strip())
                    return count
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        return len(data)
                    elif isinstance(data, dict):
                        return 1
                    else:
                        return 0
        except Exception as e:
            print(f"Error counting samples: {e}")
            return 0
    
    def _extract_schema(self):
        """Extract schema from first sample"""
        opener = self._get_file_opener()
        
        try:
            with opener(self.file_path) as f:
                if self.is_jsonl:
                    first_line = f.readline().strip()
                    if first_line:
                        sample = json.loads(first_line)
                        self.schema = list(sample.keys())
                        self.default_tensor_length = len(self.schema)
                else:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        self.schema = list(data[0].keys())
                        self.default_tensor_length = len(self.schema)
                    else:
                        self.schema = []
                        self.default_tensor_length = 1
        except:
            self.schema = []
            self.default_tensor_length = 1
    
    def _load_chunk(self, chunk_idx: int) -> pd.DataFrame:
        """Load a specific chunk of data"""
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
                    lines = []
                    for i, line in enumerate(f):
                        if start_idx <= i < end_idx:
                            try:
                                lines.append(json.loads(line.strip()))
                            except json.JSONDecodeError:
                                continue
                        elif i >= end_idx:
                            break
                    chunk_data = pd.DataFrame(lines)
                else:
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
            if self.cache_order:
                oldest_chunk = self.cache_order.pop(0)
                if oldest_chunk in self.chunk_cache:
                    del self.chunk_cache[oldest_chunk]
                gc.collect()
        
        # Add to cache
        self.chunk_cache[chunk_idx] = chunk_data
        self.cache_order.append(chunk_idx)
        
        return chunk_data
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int):
        if idx >= self.num_samples or idx < 0:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")
        
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size
        
        chunk_data = self._load_chunk(chunk_idx)
        
        if chunk_data.empty or local_idx >= len(chunk_data):
            return self._get_default_sample()
        
        try:
            sample = chunk_data.iloc[local_idx]
            
            # Convert to CF-friendly format
            if self.transform:
                return self.transform(sample)
            else:
                return self._convert_sample_to_cf_format(sample)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return self._get_default_sample()
    
    def _convert_sample_to_cf_format(self, sample: pd.Series) -> Dict[str, torch.Tensor]:
        """Convert sample to CF format"""
        try:
            # Map common fields
            user_id = sample.get('reviewerID', sample.get('user_id', 'unknown'))
            item_id = sample.get('asin', sample.get('item_id', 'unknown'))
            rating = float(sample.get('overall', sample.get('rating', 1.0)))
            timestamp = int(sample.get('unixReviewTime', sample.get('timestamp', 0)))
            
            return {
                'user_id': user_id,
                'item_id': item_id,
                'rating': torch.tensor(rating, dtype=torch.float32),
                'timestamp': torch.tensor(timestamp, dtype=torch.long),
                'raw_data': sample.to_dict()
            }
        except Exception as e:
            print(f"Conversion error: {e}")
            return self._get_default_sample()
    
    def _get_default_sample(self):
        """Get default sample for error cases"""
        return {
            'user_id': 'unknown',
            'item_id': 'unknown',
            'rating': torch.tensor(0.0, dtype=torch.float32),
            'timestamp': torch.tensor(0, dtype=torch.long),
            'raw_data': {}
        }


class UserBatchSampler(Sampler):
    """Custom sampler that ensures each batch contains multiple items per user"""
    
    def __init__(self, dataset: CFDataset, batch_size: int, items_per_user: int = 4):
        """
        Initialize user-based batch sampler
        
        Args:
            dataset: CFDataset instance
            batch_size: Total batch size
            items_per_user: Number of items to sample per user
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.items_per_user = items_per_user
        self.users_per_batch = max(1, batch_size // items_per_user)
        
        # Group samples by user
        self.user_samples = defaultdict(list)
        for idx, (user_idx, _, _, _) in enumerate(dataset.positive_samples):
            self.user_samples[user_idx].append(idx)
        
        # Filter users with enough samples
        self.valid_users = [
            user for user, samples in self.user_samples.items()
            if len(samples) >= items_per_user
        ]
        
        self.n_batches = len(self.valid_users) // self.users_per_batch
    
    def __iter__(self):
        # Shuffle users
        random.shuffle(self.valid_users)
        
        for batch_idx in range(self.n_batches):
            batch_indices = []
            
            # Select users for this batch
            start_idx = batch_idx * self.users_per_batch
            end_idx = min(start_idx + self.users_per_batch, len(self.valid_users))
            batch_users = self.valid_users[start_idx:end_idx]
            
            # Sample items for each user
            for user in batch_users:
                user_samples = self.user_samples[user]
                sampled_indices = random.sample(
                    user_samples, 
                    min(self.items_per_user, len(user_samples))
                )
                batch_indices.extend(sampled_indices)
            
            yield batch_indices
    
    def __len__(self):
        return self.n_batches


def create_cf_dataloader(df: pd.DataFrame, batch_size: int = 32, 
                        shuffle: bool = True, num_workers: int = 0,
                        negative_sampling_ratio: int = 4,
                        use_user_batching: bool = False) -> DataLoader:
    """
    Create a dataloader optimized for Collaborative Filtering
    
    Args:
        df: DataFrame with user-item interactions
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        negative_sampling_ratio: Ratio of negative to positive samples
        use_user_batching: Whether to use user-based batching
    
    Returns:
        DataLoader configured for CF
    """
    try:
        dataset = CFDataset(df, negative_sampling_ratio=negative_sampling_ratio)
        
        if use_user_batching and batch_size >= 8:
            sampler = UserBatchSampler(dataset, batch_size)
            dataloader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=num_workers,
                pin_memory=False
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=False
            )
        
        return dataloader
        
    except Exception as e:
        print(f"Error creating CF dataloader: {e}")
        raise


def create_efficient_dataloader(file_path: str, batch_size: int = 8, 
                               chunk_size: int = 500, preload_chunks: int = 2, 
                               num_workers: int = 0, shuffle: bool = False) -> DataLoader:
    """
    Create a memory-efficient dataloader
    
    Args:
        file_path: Path to data file (supports .json and .json.gz)
        batch_size: Batch size for training
        chunk_size: Size of data chunks to load
        preload_chunks: Number of chunks to keep in memory
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader instance
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
            pin_memory=False,
            drop_last=False,
            persistent_workers=False
        )
        
        return dataloader
        
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        raise


def validate_cf_dataloader(dataloader: DataLoader, max_batches: int = 3) -> bool:
    """
    Validate that the CF dataloader works correctly
    
    Args:
        dataloader: DataLoader to validate
        max_batches: Maximum number of batches to test
    
    Returns:
        bool: True if validation passes
    """
    try:
        print(f"Validating CF dataloader...")
        print(f"Dataset size: {len(dataloader.dataset)}")
        
        for i, batch in enumerate(dataloader):
            if isinstance(batch, dict):
                print(f"\nBatch {i}:")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: Shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"Batch {i}: Shape {batch.shape}, dtype {batch.dtype}")
            
            if i >= max_batches - 1:
                break
        
        print("\nDataloader validation completed successfully")
        return True
        
    except Exception as e:
        print(f"Dataloader validation failed: {e}")
        return False


# Usage example
if __name__ == "__main__":
    import pandas as pd
    
    # Example: Create sample CF data
    sample_data = pd.DataFrame({
        'user_id': ['user1', 'user1', 'user2', 'user2', 'user3'] * 20,
        'item_id': ['item1', 'item2', 'item1', 'item3', 'item2'] * 20,
        'rating': [4.0, 5.0, 3.0, 4.0, 5.0] * 20,
        'timestamp': [1234567890 + i for i in range(100)]
    })
    
    # Create CF dataloader
    cf_dataloader = create_cf_dataloader(
        sample_data,
        batch_size=16,
        negative_sampling_ratio=4,
        use_user_batching=True
    )
    
    # Validate
    if validate_cf_dataloader(cf_dataloader, max_batches=2):
        print("\nCF Dataloader is ready for use!")
    
    # For file-based loading
    file_path = '/home/zalert_rig305/Desktop/EE/Programs/Movies_and_TV.json.gz'
    if os.path.exists(file_path):
        print(f"\nCreating dataloader from file: {file_path}")
        file_dataloader = create_efficient_dataloader(
            file_path=file_path,
            batch_size=32,
            chunk_size=1000,
            preload_chunks=3
        )
        print("File-based dataloader created successfully!")