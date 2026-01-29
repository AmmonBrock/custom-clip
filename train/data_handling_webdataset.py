import torch
from torch.utils.data import DataLoader
import webdataset as wds
from PIL import Image
import torchvision.transforms as T
import glob

def create_webdataset_dataloader(
    shard_pattern="webdataset_shards/shard-{000000..000099}.tar",
    batch_size=32,
    shuffle=True,
    num_workers=4
):
    """
    Create a DataLoader from WebDataset shards.
    
    Args:
        shard_pattern: Glob pattern for shard files. Examples:
            - "webdataset_shards/shard-{000000..000099}.tar"  # Specific range (brace expansion)
            - "webdataset_shards/shard-*.tar"  # All shards (glob pattern)
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
    
    Returns:
        DataLoader iterator
    """
    
    # Expand glob patterns to actual file list
    if '*' in shard_pattern:
        shard_urls = sorted(glob.glob(shard_pattern))
        if not shard_urls:
            raise FileNotFoundError(f"No shards found matching pattern: {shard_pattern}")
    else:
        # Use the pattern directly (for brace expansion like {000000..000099})
        shard_urls = shard_pattern
    
    def warn_and_continue(exn):
        """Log errors but continue processing"""
        if isinstance(exn, (wds.autodecode.DecodingError, OSError)):
            print(f"Warning: Skipping corrupted sample - {exn}")
            return True  
        return False  
    
    # Define the same transforms as before
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
    ])
    
    # Create the WebDataset pipeline
    dataset = (
        wds.WebDataset(shard_urls, shardshuffle=shuffle) 
        .shuffle(1000 if shuffle else 0)  # Shuffle buffer size
        .decode("pil", handler=warn_and_continue)  
        .to_tuple("jpg", "txt")  
        .map_tuple(transform, lambda x: x)  
        .map(lambda x: {"image": x[0], "caption": x[1]})  # Convert to dict format
        .batched(batch_size)  # Batch the data
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=None,  # Batching is handled by WebDataset
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader