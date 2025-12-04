"""FineWeb-Edu streaming dataloader using fast_loader."""

from typing import Iterator, Any

from fast_loader import FastFinewebDataset
from .tokenizer import DiffusionTokenizer


def create_dataloader(
    tokenizer: DiffusionTokenizer,
    batch_size: int,
    max_seq_len: int = 128,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
    num_workers: int = 4,  # Kept for compatibility, but might be unused by fast_loader
    prefetch_factor: int = 4,  # Kept for compatibility
) -> Any:
    """Create a streaming DataLoader for FineWeb-Edu.

    Args:
        tokenizer: The tokenizer to use
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        dataset_name: HuggingFace dataset name (unused by fast_loader currently, assumes fineweb)
        dataset_config: Dataset configuration/subset (unused by fast_loader currently)
        num_workers: Number of data loading workers (unused)
        prefetch_factor: Number of batches to prefetch (unused)

    Returns:
        FastFinewebDataset iterator yielding batches
    """
    # Initialize FastFinewebDataset
    # Note: fast_loader handles batching, sharding, and prefetching internally
    dataset = FastFinewebDataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        split="train",
        batch_size=batch_size,
        max_length=max_seq_len,
        prefetch_batches=prefetch_factor * 4,
        num_workers=num_workers if num_workers > 0 else 1,
    )

    return dataset
