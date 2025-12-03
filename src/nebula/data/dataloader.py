"""FineWeb-Edu streaming dataloader."""

from typing import Iterator, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

from .tokenizer import DiffusionTokenizer


class FineWebEduDataset(IterableDataset):
    """Streaming dataset for FineWeb-Edu.

    Streams data from HuggingFace, tokenizes, and chunks into fixed-length sequences.
    """

    def __init__(
        self,
        tokenizer: DiffusionTokenizer,
        max_seq_len: int = 128,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        dataset_config: str = "sample-10BT",
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split

    def __iter__(self) -> Iterator[dict]:
        """Iterate over tokenized and chunked sequences."""
        # Load dataset in streaming mode
        dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )

        # Buffer for accumulating tokens
        token_buffer = []

        for example in dataset:
            # Tokenize the text
            text = example.get("text", "")
            if not text:
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(tokens)

            # Yield chunks of max_seq_len
            while len(token_buffer) >= self.max_seq_len:
                chunk = token_buffer[: self.max_seq_len]
                token_buffer = token_buffer[self.max_seq_len :]

                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                }


def create_dataloader(
    tokenizer: DiffusionTokenizer,
    batch_size: int,
    max_seq_len: int = 128,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
    num_workers: int = 0,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    """Create a streaming DataLoader for FineWeb-Edu.

    Args:
        tokenizer: The tokenizer to use
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration/subset
        num_workers: Number of data loading workers
        prefetch_factor: Number of batches to prefetch per worker

    Returns:
        DataLoader yielding batches of tokenized sequences
    """
    dataset = FineWebEduDataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
    )

    # Note: For streaming datasets, num_workers > 0 can cause issues
    # Each worker would iterate from the beginning
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)
