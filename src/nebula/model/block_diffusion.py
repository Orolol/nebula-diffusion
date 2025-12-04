"""Block Diffusion for efficient masked diffusion with KV caching.

Implements the BD3-LMs approach:
- Autoregressive across blocks (enables exact KV caching for completed blocks)
- Diffusion within blocks (enables parallel token generation)

Block size of 4-8 tokens balances caching efficiency with parallel generation.

Reference: Block Diffusion (arXiv:2503.09573)
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockDiffusion:
    """Block-level masked diffusion process.

    Key concepts:
    - Sequence is divided into blocks of fixed size
    - Blocks are processed autoregressively (each block conditions on previous blocks)
    - Within each block, tokens are denoised via diffusion
    - This enables KV caching for completed blocks while retaining diffusion benefits

    Training:
        log p(x) = Σ log p(x^b | x^{<b})

    Each term log p(x^b | x^{<b}) is trained with masked diffusion:
        1. Sample masking ratio t ~ Uniform[0, 1]
        2. Mask tokens within the block
        3. Predict masked tokens given context + masked block
        4. Compute cross-entropy loss on masked positions
    """

    def __init__(
        self,
        mask_token_id: int,
        vocab_size: int,
        block_size: int = 8,
    ):
        """
        Args:
            mask_token_id: ID of the [MASK] token
            vocab_size: Size of vocabulary
            block_size: Number of tokens per block
        """
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.block_size = block_size

    def split_into_blocks(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Split sequence into blocks.

        Args:
            input_ids: [batch, seq_len]

        Returns:
            List of [batch, block_size] tensors
        """
        batch_size, seq_len = input_ids.shape

        # Pad to multiple of block_size if needed
        if seq_len % self.block_size != 0:
            pad_len = self.block_size - (seq_len % self.block_size)
            input_ids = F.pad(input_ids, (0, pad_len), value=self.mask_token_id)

        num_blocks = input_ids.shape[1] // self.block_size
        blocks = input_ids.chunk(num_blocks, dim=1)

        return list(blocks)

    def merge_blocks(self, blocks: List[torch.Tensor], original_len: int) -> torch.Tensor:
        """Merge blocks back into sequence.

        Args:
            blocks: List of [batch, block_size] tensors
            original_len: Original sequence length (to remove padding)

        Returns:
            [batch, original_len] tensor
        """
        merged = torch.cat(blocks, dim=1)
        return merged[:, :original_len]

    def sample_masking_ratio(
        self,
        batch_size: int,
        device: torch.device,
        min_ratio: float = 0.0,
        max_ratio: float = 1.0,
    ) -> torch.Tensor:
        """Sample masking ratio for each block in batch."""
        return torch.rand(batch_size, device=device) * (max_ratio - min_ratio) + min_ratio

    def mask_block(
        self,
        block: torch.Tensor,
        masking_ratio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply masking to a single block.

        Args:
            block: [batch, block_size]
            masking_ratio: [batch] or scalar

        Returns:
            masked_block: [batch, block_size]
            mask_indicator: [batch, block_size] boolean
        """
        batch_size, block_size = block.shape
        device = block.device

        rand = torch.rand(batch_size, block_size, device=device)

        if masking_ratio.dim() == 0:
            mask_indicator = rand < masking_ratio
        else:
            mask_indicator = rand < masking_ratio.unsqueeze(1)

        masked_block = block.clone()
        masked_block[mask_indicator] = self.mask_token_id

        return masked_block, mask_indicator

    def prepare_training_batch(
        self,
        input_ids: torch.Tensor,
        min_ratio: float = 0.0,
        max_ratio: float = 1.0,
    ) -> dict:
        """Prepare a training batch with random masking - fully vectorized.

        For efficient training, we mask a random fraction of ALL tokens.
        The masking ratio determines what fraction of the ENTIRE sequence is masked,
        providing substantial training signal per batch.

        Note: This is standard masked LM training, not strict block diffusion.
        Block structure is preserved for generation but not enforced during training.

        Args:
            input_ids: [batch, seq_len]
            min_ratio: Minimum masking ratio (fraction of sequence to mask)
            max_ratio: Maximum masking ratio (fraction of sequence to mask)

        Returns:
            Dictionary with:
                - input_ids: Sequence with masked tokens
                - target_ids: Original tokens
                - mask_indicator: Which positions are masked
                - block_indices: Dummy block indices (for compatibility)
                - masking_ratio: Actual masking ratio used
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Sample masking ratios per sample (fraction of entire sequence)
        masking_ratio = self.sample_masking_ratio(batch_size, device, min_ratio, max_ratio)

        # Debug: Check masking ratio bounds
        if (masking_ratio < 0.0).any() or (masking_ratio > 1.0).any():
            print(f"WARNING: Invalid masking ratio detected! Min: {masking_ratio.min()}, Max: {masking_ratio.max()}")
            masking_ratio = torch.clamp(masking_ratio, 0.0, 1.0)

        # Generate random values for each position
        rand = torch.rand(batch_size, seq_len, device=device)

        # Mask positions where random < masking_ratio
        mask_indicator = rand < masking_ratio.unsqueeze(1)

        # Create masked input
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask_indicator] = self.mask_token_id

        # Dummy block indices for compatibility
        block_indices = torch.zeros(batch_size, dtype=torch.long, device=device)

        return {
            "input_ids": masked_input_ids,
            "target_ids": input_ids,
            "mask_indicator": mask_indicator,
            "block_indices": block_indices,
            "masking_ratio": masking_ratio,
        }

    def compute_loss(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        mask_indicator: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss on masked positions.

        Args:
            logits: [batch, seq_len, vocab_size]
            target_ids: [batch, seq_len]
            mask_indicator: [batch, seq_len] boolean

        Returns:
            Scalar loss
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Flatten
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = target_ids.view(-1)
        mask_flat = mask_indicator.view(-1)

        # CE loss on all positions
        loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")

        # Average only over masked positions
        masked_loss = loss_per_token[mask_flat]

        if masked_loss.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return masked_loss.mean()


class BlockCausalMask:
    """Utility for creating block-causal attention masks.

    In block diffusion:
    - Tokens within a block can attend to each other (bidirectional)
    - Tokens can only attend to previous blocks (causal across blocks)
    - Current block being generated can attend to all context blocks
    """

    def __init__(self, block_size: int):
        self.block_size = block_size

    def create_mask(
        self,
        seq_len: int,
        current_block_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create attention mask for block-level generation - vectorized.

        Args:
            seq_len: Total sequence length (context + current block)
            current_block_idx: Index of the block being generated
            device: Device to create mask on

        Returns:
            Attention mask [seq_len, seq_len]
            True means "can attend", False means "cannot attend"
        """
        # Create position indices
        positions = torch.arange(seq_len, device=device)
        block_ids = positions // self.block_size  # Which block each position belongs to

        # Row and column block indices
        row_blocks = block_ids.unsqueeze(1)  # [seq_len, 1]
        col_blocks = block_ids.unsqueeze(0)  # [1, seq_len]

        num_context_blocks = current_block_idx
        context_len = num_context_blocks * self.block_size

        # Context positions (completed blocks)
        is_context_row = positions.unsqueeze(1) < context_len
        is_context_col = positions.unsqueeze(0) < context_len

        # Within same block (bidirectional)
        same_block = row_blocks == col_blocks

        # Context can attend to same or earlier blocks (causal at block level)
        context_causal = (row_blocks >= col_blocks) & is_context_row & is_context_col

        # Current block can attend to all context and itself
        is_current_row = ~is_context_row.squeeze(1)
        current_to_context = is_current_row.unsqueeze(1) & is_context_col
        current_bidirectional = is_current_row.unsqueeze(1) & (~is_context_col)

        # Combine masks
        mask = same_block | context_causal | current_to_context | current_bidirectional

        return mask

    def create_training_mask(
        self,
        seq_len: int,
        block_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Create per-sample training masks - vectorized.

        During training, each sample may have a different block being trained.

        Args:
            seq_len: Sequence length
            block_indices: [batch] which block is being trained per sample
            device: Device

        Returns:
            [batch, seq_len, seq_len] attention masks
        """
        batch_size = block_indices.shape[0]

        # Create position indices
        positions = torch.arange(seq_len, device=device)
        block_ids = positions // self.block_size  # [seq_len]

        # Row and column block indices: [1, seq_len, 1] and [1, 1, seq_len]
        row_blocks = block_ids.view(1, seq_len, 1)
        col_blocks = block_ids.view(1, 1, seq_len)

        # Context lengths per sample: [batch, 1, 1]
        context_lens = (block_indices * self.block_size).view(batch_size, 1, 1)

        # Position grids
        row_positions = positions.view(1, seq_len, 1)  # [1, seq_len, 1]
        col_positions = positions.view(1, 1, seq_len)  # [1, 1, seq_len]

        # Context masks per sample
        is_context_row = row_positions < context_lens  # [batch, seq_len, 1]
        is_context_col = col_positions < context_lens  # [batch, 1, seq_len]

        # Same block (bidirectional within block)
        same_block = row_blocks == col_blocks  # [1, seq_len, seq_len]

        # Context causal at block level
        context_causal = (row_blocks >= col_blocks) & is_context_row & is_context_col

        # Current block to context
        is_current_row = ~is_context_row  # [batch, seq_len, 1]
        current_to_context = is_current_row & is_context_col

        # Current block bidirectional
        current_bidirectional = is_current_row & (~is_context_col)

        # Combine
        mask = same_block | context_causal | current_to_context | current_bidirectional

        return mask
