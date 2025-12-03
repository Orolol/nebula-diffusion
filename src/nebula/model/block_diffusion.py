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
        """Prepare a training batch with block-level masking.

        For efficient training, we randomly select which block to train on
        for each sequence, then mask that block while keeping context clean.

        Args:
            input_ids: [batch, seq_len]
            min_ratio: Minimum masking ratio
            max_ratio: Maximum masking ratio

        Returns:
            Dictionary with:
                - input_ids: Full sequence with one masked block per sample
                - target_ids: Original tokens
                - mask_indicator: Which positions are masked
                - block_indices: Which block was masked for each sample
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Split into blocks
        blocks = self.split_into_blocks(input_ids)
        num_blocks = len(blocks)

        # Randomly select a block to mask for each sample
        block_indices = torch.randint(0, num_blocks, (batch_size,), device=device)

        # Sample masking ratios
        masking_ratio = self.sample_masking_ratio(batch_size, device, min_ratio, max_ratio)

        # Build masked sequence
        masked_blocks = []
        mask_indicators = []

        for block_idx, block in enumerate(blocks):
            # Which samples should have this block masked?
            should_mask = block_indices == block_idx

            # For samples that should have this block masked, apply masking
            if should_mask.any():
                masked_block = block.clone()
                mask_indicator = torch.zeros_like(block, dtype=torch.bool)

                # Apply masking for selected samples
                for sample_idx in should_mask.nonzero(as_tuple=True)[0]:
                    ratio = masking_ratio[sample_idx]
                    rand = torch.rand(self.block_size, device=device)
                    sample_mask = rand < ratio
                    masked_block[sample_idx, sample_mask] = self.mask_token_id
                    mask_indicator[sample_idx] = sample_mask

                masked_blocks.append(masked_block)
                mask_indicators.append(mask_indicator)
            else:
                # This block is context for all samples
                masked_blocks.append(block)
                mask_indicators.append(torch.zeros_like(block, dtype=torch.bool))

        # Merge back
        masked_input_ids = torch.cat(masked_blocks, dim=1)[:, :seq_len]
        full_mask_indicator = torch.cat(mask_indicators, dim=1)[:, :seq_len]

        return {
            "input_ids": masked_input_ids,
            "target_ids": input_ids,
            "mask_indicator": full_mask_indicator,
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
        """Create attention mask for block-level generation.

        Args:
            seq_len: Total sequence length (context + current block)
            current_block_idx: Index of the block being generated
            device: Device to create mask on

        Returns:
            Attention mask [seq_len, seq_len]
            True means "can attend", False means "cannot attend"
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Number of context blocks
        num_context_blocks = current_block_idx
        context_len = num_context_blocks * self.block_size

        # Context can attend to itself (bidirectionally within each completed block)
        for b in range(num_context_blocks):
            start = b * self.block_size
            end = (b + 1) * self.block_size
            mask[start:end, start:end] = True

        # Context blocks can attend to all previous blocks
        for b in range(num_context_blocks):
            for prev_b in range(b):
                curr_start, curr_end = b * self.block_size, (b + 1) * self.block_size
                prev_start, prev_end = prev_b * self.block_size, (prev_b + 1) * self.block_size
                mask[curr_start:curr_end, prev_start:prev_end] = True

        # Current block can attend to all context
        if context_len < seq_len:
            mask[context_len:, :context_len] = True

            # Current block has bidirectional attention within itself
            mask[context_len:, context_len:] = True

        return mask

    def create_training_mask(
        self,
        seq_len: int,
        block_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Create per-sample training masks.

        During training, each sample may have a different block being trained.

        Args:
            seq_len: Sequence length
            block_indices: [batch] which block is being trained per sample
            device: Device

        Returns:
            [batch, seq_len, seq_len] attention masks
        """
        batch_size = block_indices.shape[0]
        masks = []

        for i in range(batch_size):
            block_idx = block_indices[i].item()
            mask = self.create_mask(seq_len, block_idx, device)
            masks.append(mask)

        return torch.stack(masks)
