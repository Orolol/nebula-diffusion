"""Generation with Dilated Unmasking Scheduler (DUS) for block diffusion."""

from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..model.transformer import HybridDiffusionTransformer
from ..model.block_diffusion import BlockDiffusion
from .dus import DilatedUnmaskingScheduler, ConfidenceKLScheduler


class BlockDiffusionSampler:
    """Optimized sampler for block diffusion with DUS.

    Generates text block-by-block autoregressively, using
    diffusion within each block with Dilated Unmasking Scheduler.
    """

    def __init__(
        self,
        model: HybridDiffusionTransformer,
        mask_token_id: int,
        block_size: int = 8,
        num_steps_per_block: int = 8,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        use_dus: bool = True,
        dus_num_groups: int = 3,
        confidence_threshold: float = 0.5,
    ):
        self.model = model
        self.mask_token_id = mask_token_id
        self.block_size = block_size
        self.num_steps_per_block = num_steps_per_block
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.use_dus = use_dus
        self.confidence_threshold = confidence_threshold

        if use_dus:
            self.dus = DilatedUnmaskingScheduler(
                num_groups=dus_num_groups,
                confidence_threshold=confidence_threshold,
            )
        else:
            self.dus = None

        # Pre-compile the generation function if using torch 2.0+
        self._compiled = False

        # Cache the underlying model (handles torch.compile wrapper)
        self._orig_model = None

    def _get_logits(self, sequence: torch.Tensor) -> torch.Tensor:
        """Get logits from model, handling torch.compile wrapper."""
        if self._orig_model is None:
            self._orig_model = getattr(self.model, '_orig_mod', self.model)

        if hasattr(self._orig_model, 'generate_forward'):
            return self._orig_model.generate_forward(sequence)
        else:
            return self.model(sequence)["logits"]

    @torch.no_grad()
    def generate(
        self,
        num_blocks: int = 16,
        batch_size: int = 1,
        prompt_ids: Optional[torch.Tensor] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Generate text block-by-block.

        Args:
            num_blocks: Number of blocks to generate
            batch_size: Number of sequences in parallel
            prompt_ids: Optional prompt tokens [batch, prompt_len]
            show_progress: Show progress bar

        Returns:
            Generated tokens [batch, num_blocks * block_size]
        """
        device = next(self.model.parameters()).device
        self.model.eval()

        total_len = num_blocks * self.block_size

        # Initialize with prompt or empty
        if prompt_ids is not None:
            prompt_ids = prompt_ids.to(device)
            if prompt_ids.dim() == 1:
                prompt_ids = prompt_ids.unsqueeze(0).expand(batch_size, -1)
            prompt_len = prompt_ids.size(1)

            # Pad prompt to block boundary
            if prompt_len % self.block_size != 0:
                pad_len = self.block_size - (prompt_len % self.block_size)
                prompt_ids = F.pad(prompt_ids, (0, pad_len), value=self.mask_token_id)
                prompt_len = prompt_ids.size(1)

            num_prompt_blocks = prompt_len // self.block_size
            start_block = num_prompt_blocks
        else:
            prompt_ids = None
            prompt_len = 0
            start_block = 0

        # Build sequence
        generated = torch.full(
            (batch_size, total_len), self.mask_token_id,
            dtype=torch.long, device=device
        )

        if prompt_ids is not None:
            generated[:, :prompt_len] = prompt_ids

        # Generate blocks
        iterator = range(start_block, num_blocks)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating blocks")

        for block_idx in iterator:
            block_start = block_idx * self.block_size
            block_end = block_start + self.block_size

            # Generate this block with diffusion
            generated = self._generate_block_fast(
                generated,
                block_start,
                block_end,
            )

        return generated

    def _generate_block_fast(
        self,
        sequence: torch.Tensor,
        block_start: int,
        block_end: int,
    ) -> torch.Tensor:
        """Generate a single block using optimized diffusion.

        Fully vectorized - no Python loops over batch dimension.
        Uses confidence-based progressive unmasking.
        """
        batch_size = sequence.shape[0]
        device = sequence.device
        block_size = block_end - block_start

        # Number of tokens to unmask per step (evenly distributed)
        tokens_per_step = max(1, block_size // self.num_steps_per_block)

        for step in range(self.num_steps_per_block):
            # Get model predictions
            logits = self._get_logits(sequence)

            # Apply temperature
            block_logits = logits[:, block_start:block_end] / self.temperature

            # Apply top-k filtering if specified
            if self.top_k > 0:
                topk_vals = torch.topk(block_logits, self.top_k, dim=-1).values
                threshold = topk_vals[..., -1:]
                block_logits = block_logits.masked_fill(block_logits < threshold, float("-inf"))

            probs = F.softmax(block_logits, dim=-1)

            # Sample tokens - use argmax for high confidence, multinomial otherwise
            if self.temperature < 0.5:
                sampled = probs.argmax(dim=-1)  # [batch, block_size]
            else:
                sampled = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), num_samples=1
                ).view(batch_size, block_size)

            # Get confidence and mask
            confidence = probs.max(dim=-1).values  # [batch, block_size]
            block_ids = sequence[:, block_start:block_end]
            is_masked = block_ids == self.mask_token_id  # [batch, block_size]

            # Early exit if fully unmasked
            if not is_masked.any():
                break

            # Progressive schedule: unmask more tokens as we progress
            progress = (step + 1) / self.num_steps_per_block
            target_unmasked = int(block_size * progress)

            # Set confidence of already-unmasked positions to -inf
            masked_confidence = confidence.masked_fill(~is_masked, float("-inf"))

            # Get threshold for top-k confidence (vectorized across batch)
            # Sort and get the k-th highest confidence per batch
            sorted_conf, _ = masked_confidence.sort(dim=-1, descending=True)

            # Number to unmask this step
            num_to_unmask = min(tokens_per_step, target_unmasked)
            num_to_unmask = max(1, num_to_unmask)

            # Get threshold - the num_to_unmask-th highest confidence
            threshold_idx = min(num_to_unmask - 1, block_size - 1)
            thresholds = sorted_conf[:, threshold_idx:threshold_idx+1]  # [batch, 1]

            # Unmask positions above threshold (vectorized)
            should_unmask = (masked_confidence >= thresholds) & is_masked

            # Apply unmasking
            sequence[:, block_start:block_end] = torch.where(
                should_unmask,
                sampled,
                block_ids
            )

        # Final pass - unmask any remaining masks
        block_ids = sequence[:, block_start:block_end]
        remaining_masks = block_ids == self.mask_token_id

        if remaining_masks.any():
            logits = self._get_logits(sequence)
            block_logits = logits[:, block_start:block_end] / self.temperature
            probs = F.softmax(block_logits, dim=-1)

            if self.temperature < 0.5:
                sampled = probs.argmax(dim=-1)
            else:
                sampled = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), num_samples=1
                ).view(batch_size, block_size)

            sequence[:, block_start:block_end] = torch.where(
                remaining_masks,
                sampled,
                block_ids
            )

        return sequence

    def _generate_block_oneshot(
        self,
        sequence: torch.Tensor,
        block_start: int,
        block_end: int,
    ) -> torch.Tensor:
        """Generate a single block in one forward pass.

        Fastest possible generation - unmasks all tokens at once.
        Quality may be lower than iterative approach.
        """
        batch_size = sequence.shape[0]
        block_size = block_end - block_start

        # Single forward pass (optimized)
        logits = self._get_logits(sequence)
        block_logits = logits[:, block_start:block_end] / self.temperature

        if self.top_k > 0:
            topk_vals = torch.topk(block_logits, self.top_k, dim=-1).values
            threshold = topk_vals[..., -1:]
            block_logits = block_logits.masked_fill(block_logits < threshold, float("-inf"))

        probs = F.softmax(block_logits, dim=-1)

        # Sample all tokens at once
        if self.temperature < 0.5:
            sampled = probs.argmax(dim=-1)
        else:
            sampled = torch.multinomial(
                probs.view(-1, probs.shape[-1]), num_samples=1
            ).view(batch_size, block_size)

        # Replace masked positions
        block_ids = sequence[:, block_start:block_end]
        is_masked = block_ids == self.mask_token_id

        sequence[:, block_start:block_end] = torch.where(
            is_masked,
            sampled,
            block_ids
        )

        return sequence

    def _compute_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature, top-k, top-p filtering."""
        logits = logits / self.temperature

        if self.top_k > 0:
            threshold = torch.topk(logits, self.top_k, dim=-1).values[..., -1, None]
            logits = torch.where(logits < threshold, float("-inf"), logits)

        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative > self.top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            indices_remove = remove.scatter(-1, sorted_indices, remove)
            logits = torch.where(indices_remove, float("-inf"), logits)

        return F.softmax(logits, dim=-1)


# Backwards compatibility
class DiffusionSampler(BlockDiffusionSampler):
    """Alias for BlockDiffusionSampler."""

    def __init__(
        self,
        model: HybridDiffusionTransformer,
        mask_token_id: int,
        num_steps: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ):
        block_size = 8
        steps_per_block = max(4, num_steps // 16)  # Reduced steps

        super().__init__(
            model=model,
            mask_token_id=mask_token_id,
            block_size=block_size,
            num_steps_per_block=steps_per_block,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    @torch.no_grad()
    def generate(
        self,
        seq_len: int = 128,
        batch_size: int = 1,
        prompt_ids: Optional[torch.Tensor] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Generate with sequence length instead of block count."""
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        result = super().generate(
            num_blocks=num_blocks,
            batch_size=batch_size,
            prompt_ids=prompt_ids,
            show_progress=show_progress,
        )
        return result[:, :seq_len]
