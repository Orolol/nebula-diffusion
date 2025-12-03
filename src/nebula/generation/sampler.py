"""Generation with Dilated Unmasking Scheduler (DUS) for block diffusion."""

from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..model.transformer import HybridDiffusionTransformer
from ..model.block_diffusion import BlockDiffusion
from .dus import DilatedUnmaskingScheduler, ConfidenceKLScheduler


class BlockDiffusionSampler:
    """Sampler for block diffusion with DUS.

    Generates text block-by-block autoregressively, using
    diffusion within each block with Dilated Unmasking Scheduler.
    """

    def __init__(
        self,
        model: HybridDiffusionTransformer,
        mask_token_id: int,
        block_size: int = 8,
        num_steps_per_block: int = 16,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        use_dus: bool = True,
        dus_num_groups: int = 3,
        confidence_threshold: float = 0.5,
    ):
        """
        Args:
            model: The hybrid diffusion transformer
            mask_token_id: ID of [MASK] token
            block_size: Number of tokens per block
            num_steps_per_block: Diffusion steps per block
            temperature: Sampling temperature
            top_k: Top-k sampling (0 to disable)
            top_p: Nucleus sampling (1.0 to disable)
            use_dus: Whether to use Dilated Unmasking Scheduler
            dus_num_groups: Number of dilated groups
            confidence_threshold: Threshold for confident predictions
        """
        self.model = model
        self.mask_token_id = mask_token_id
        self.block_size = block_size
        self.num_steps_per_block = num_steps_per_block
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.use_dus = use_dus

        # DUS scheduler
        if use_dus:
            self.dus = DilatedUnmaskingScheduler(
                num_groups=dus_num_groups,
                confidence_threshold=confidence_threshold,
            )
        else:
            self.dus = None

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
            generated = self._generate_block(
                generated,
                block_start,
                block_end,
            )

        return generated

    def _generate_block(
        self,
        sequence: torch.Tensor,
        block_start: int,
        block_end: int,
    ) -> torch.Tensor:
        """Generate a single block using diffusion with DUS.

        Args:
            sequence: Current sequence [batch, seq_len]
            block_start: Start position of block
            block_end: End position of block

        Returns:
            Updated sequence with block filled in
        """
        batch_size = sequence.shape[0]
        device = sequence.device
        block_size = block_end - block_start

        # Track fixed positions (everything except current block)
        is_fixed = torch.ones_like(sequence, dtype=torch.bool)
        is_fixed[:, block_start:block_end] = False

        prev_probs = None

        for step in range(self.num_steps_per_block):
            # Get model predictions
            result = self.model(sequence)
            logits = result["logits"]

            # Apply sampling
            probs = self._compute_probs(logits)

            if self.use_dus and self.dus is not None:
                # Use DUS for position selection
                positions, sampled = self.dus.select_positions_to_unmask(
                    sequence,
                    logits[:, block_start:block_end],
                    self.mask_token_id,
                    step,
                    self.num_steps_per_block,
                    prev_probs[:, block_start:block_end] if prev_probs is not None else None,
                    None,  # No fixed within block
                )

                # Apply unmasking
                for b in range(batch_size):
                    valid = positions[b][positions[b] >= 0]
                    if len(valid) > 0:
                        global_pos = valid + block_start
                        sequence[b, global_pos] = sampled[b, valid]
            else:
                # Standard confidence-based unmasking
                sampled_tokens = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), num_samples=1
                ).view(batch_size, -1)

                confidence = probs.max(dim=-1).values
                block_mask = sequence[:, block_start:block_end] == self.mask_token_id

                unmask_fraction = (step + 1) / self.num_steps_per_block

                for b in range(batch_size):
                    masked_local = block_mask[b].nonzero(as_tuple=True)[0]
                    if len(masked_local) == 0:
                        continue

                    conf_local = confidence[b, block_start:block_end][masked_local]
                    num_unmask = max(1, int(len(masked_local) * unmask_fraction))
                    _, top_idx = conf_local.topk(min(num_unmask, len(masked_local)))

                    positions_local = masked_local[top_idx]
                    positions_global = positions_local + block_start
                    sequence[b, positions_global] = sampled_tokens[b, positions_global]

            prev_probs = probs.detach()

        # Final fill for any remaining masks
        block_masked = sequence[:, block_start:block_end] == self.mask_token_id
        if block_masked.any():
            result = self.model(sequence)
            probs = self._compute_probs(result["logits"])
            sampled = torch.multinomial(
                probs.view(-1, probs.shape[-1]), num_samples=1
            ).view(batch_size, -1)

            for b in range(batch_size):
                still_masked = (sequence[b, block_start:block_end] == self.mask_token_id)
                if still_masked.any():
                    positions = still_masked.nonzero(as_tuple=True)[0] + block_start
                    sequence[b, positions] = sampled[b, positions]

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
        # Estimate blocks from steps
        block_size = 8
        steps_per_block = max(8, num_steps // 8)

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
