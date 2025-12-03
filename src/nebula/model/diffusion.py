"""Masked diffusion logic for LLaDA-style training and inference."""

from typing import Tuple

import torch
import torch.nn.functional as F


class MaskedDiffusion:
    """Implements LLaDA's masked diffusion process.

    Key concepts:
    - Forward process: Replace tokens with [MASK] according to ratio t
    - Training: Sample t uniformly from [0,1], compute CE loss on masked tokens
    - Inference: Iteratively denoise from fully masked to unmasked

    The forward process is:
        q(x_t | x_0) = (1 - t) * x_0 + t * [MASK]

    In practice, for each position independently:
    - With probability t, replace token with [MASK]
    - With probability 1 - t, keep original token
    """

    def __init__(self, mask_token_id: int, vocab_size: int):
        """
        Args:
            mask_token_id: The ID of the [MASK] token
            vocab_size: Size of the vocabulary
        """
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size

    def sample_masking_ratio(
        self,
        batch_size: int,
        device: torch.device,
        min_ratio: float = 0.0,
        max_ratio: float = 1.0,
    ) -> torch.Tensor:
        """Sample masking ratio uniformly from [min_ratio, max_ratio] for each batch item.

        Args:
            batch_size: Number of samples
            device: Device to create tensor on
            min_ratio: Minimum masking ratio
            max_ratio: Maximum masking ratio

        Returns:
            Tensor of shape [batch_size] with masking ratios
        """
        return torch.rand(batch_size, device=device) * (max_ratio - min_ratio) + min_ratio

    def apply_mask(
        self,
        input_ids: torch.Tensor,
        masking_ratio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random masking to input tokens.

        Args:
            input_ids: Original token IDs [batch_size, seq_len]
            masking_ratio: Masking ratio for each batch item [batch_size] or scalar

        Returns:
            masked_ids: Input with some tokens replaced by [MASK]
            mask_indicator: Boolean tensor indicating which positions are masked
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Generate random values for each position
        rand = torch.rand(batch_size, seq_len, device=device)

        # Expand masking_ratio to [batch_size, 1] for broadcasting if needed
        if masking_ratio.dim() == 0:
            # Scalar - same ratio for all
            mask_indicator = rand < masking_ratio
        else:
            # Per-batch ratio
            mask_indicator = rand < masking_ratio.unsqueeze(1)

        # Apply masking
        masked_ids = input_ids.clone()
        masked_ids[mask_indicator] = self.mask_token_id

        return masked_ids, mask_indicator

    def compute_loss(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        mask_indicator: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss only on masked positions.

        This is the core LLaDA training objective: predict the original tokens
        at masked positions.

        Args:
            logits: Model output [batch_size, seq_len, vocab_size]
            target_ids: Original (unmasked) token IDs [batch_size, seq_len]
            mask_indicator: Boolean tensor [batch_size, seq_len] indicating masked positions

        Returns:
            Scalar loss (mean cross-entropy over masked positions)
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Flatten for cross-entropy
        logits_flat = logits.view(-1, vocab_size)  # [batch*seq, vocab]
        targets_flat = target_ids.view(-1)  # [batch*seq]
        mask_flat = mask_indicator.view(-1)  # [batch*seq]

        # Compute per-token loss
        loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")

        # Average only over masked positions
        masked_loss = loss_per_token[mask_flat]

        if masked_loss.numel() == 0:
            # No masked positions (shouldn't happen in practice)
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return masked_loss.mean()

    def compute_loss_with_weights(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        mask_indicator: torch.Tensor,
        masking_ratio: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss with optional weighting by masking ratio.

        Some diffusion models weight the loss inversely by t to give more weight
        to predictions at lower noise levels.

        Args:
            logits: Model output [batch_size, seq_len, vocab_size]
            target_ids: Original token IDs [batch_size, seq_len]
            mask_indicator: Boolean tensor indicating masked positions
            masking_ratio: Masking ratio used for each sample [batch_size]

        Returns:
            Weighted scalar loss
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Compute per-token loss
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = target_ids.view(-1)
        loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        loss_per_token = loss_per_token.view(batch_size, seq_len)

        # Mask out non-masked positions
        loss_per_token = loss_per_token * mask_indicator.float()

        # Sum per sample and normalize by number of masked tokens
        num_masked = mask_indicator.sum(dim=1).clamp(min=1)  # [batch_size]
        loss_per_sample = loss_per_token.sum(dim=1) / num_masked  # [batch_size]

        # Average across batch
        return loss_per_sample.mean()

    @torch.no_grad()
    def get_confidence_scores(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Get confidence scores (max probability) for each position.

        Args:
            logits: Model output [batch_size, seq_len, vocab_size]
            temperature: Temperature for softmax

        Returns:
            Confidence scores [batch_size, seq_len]
        """
        probs = F.softmax(logits / temperature, dim=-1)
        confidence = probs.max(dim=-1).values
        return confidence
