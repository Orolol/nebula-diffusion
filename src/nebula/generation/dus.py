"""Dilated Unmasking Scheduler (DUS) for parallel token generation.

DUS addresses the critical insight that confidence-based planners ignore
pairwise token interactions. By unmasking spatially distant tokens together,
we minimize mutual information between parallel predictions.

Key features:
- Partitions positions into non-adjacent dilated groups
- Unmasking spatially distant tokens together reduces prediction errors
- Training-free and model-agnostic
- Combines with confidence thresholding for adaptive step counts

Reference: arXiv:2506.19037
"""

from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F


class DilatedUnmaskingScheduler:
    """Dilated Unmasking Scheduler for parallel demasking.

    Instead of unmasking adjacent tokens (which have high mutual information),
    DUS groups positions into dilated patterns:
        Group 0: [0, 3, 6, 9, ...]
        Group 1: [1, 4, 7, 10, ...]
        Group 2: [2, 5, 8, 11, ...]

    Tokens from the same group are spatially distant, so their predictions
    are more independent.
    """

    def __init__(
        self,
        num_groups: int = 3,
        confidence_threshold: float = 0.5,
        use_kl_divergence: bool = True,
        kl_threshold: float = 0.1,
    ):
        """
        Args:
            num_groups: Number of dilated groups (dilation factor)
            confidence_threshold: Minimum confidence to unmask
            use_kl_divergence: Use KL divergence for stability detection
            kl_threshold: KL threshold for stable predictions
        """
        self.num_groups = num_groups
        self.confidence_threshold = confidence_threshold
        self.use_kl_divergence = use_kl_divergence
        self.kl_threshold = kl_threshold

    def get_dilated_groups(self, seq_len: int, device: torch.device) -> List[torch.Tensor]:
        """Get dilated position groups.

        Args:
            seq_len: Sequence length
            device: Device

        Returns:
            List of position tensors, one per group
        """
        positions = torch.arange(seq_len, device=device)
        groups = []

        for g in range(self.num_groups):
            group_positions = positions[g::self.num_groups]
            groups.append(group_positions)

        return groups

    def select_positions_to_unmask(
        self,
        current_ids: torch.Tensor,
        logits: torch.Tensor,
        mask_token_id: int,
        step: int,
        total_steps: int,
        prev_probs: Optional[torch.Tensor] = None,
        fixed_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select positions to unmask using DUS strategy.

        Args:
            current_ids: Current token IDs [batch, seq_len]
            logits: Model predictions [batch, seq_len, vocab_size]
            mask_token_id: ID of [MASK] token
            step: Current denoising step
            total_steps: Total denoising steps
            prev_probs: Probabilities from previous step for KL computation
            fixed_positions: Positions that shouldn't be unmasked (e.g., prompt)

        Returns:
            positions_to_unmask: [batch, num_to_unmask] position indices
            sampled_tokens: [batch, seq_len] sampled token IDs
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device

        # Compute probabilities and confidence
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values  # [batch, seq_len]

        # Sample tokens
        sampled_tokens = torch.multinomial(
            probs.view(-1, vocab_size), num_samples=1
        ).view(batch_size, seq_len)

        # Get dilated groups
        groups = self.get_dilated_groups(seq_len, device)

        # Select which group to focus on this step
        # Cycle through groups to ensure all positions get attention
        group_idx = step % self.num_groups
        current_group = groups[group_idx]

        # Find masked positions in current group
        is_masked = current_ids == mask_token_id  # [batch, seq_len]

        if fixed_positions is not None:
            is_masked = is_masked & ~fixed_positions

        # Get masked positions that belong to current dilated group
        group_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        group_mask[current_group] = True

        eligible = is_masked & group_mask.unsqueeze(0)  # [batch, seq_len]

        # Apply confidence thresholding
        high_confidence = confidence > self.confidence_threshold

        # Optionally use KL divergence for stability
        if self.use_kl_divergence and prev_probs is not None:
            # Compute KL divergence between current and previous predictions
            kl_div = F.kl_div(
                probs.log(),
                prev_probs,
                reduction="none"
            ).sum(dim=-1)  # [batch, seq_len]

            # Low KL means stable prediction
            is_stable = kl_div < self.kl_threshold
            eligible = eligible & is_stable

        eligible = eligible & high_confidence

        # Determine how many to unmask (progressive schedule)
        progress = (step + 1) / total_steps

        positions_list = []

        for b in range(batch_size):
            eligible_positions = eligible[b].nonzero(as_tuple=True)[0]

            if len(eligible_positions) == 0:
                # Fall back to any masked position in group
                fallback = is_masked[b] & group_mask
                eligible_positions = fallback.nonzero(as_tuple=True)[0]

            if len(eligible_positions) == 0:
                positions_list.append(torch.tensor([], device=device, dtype=torch.long))
                continue

            # Number to unmask this step
            num_masked_in_group = (is_masked[b] & group_mask).sum().item()
            num_to_unmask = max(1, int(num_masked_in_group * progress))
            num_to_unmask = min(num_to_unmask, len(eligible_positions))

            # Select highest confidence among eligible
            eligible_conf = confidence[b, eligible_positions]
            _, top_indices = eligible_conf.topk(num_to_unmask)
            selected = eligible_positions[top_indices]

            positions_list.append(selected)

        # Pad to same length for batching
        max_len = max(len(p) for p in positions_list) if positions_list else 0
        if max_len == 0:
            max_len = 1

        positions_padded = torch.full((batch_size, max_len), -1, device=device, dtype=torch.long)
        for b, pos in enumerate(positions_list):
            if len(pos) > 0:
                positions_padded[b, :len(pos)] = pos

        return positions_padded, sampled_tokens

    def unmask_positions(
        self,
        current_ids: torch.Tensor,
        positions: torch.Tensor,
        sampled_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Apply unmasking at selected positions.

        Args:
            current_ids: Current sequence [batch, seq_len]
            positions: Positions to unmask [batch, num_positions] (-1 for padding)
            sampled_tokens: Sampled tokens [batch, seq_len]

        Returns:
            Updated sequence [batch, seq_len]
        """
        result = current_ids.clone()
        batch_size = current_ids.shape[0]

        for b in range(batch_size):
            valid_positions = positions[b][positions[b] >= 0]
            if len(valid_positions) > 0:
                result[b, valid_positions] = sampled_tokens[b, valid_positions]

        return result


class ConfidenceKLScheduler:
    """Confidence + KL divergence based scheduler (KLASS-style).

    Uses token-level KL divergence between consecutive steps to identify
    stable, high-confidence predictions.

    Reference: KLASS (arXiv:2511.05664)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        kl_threshold: float = 0.05,
        min_unmask_ratio: float = 0.1,
    ):
        self.confidence_threshold = confidence_threshold
        self.kl_threshold = kl_threshold
        self.min_unmask_ratio = min_unmask_ratio
        self.prev_probs = None

    def reset(self):
        """Reset state for new generation."""
        self.prev_probs = None

    def select_and_unmask(
        self,
        current_ids: torch.Tensor,
        logits: torch.Tensor,
        mask_token_id: int,
        fixed_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Select positions to unmask based on confidence and KL stability.

        Args:
            current_ids: [batch, seq_len]
            logits: [batch, seq_len, vocab_size]
            mask_token_id: [MASK] token ID
            fixed_positions: Positions not to unmask

        Returns:
            updated_ids: [batch, seq_len]
            probs: Current probabilities (for next step's KL)
            should_stop: Whether all masks are unmasked
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device

        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values

        # Sample tokens
        sampled = torch.multinomial(
            probs.view(-1, vocab_size), num_samples=1
        ).view(batch_size, seq_len)

        is_masked = current_ids == mask_token_id
        if fixed_positions is not None:
            is_masked = is_masked & ~fixed_positions

        # Compute eligibility
        eligible = is_masked & (confidence > self.confidence_threshold)

        if self.prev_probs is not None:
            kl = F.kl_div(probs.log(), self.prev_probs, reduction="none").sum(-1)
            is_stable = kl < self.kl_threshold
            eligible = eligible & is_stable

        # Ensure minimum progress
        num_masked = is_masked.sum(dim=1, keepdim=True).float()
        min_unmask = (num_masked * self.min_unmask_ratio).long().clamp(min=1)

        result = current_ids.clone()

        for b in range(batch_size):
            eligible_pos = eligible[b].nonzero(as_tuple=True)[0]

            if len(eligible_pos) < min_unmask[b, 0].item():
                # Fall back to highest confidence masked positions
                masked_pos = is_masked[b].nonzero(as_tuple=True)[0]
                if len(masked_pos) > 0:
                    masked_conf = confidence[b, masked_pos]
                    num_select = min(min_unmask[b, 0].item(), len(masked_pos))
                    _, top_idx = masked_conf.topk(num_select)
                    eligible_pos = masked_pos[top_idx]

            if len(eligible_pos) > 0:
                result[b, eligible_pos] = sampled[b, eligible_pos]

        self.prev_probs = probs.detach()

        # Check if done
        still_masked = (result == mask_token_id)
        if fixed_positions is not None:
            still_masked = still_masked & ~fixed_positions
        should_stop = not still_masked.any()

        return result, probs, should_stop
