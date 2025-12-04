"""Multi-Token Prediction (MTP) Head.

Implements DeepSeek-V3 style multi-token prediction:
- Predicts N tokens simultaneously during training
- Auxiliary loss that improves representation learning
- Enables speculative decoding at inference

Reference: DeepSeek-V3 (arXiv:2412.19437)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiTokenPredictionHead(nn.Module):
    """Multi-Token Prediction head for auxiliary training and speculative decoding.

    Key features:
    - Predicts next N tokens in parallel
    - Shared embedding projection with different prediction heads
    - Auxiliary training loss improves main model representations
    - Can be used for speculative decoding at inference
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        num_tokens: int = 2,
        share_embeddings: bool = True,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """
        Args:
            hidden_dim: Model hidden dimension
            vocab_size: Vocabulary size
            num_tokens: Number of tokens to predict (default 2 like DeepSeek-V3)
            share_embeddings: Whether to share embedding weights with LM head
            intermediate_dim: Optional intermediate projection dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.share_embeddings = share_embeddings

        # Optional intermediate projection
        if intermediate_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            proj_dim = intermediate_dim
        else:
            self.projection = None
            proj_dim = hidden_dim

        # Separate head for each future token position
        # Head 0 predicts next token, Head 1 predicts token after that, etc.
        self.heads = nn.ModuleList([
            nn.Linear(proj_dim, vocab_size, bias=False)
            for _ in range(num_tokens)
        ])

        # Positional offset embeddings (help distinguish which future position)
        self.position_offsets = nn.Parameter(
            torch.zeros(num_tokens, hidden_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for head in self.heads:
            nn.init.normal_(head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_offsets, mean=0.0, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for multi-token prediction.

        Args:
            hidden_states: Model output [batch, seq_len, hidden_dim]
            target_ids: Optional target tokens for loss [batch, seq_len]

        Returns:
            logits: Predictions for each future token [batch, seq_len, num_tokens, vocab_size]
            loss: Optional MTP loss if targets provided
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Apply optional projection
        if self.projection is not None:
            h = self.projection(hidden_states)
        else:
            h = hidden_states
        h = h.clone()

        # Predict each future token
        all_logits = []

        for token_idx in range(self.num_tokens):
            # Add position-specific offset
            h_offset = h.clone() + self.position_offsets[token_idx]

            # Get logits for this future position
            logits = self.heads[token_idx](h_offset)  # [batch, seq, vocab]
            all_logits.append(logits)

        # Stack: [batch, seq, num_tokens, vocab]
        all_logits = torch.stack(all_logits, dim=2)

        # Compute loss if targets provided
        loss = None
        if target_ids is not None:
            loss = self._compute_loss(all_logits, target_ids)

        return all_logits, loss

    def _compute_loss(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MTP loss.

        For each position i, we predict tokens at i+1, i+2, ..., i+num_tokens.

        Args:
            logits: [batch, seq_len, num_tokens, vocab_size]
            target_ids: [batch, seq_len]

        Returns:
            Scalar loss
        """
        batch_size, seq_len, num_tokens, vocab_size = logits.shape
        device = logits.device

        total_loss = 0.0
        num_valid = 0

        for token_offset in range(num_tokens):
            # For position i, predict token at position i + token_offset + 1
            # Shifted logits: positions 0 to seq_len - token_offset - 1
            # Shifted targets: positions token_offset + 1 to seq_len

            if seq_len <= token_offset + 1:
                continue

            # Get predictions for this offset
            pred_logits = logits[:, :-token_offset - 1, token_offset, :]  # [batch, valid_seq, vocab]
            pred_targets = target_ids[:, token_offset + 1:]  # [batch, valid_seq]

            # Flatten and compute CE
            pred_logits = pred_logits.reshape(-1, vocab_size)
            pred_targets = pred_targets.reshape(-1)

            loss = F.cross_entropy(pred_logits, pred_targets, reduction="mean")
            total_loss = total_loss + loss
            num_valid += 1

        if num_valid > 0:
            return total_loss / num_valid

        return torch.tensor(0.0, device=device, requires_grad=True)

    @torch.no_grad()
    def speculative_decode(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate speculative token predictions for verification.

        Args:
            hidden_states: [batch, seq_len, hidden_dim] (typically just last position)
            temperature: Sampling temperature

        Returns:
            predicted_tokens: [batch, num_tokens] speculative predictions
            confidences: [batch, num_tokens] prediction confidences
        """
        logits, _ = self.forward(hidden_states)  # [batch, seq, num_tokens, vocab]

        # Take last position's predictions
        logits = logits[:, -1, :, :]  # [batch, num_tokens, vocab]

        # Apply temperature and get probabilities
        probs = F.softmax(logits / temperature, dim=-1)

        # Sample tokens
        batch_size, num_tokens, vocab_size = probs.shape
        probs_flat = probs.view(-1, vocab_size)
        tokens_flat = torch.multinomial(probs_flat, num_samples=1)
        predicted_tokens = tokens_flat.view(batch_size, num_tokens)

        # Get confidences
        confidences = probs.max(dim=-1).values  # [batch, num_tokens]

        return predicted_tokens, confidences


class MTPAuxiliaryLoss(nn.Module):
    """Wrapper for adding MTP as auxiliary loss during training."""

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        num_tokens: int = 2,
        loss_weight: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Model hidden dimension
            vocab_size: Vocabulary size
            num_tokens: Number of future tokens to predict
            loss_weight: Weight for MTP loss relative to main loss
        """
        super().__init__()

        self.mtp_head = MultiTokenPredictionHead(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            num_tokens=num_tokens,
        )
        self.loss_weight = loss_weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted MTP auxiliary loss.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            target_ids: [batch, seq_len]

        Returns:
            Weighted MTP loss
        """
        _, loss = self.mtp_head(hidden_states, target_ids)

        if loss is not None:
            return self.loss_weight * loss

        return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)
