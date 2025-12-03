"""Bidirectional multi-head attention for diffusion models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class BidirectionalAttention(nn.Module):
    """Multi-head bidirectional attention.

    Every token attends to every other token (no causal mask).
    Uses PyTorch's scaled_dot_product_attention for efficiency (FlashAttention when available).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        inner_dim = num_heads * head_dim

        self.q_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, hidden_dim, bias=False)

        self.dropout = dropout

        # Initialize projections
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
                           where True/1 indicates positions to attend to

        Returns:
            Output tensor [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: [batch, seq, heads, head_dim]
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        # Prepare attention mask if provided
        # scaled_dot_product_attention expects mask where True means "do NOT attend"
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: [batch, seq] with True for valid positions
            # Convert to [batch, 1, 1, seq] for broadcasting
            # Invert: True (valid) -> False (attend), False (pad) -> True (mask out)
            attn_mask = ~attention_mask.bool()
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq]

        # Scaled dot-product attention (uses FlashAttention when available)
        # No causal mask - bidirectional attention
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # Bidirectional!
        )

        # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        out = rearrange(out, "b h s d -> b s (h d)")

        # Output projection
        out = self.out_proj(out)

        return out
