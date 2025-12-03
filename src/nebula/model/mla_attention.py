"""Multi-head Latent Attention (MLA) for efficient full attention.

MLA achieves 93% KV cache reduction through low-rank key-value compression.
Used in 25% of layers for strong recall capability while maintaining efficiency.

Reference: DeepSeek-V2 (arXiv:2405.04434)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadLatentAttention(nn.Module):
    """Multi-head Latent Attention with low-rank KV compression.

    Key features:
    - Compresses K and V into low-dimensional latent vectors
    - Achieves ~93% KV cache reduction
    - Full bidirectional attention for diffusion models
    - Uses FlashAttention when available
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        kv_latent_dim: int = 64,
        q_latent_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """
        Args:
            hidden_dim: Input/output dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head for queries
            kv_latent_dim: Latent dimension for KV compression
            q_latent_dim: Optional latent dimension for Q (if None, no compression)
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_latent_dim = kv_latent_dim
        self.q_latent_dim = q_latent_dim or head_dim
        self.scale = head_dim ** -0.5

        # Query projection
        # Can optionally compress queries too
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)

        # KV compression: hidden -> latent
        # Single down-projection shared for K and V
        self.kv_down_proj = nn.Linear(hidden_dim, kv_latent_dim, bias=False)

        # Up-projection from latent to K and V heads
        # K and V are decompressed from the same latent representation
        self.k_up_proj = nn.Linear(kv_latent_dim, num_heads * head_dim, bias=False)
        self.v_up_proj = nn.Linear(kv_latent_dim, num_heads * head_dim, bias=False)

        # Optional: learnable bias for up-projections
        self.k_bias = nn.Parameter(torch.zeros(num_heads * head_dim))
        self.v_bias = nn.Parameter(torch.zeros(num_heads * head_dim))

        # Output projection
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        # RoPE can be added here if positional encoding is needed in attention
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for proj in [self.q_proj, self.kv_down_proj, self.k_up_proj, self.v_up_proj, self.out_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)

        nn.init.zeros_(self.k_bias)
        nn.init.zeros_(self.v_bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
        return_kv_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional KV caching.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            attention_mask: Optional mask [batch, seq_len]
            kv_cache: Optional cached KV latents [batch, cached_len, kv_latent_dim]
            return_kv_cache: Whether to return updated KV cache

        Returns:
            output: [batch, seq_len, hidden_dim]
            kv_cache: Optional updated cache
        """
        batch_size, seq_len, _ = x.shape

        # Compute queries
        q = self.q_proj(x)  # [batch, seq, num_heads * head_dim]
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)

        # Compress to KV latent space
        kv_latent = self.kv_down_proj(x)  # [batch, seq, kv_latent_dim]

        # Handle KV caching for autoregressive generation
        if kv_cache is not None:
            kv_latent = torch.cat([kv_cache, kv_latent], dim=1)

        # Decompress to K and V
        k = self.k_up_proj(kv_latent) + self.k_bias  # [batch, seq, num_heads * head_dim]
        v = self.v_up_proj(kv_latent) + self.v_bias

        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        # Prepare attention mask if provided
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: [batch, seq] -> [batch, 1, 1, seq]
            # True means "do attend", False means "don't attend"
            attn_mask = ~attention_mask.bool()
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)

        # Scaled dot-product attention (uses FlashAttention when available)
        # is_causal=False for bidirectional (diffusion models need this!)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,  # Bidirectional attention for diffusion
        )

        # Reshape and project output
        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.out_proj(out)
        out = self.dropout(out)

        if return_kv_cache:
            return out, kv_latent

        return out, None

    def get_kv_cache_size(self, seq_len: int, batch_size: int = 1) -> int:
        """Calculate KV cache size in bytes (for comparison)."""
        # MLA: only store latent representation
        mla_size = batch_size * seq_len * self.kv_latent_dim * 2  # float16

        # Standard MHA would need:
        # mha_size = batch_size * seq_len * self.num_heads * self.head_dim * 2 * 2

        return mla_size

    def compression_ratio(self) -> float:
        """Calculate KV compression ratio."""
        full_kv_dim = self.num_heads * self.head_dim * 2  # K and V
        compressed_dim = self.kv_latent_dim
        return 1 - (compressed_dim / full_kv_dim)


class MLABlock(nn.Module):
    """Transformer block with Multi-head Latent Attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        kv_latent_dim: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadLatentAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            kv_latent_dim=kv_latent_dim,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        intermediate_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
        return_kv_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            attention_mask: Optional mask
            kv_cache: Optional KV cache
            return_kv_cache: Whether to return cache

        Returns:
            output: [batch, seq_len, hidden_dim]
            kv_cache: Optional updated cache
        """
        # Pre-norm attention with residual
        attn_out, new_cache = self.attn(
            self.norm1(x),
            attention_mask,
            kv_cache,
            return_kv_cache,
        )
        x = x + attn_out

        # Pre-norm MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x, new_cache
