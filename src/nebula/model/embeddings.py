"""Token and positional embeddings, normalization layers."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Faster than LayerNorm and equally effective for transformers.
    Used in LLaMA, Mistral, Gemma, etc.

    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation function with gated linear unit.

    SwiGLU(x) = Swish(xW) ⊙ (xV)

    More effective than GELU for LLMs.
    Used in LLaMA, Mistral, PaLM, etc.

    Reference: https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        # SwiGLU uses 2/3 of the intermediate dim for gate and up projections
        # to match parameter count with standard FFN
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=bias)  # gate
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=bias)  # down
        self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=bias)  # up
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w2.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w3.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(x @ W1) * (x @ W3) @ W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TokenEmbedding(nn.Module):
    """Token embedding layer."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.hidden_dim = hidden_dim

        # Initialize with small values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]

        Returns:
            embeddings: [batch_size, seq_len, hidden_dim]
        """
        input_ids = input_ids.clone()
        return self.embedding(input_ids)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encodings.

    Following "Attention Is All You Need" (Vaswani et al., 2017).
    """

    def __init__(self, hidden_dim: int, max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but should be saved/loaded)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_seq_len, hidden_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]

        Returns:
            Tensor with positional encoding added: [batch_size, seq_len, hidden_dim]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encodings."""

    def __init__(self, hidden_dim: int, max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]

        Returns:
            Tensor with positional encoding added: [batch_size, seq_len, hidden_dim]
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.embedding(positions)
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Applies rotary position embeddings directly to Q and K in attention.
    This provides better length generalization than absolute position embeddings.

    Reference: RoFormer (arXiv:2104.09864)
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        """
        Args:
            head_dim: Dimension per attention head
            max_seq_len: Maximum sequence length
            base: Base for computing inverse frequencies
        """
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute inverse frequencies: theta_i = 1 / (base^(2i/d))
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin for max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Precompute cos and sin values for efficiency."""
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)  # [seq_len, head_dim/2]

        # Duplicate for full head_dim
        freqs = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]

        cos = freqs.cos()
        sin = freqs.sin()

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of x for RoPE."""
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to Q and K.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_heads, seq_len, head_dim]
            seq_len: Current sequence length
            offset: Position offset for caching during generation

        Returns:
            q_embed: Rotated queries [batch, num_heads, seq_len, head_dim]
            k_embed: Rotated keys [batch, num_heads, seq_len, head_dim]
        """
        # Extend cache if needed
        if seq_len + offset > self.cos_cached.shape[0]:
            self._build_cache(seq_len + offset)

        # Get cos/sin for current positions
        cos = self.cos_cached[offset : offset + seq_len].to(q.dtype)  # [seq_len, head_dim]
        sin = self.sin_cached[offset : offset + seq_len].to(q.dtype)

        # Reshape for broadcasting: [1, 1, seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply rotary embedding
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed
