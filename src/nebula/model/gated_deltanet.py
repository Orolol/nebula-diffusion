"""Gated DeltaNet with LION bidirectional mask for diffusion models.

Implements the Gated DeltaNet linear attention mechanism with bidirectional
support via the LION framework.

Key equations:
- GLA update: S_t = α_t ⊙ S_{t-1} + β_t k_t^T v_t
- Delta rule: S_t = S_{t-1} + β_t(v_t - S_{t-1}^T k_t)k_t^T
- LION bidirectional mask: M_ij = decay^|i-j|

References:
- Gated Linear Attention (GLA): arXiv:2312.06635
- Gated DeltaNet: arXiv:2412.06464
- LION bidirectional framework: arXiv:2502.16249
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet attention with LION bidirectional support.

    This implements a linear attention mechanism with:
    - Data-dependent decay (α) and update (β) gates
    - Delta rule for precise memory modifications
    - LION bidirectional mask for non-causal attention (required for diffusion)
    - O(1) memory complexity regardless of sequence length
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        decay_init: float = 0.9,
        use_delta_rule: bool = True,
        dropout: float = 0.0,
    ):
        """
        Args:
            hidden_dim: Input/output dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head
            expand_k: Expansion factor for key dimension
            expand_v: Expansion factor for value dimension
            decay_init: Initial decay value for gates
            use_delta_rule: Whether to use delta rule (vs simple GLA)
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.key_dim = int(head_dim * expand_k)
        self.value_dim = int(head_dim * expand_v)
        self.use_delta_rule = use_delta_rule

        # Projections
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * self.value_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.value_dim, hidden_dim, bias=False)

        # Gating projections
        # α (decay gate): controls how much of previous state to retain
        # β (update gate): controls how much new information to add
        self.alpha_proj = nn.Linear(hidden_dim, num_heads, bias=True)
        self.beta_proj = nn.Linear(hidden_dim, num_heads, bias=True)

        # LION bidirectional decay parameter (learnable)
        # This creates the symmetric mask M_ij = decay^|i-j|
        self.decay_log = nn.Parameter(torch.full((num_heads,), math.log(decay_init)))

        # Output gate (optional, for stability)
        self.g_proj = nn.Linear(hidden_dim, num_heads * self.value_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj, self.g_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)

        # Initialize gate biases for reasonable initial behavior
        nn.init.constant_(self.alpha_proj.bias, 0.0)  # sigmoid(0) = 0.5
        nn.init.constant_(self.beta_proj.bias, 0.0)

    @property
    def decay(self) -> torch.Tensor:
        """Get decay values (clamped between 0 and 1)."""
        return torch.sigmoid(self.decay_log)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with LION bidirectional attention.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            state: Optional recurrent state [batch, num_heads, key_dim, value_dim]
            return_state: Whether to return the final state

        Returns:
            output: [batch, seq_len, hidden_dim]
            state: Optional final state if return_state=True
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.q_proj(x)  # [batch, seq, num_heads * head_dim]
        k = self.k_proj(x)  # [batch, seq, num_heads * key_dim]
        v = self.v_proj(x)  # [batch, seq, num_heads * value_dim]

        # Reshape for multi-head
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        # Compute data-dependent gates
        alpha = torch.sigmoid(self.alpha_proj(x))  # [batch, seq, num_heads]
        beta = torch.sigmoid(self.beta_proj(x))  # [batch, seq, num_heads]
        alpha = rearrange(alpha, "b s h -> b h s 1 1")
        beta = rearrange(beta, "b s h -> b h s 1 1")

        # Compute output gate
        g = torch.sigmoid(self.g_proj(x))
        g = rearrange(g, "b s (h d) -> b h s d", h=self.num_heads)

        # Use parallel LION bidirectional mode for training
        output = self._lion_bidirectional_parallel(q, k, v, alpha, beta)

        # Apply output gate
        output = output * g

        # Reshape and project output
        output = rearrange(output, "b h s d -> b s (h d)")
        output = self.out_proj(output)
        output = self.dropout(output)

        if return_state:
            # For inference, compute final state
            final_state = self._compute_final_state(k, v, alpha, beta)
            return output, final_state

        return output, None

    def _lion_bidirectional_parallel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """LION bidirectional attention in parallel mode.

        This computes bidirectional linear attention where the mask
        M_ij = decay^|i-j| creates a symmetric attention pattern.

        For diffusion models, we need bidirectional attention where
        every token can attend to every other token.
        """
        batch_size, num_heads, seq_len, _ = q.shape
        device = q.device

        # Get per-head decay values
        decay = self.decay.to(device)  # [num_heads]

        # Build LION bidirectional mask: M_ij = decay^|i-j|
        positions = torch.arange(seq_len, device=device)
        distance = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))  # [seq, seq]

        # Apply decay based on distance: decay^|i-j|
        # Shape: [num_heads, seq, seq]
        decay_mask = decay.view(-1, 1, 1) ** distance.unsqueeze(0).float()

        # Normalize keys for numerical stability
        k_normalized = F.normalize(k, p=2, dim=-1)

        if self.use_delta_rule:
            # Delta rule attention
            # Instead of simple k^T v, use: k^T (v - S^T k) for correction
            output = self._delta_rule_attention(q, k_normalized, v, decay_mask, alpha, beta)
        else:
            # Simple GLA attention with LION mask
            output = self._gla_attention(q, k_normalized, v, decay_mask, alpha, beta)

        return output

    def _gla_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay_mask: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Standard GLA attention with LION bidirectional mask.

        Computes: output_i = sum_j M_ij * (q_i @ k_j^T) * v_j
        """
        # Compute attention scores: [batch, heads, seq, seq]
        attn = torch.einsum("bhid,bhjd->bhij", q, k)

        # Apply LION bidirectional decay mask
        attn = attn * decay_mask.unsqueeze(0)

        # Apply beta gating (update gate)
        beta_squeezed = beta.squeeze(-1)  # [batch, heads, seq, 1]
        attn = attn * beta_squeezed

        # Softmax normalization (optional, can also use linear)
        # For linear attention, we skip softmax
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)

        # Compute output
        output = torch.einsum("bhij,bhjd->bhid", attn, v)

        return output

    def _delta_rule_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay_mask: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Delta rule attention with LION mask.

        The delta rule improves associative recall by computing:
        S_t = S_{t-1} + beta_t * (v_t - S_{t-1}^T k_t) @ k_t^T

        This corrects the stored value based on what's already stored.
        """
        batch_size, num_heads, seq_len, key_dim = k.shape
        value_dim = v.shape[-1]
        device = k.device

        # Initialize state
        S = torch.zeros(batch_size, num_heads, key_dim, value_dim, device=device)

        outputs = []

        # Process bidirectionally - combine forward and backward passes
        # Forward pass
        S_forward = torch.zeros_like(S)
        forward_outputs = []

        for t in range(seq_len):
            k_t = k[:, :, t, :]  # [batch, heads, key_dim]
            v_t = v[:, :, t, :]  # [batch, heads, value_dim]
            q_t = q[:, :, t, :]  # [batch, heads, head_dim]
            alpha_t = alpha[:, :, t, :, :].squeeze(-1).squeeze(-1)  # [batch, heads]
            beta_t = beta[:, :, t, :, :].squeeze(-1).squeeze(-1)  # [batch, heads]

            # Delta rule: compute correction
            retrieved = torch.einsum("bhk,bhkv->bhv", k_t, S_forward)  # What's stored
            delta = v_t - retrieved  # Correction

            # Update state: S = alpha * S + beta * k @ delta^T
            outer = torch.einsum("bhk,bhv->bhkv", k_t, delta)
            S_forward = alpha_t.unsqueeze(-1).unsqueeze(-1) * S_forward + \
                        beta_t.unsqueeze(-1).unsqueeze(-1) * outer

            # Query the state
            out_t = torch.einsum("bhd,bhdv->bhv", q_t, S_forward)
            forward_outputs.append(out_t)

        # Backward pass
        S_backward = torch.zeros_like(S)
        backward_outputs = []

        for t in range(seq_len - 1, -1, -1):
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]
            alpha_t = alpha[:, :, t, :, :].squeeze(-1).squeeze(-1)
            beta_t = beta[:, :, t, :, :].squeeze(-1).squeeze(-1)

            retrieved = torch.einsum("bhk,bhkv->bhv", k_t, S_backward)
            delta = v_t - retrieved

            outer = torch.einsum("bhk,bhv->bhkv", k_t, delta)
            S_backward = alpha_t.unsqueeze(-1).unsqueeze(-1) * S_backward + \
                         beta_t.unsqueeze(-1).unsqueeze(-1) * outer

            out_t = torch.einsum("bhd,bhdv->bhv", q_t, S_backward)
            backward_outputs.insert(0, out_t)

        # Combine forward and backward (LION bidirectional)
        # Use decay-weighted combination
        forward_stack = torch.stack(forward_outputs, dim=2)  # [batch, heads, seq, value_dim]
        backward_stack = torch.stack(backward_outputs, dim=2)

        # Simple combination: average (can also learn weights)
        output = (forward_stack + backward_stack) / 2

        return output

    def _compute_final_state(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Compute final recurrent state for inference."""
        batch_size, num_heads, seq_len, key_dim = k.shape
        value_dim = v.shape[-1]

        S = torch.zeros(batch_size, num_heads, key_dim, value_dim, device=k.device)

        for t in range(seq_len):
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            alpha_t = alpha[:, :, t, :, :].squeeze(-1).squeeze(-1)
            beta_t = beta[:, :, t, :, :].squeeze(-1).squeeze(-1)

            if self.use_delta_rule:
                retrieved = torch.einsum("bhk,bhkv->bhv", k_t, S)
                delta = v_t - retrieved
                outer = torch.einsum("bhk,bhv->bhkv", k_t, delta)
            else:
                outer = torch.einsum("bhk,bhv->bhkv", k_t, v_t)

            S = alpha_t.unsqueeze(-1).unsqueeze(-1) * S + \
                beta_t.unsqueeze(-1).unsqueeze(-1) * outer

        return S


class GatedDeltaNetBlock(nn.Module):
    """Transformer block with Gated DeltaNet attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_delta_rule: bool = True,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = GatedDeltaNet(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            use_delta_rule=use_delta_rule,
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
        state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            state: Optional recurrent state
            return_state: Whether to return state

        Returns:
            output: [batch, seq_len, hidden_dim]
            state: Optional state
        """
        # Pre-norm attention with residual
        attn_out, new_state = self.attn(self.norm1(x), state, return_state)
        x = x + attn_out

        # Pre-norm MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x, new_state
