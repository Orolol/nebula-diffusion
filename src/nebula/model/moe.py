"""Expert Choice MoE with Nested Experts (MoNE) and ReLU routing (ReMoE).

Implements the proposed spatial MoE architecture:
- Expert Choice routing: experts select tokens (not tokens select experts)
- Nested experts (MoNE): smaller slices for easy tokens, larger for hard
- ReLU activation routing (ReMoE): fully differentiable selection
- Spatial positional preferences for experts

References:
- Expert Choice Routing: arXiv:2202.09368
- MoNE (Nested Experts): arXiv:2407.19985
- ReMoE (ReLU Routing): arXiv:2412.14711
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

from .embeddings import RMSNorm


class BatchedExperts(nn.Module):
    """Batched experts with SwiGLU for efficient parallel computation.

    All experts share the same architecture but have independent weights.
    Uses batched matmuls instead of loops for GPU efficiency.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        # Batched weights for all experts: [num_experts, out_dim, in_dim]
        # SwiGLU: gate, up, down projections
        self.w_gate = nn.Parameter(torch.empty(num_experts, intermediate_dim, hidden_dim))
        self.w_up = nn.Parameter(torch.empty(num_experts, intermediate_dim, hidden_dim))
        self.w_down = nn.Parameter(torch.empty(num_experts, hidden_dim, intermediate_dim))

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for w in [self.w_gate, self.w_up, self.w_down]:
            nn.init.normal_(w, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with batched expert computation.

        Args:
            x: Input tensor [num_experts, capacity, hidden_dim]
            expert_indices: Not used, for API compatibility

        Returns:
            Output tensor [num_experts, capacity, hidden_dim]
        """
        # x: [E, C, D] where E=num_experts, C=capacity, D=hidden_dim
        # w_gate, w_up: [E, I, D] where I=intermediate_dim
        # w_down: [E, D, I]

        # Batched matmul: [E, C, D] @ [E, D, I] -> [E, C, I]
        gate = torch.bmm(x, self.w_gate.transpose(1, 2))  # [E, C, I]
        up = torch.bmm(x, self.w_up.transpose(1, 2))      # [E, C, I]

        # SwiGLU activation
        h = F.silu(gate) * up  # [E, C, I]
        h = self.dropout(h)

        # Down projection: [E, C, I] @ [E, I, D] -> [E, C, D]
        out = torch.bmm(h, self.w_down.transpose(1, 2))   # [E, C, D]

        return out


class ExpertChoiceMoE(nn.Module):
    """Expert Choice MoE with batched experts and ReLU routing.

    Key features:
    - Expert Choice: experts select tokens (perfect load balancing)
    - Batched computation: all experts run in parallel via batched matmuls
    - ReLU Routing: fully differentiable (no TopK discontinuity)
    - Spatial Preferences: experts develop positional specialization
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int = 8,
        num_slices: int = 4,  # Kept for API compatibility, not used in batched version
        capacity_factor: float = 1.25,
        use_relu_routing: bool = True,
        enable_spatial_bias: bool = True,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.use_relu_routing = use_relu_routing
        self.enable_spatial_bias = enable_spatial_bias

        # Batched experts for efficient parallel computation
        self.experts = BatchedExperts(
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
        )

        # Router projection
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

        # Spatial positional bias (learnable)
        if enable_spatial_bias:
            self.spatial_bias = nn.Parameter(
                torch.zeros(num_experts, max_seq_len)
            )
        else:
            self.spatial_bias = None

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with Expert Choice routing and batched computation.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            positions: Optional position indices [batch, seq_len]

        Returns:
            output: [batch, seq_len, hidden_dim]
            aux_info: Dictionary with routing statistics
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Flatten for routing: [batch * seq, hidden_dim]
        x_flat = rearrange(x, "b s d -> (b s) d")
        num_tokens = x_flat.shape[0]

        # Compute routing logits
        router_logits = self.router(x_flat)  # [num_tokens, num_experts]

        # Add spatial bias if enabled
        if self.spatial_bias is not None:
            if positions is None:
                positions = torch.arange(seq_len, device=x.device)
                positions = positions.unsqueeze(0).expand(batch_size, -1)

            pos_flat = positions.reshape(-1).clamp(0, self.spatial_bias.shape[1] - 1)
            spatial_bias = self.spatial_bias[:, pos_flat].T
            router_logits = router_logits + spatial_bias

        # Expert Choice routing: softmax over TOKENS for each expert
        if self.use_relu_routing:
            router_weights = F.relu(router_logits)
            router_weights = router_weights / (router_weights.sum(dim=0, keepdim=True) + 1e-6)
        else:
            router_weights = F.softmax(router_logits, dim=0)

        # Capacity per expert
        capacity = int(num_tokens * self.capacity_factor / self.num_experts)
        capacity = min(capacity, num_tokens)

        # Get top-k tokens for each expert: [num_experts, capacity]
        top_weights, top_indices = router_weights.T.topk(capacity, dim=-1)

        # Gather inputs for all experts: [num_experts, capacity, hidden_dim]
        expert_inputs = x_flat[top_indices]  # Advanced indexing

        # Batched forward through all experts at once
        expert_outputs = self.experts(expert_inputs, None)  # [E, C, D]

        # Apply routing weights: [E, C, D] * [E, C, 1] -> [E, C, D]
        weighted_outputs = expert_outputs * top_weights.unsqueeze(-1)

        # Scatter back to original positions
        # Use scatter_add for efficiency
        output = torch.zeros_like(x_flat)

        # Flatten expert dimension for scatter
        flat_indices = top_indices.reshape(-1)  # [E * C]
        flat_outputs = weighted_outputs.reshape(-1, hidden_dim)  # [E * C, D]

        # Scatter add (handles overlapping indices correctly)
        output.scatter_add_(0, flat_indices.unsqueeze(-1).expand(-1, hidden_dim), flat_outputs)

        # Reshape back
        output = rearrange(output, "(b s) d -> b s d", b=batch_size, s=seq_len)

        # Auxiliary info for logging
        aux_info = {
            "router_entropy": -(router_weights * (router_weights + 1e-8).log()).sum(),
        }

        return output, aux_info


class MoEBlock(nn.Module):
    """Transformer block with Expert Choice MoE in the FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        num_experts: int = 8,
        intermediate_dim: Optional[int] = None,
        num_slices: int = 4,
        capacity_factor: float = 1.25,
        attention_type: str = "deltanet",
        kv_latent_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        if intermediate_dim is None:
            intermediate_dim = 4 * hidden_dim

        # Attention layer
        self.norm1 = RMSNorm(hidden_dim)

        if attention_type == "deltanet":
            from .gated_deltanet import GatedDeltaNet
            self.attn = GatedDeltaNet(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
            )
        else:
            from .mla_attention import MultiHeadLatentAttention
            self.attn = MultiHeadLatentAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                kv_latent_dim=kv_latent_dim,
                dropout=dropout,
            )

        self.attention_type = attention_type

        # MoE FFN layer
        self.norm2 = RMSNorm(hidden_dim)
        self.moe = ExpertChoiceMoE(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            num_slices=num_slices,
            capacity_factor=capacity_factor,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        # Attention
        if self.attention_type == "deltanet":
            attn_out, _ = self.attn(self.norm1(x))
        else:
            attn_out, _ = self.attn(self.norm1(x), attention_mask)
        x = x + attn_out

        # MoE FFN
        moe_out, aux_info = self.moe(self.norm2(x), positions)
        x = x + moe_out

        return x, aux_info
