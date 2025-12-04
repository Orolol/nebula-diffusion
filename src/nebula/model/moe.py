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
from einops import rearrange


class NestedExpert(nn.Module):
    """Single nested expert with multiple compute slices.

    Following MoNE, the expert parameters are organized along an
    increasing compute-accuracy curve. Easy tokens use smaller slices,
    hard tokens use larger slices.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_slices: int = 4,
        dropout: float = 0.0,
    ):
        """
        Args:
            hidden_dim: Input/output dimension
            intermediate_dim: Maximum intermediate dimension
            num_slices: Number of nested slices (compute levels)
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_slices = num_slices

        # Compute slice boundaries
        # Slice i uses intermediate_dim * (i+1) / num_slices
        self.slice_sizes = [
            int(intermediate_dim * (i + 1) / num_slices)
            for i in range(num_slices)
        ]

        # Full-sized projections (we'll slice into them)
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        x: torch.Tensor,
        slice_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass using specified slice.

        Args:
            x: Input tensor [batch, hidden_dim]
            slice_idx: Which slice to use (0=smallest, num_slices-1=largest)
                      If None, uses full capacity

        Returns:
            Output tensor [batch, hidden_dim]
        """
        if slice_idx is None:
            slice_size = self.intermediate_dim
        else:
            slice_idx = min(slice_idx, self.num_slices - 1)
            slice_size = self.slice_sizes[slice_idx]

        # Use sliced weights for efficiency
        h = F.linear(x, self.fc1.weight[:slice_size], self.fc1.bias[:slice_size])
        h = F.gelu(h)
        h = self.dropout(h)
        h = F.linear(h, self.fc2.weight[:, :slice_size], self.fc2.bias)
        h = self.dropout(h)

        return h

    def get_flops(self, slice_idx: int) -> int:
        """Get FLOPs for a given slice."""
        slice_size = self.slice_sizes[min(slice_idx, self.num_slices - 1)]
        # fc1: hidden_dim * slice_size
        # fc2: slice_size * hidden_dim
        return 2 * self.hidden_dim * slice_size


class ExpertChoiceMoE(nn.Module):
    """Expert Choice MoE with nested experts and ReLU routing.

    Key features:
    - Expert Choice: experts select tokens (perfect load balancing)
    - Nested Experts: variable compute per token
    - ReLU Routing: fully differentiable (no TopK discontinuity)
    - Spatial Preferences: experts develop positional specialization
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int = 8,
        num_slices: int = 4,
        capacity_factor: float = 1.25,
        use_relu_routing: bool = True,
        enable_spatial_bias: bool = True,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        """
        Args:
            hidden_dim: Input/output dimension
            intermediate_dim: Expert intermediate dimension
            num_experts: Number of experts
            num_slices: Number of nested slices per expert
            capacity_factor: Expert capacity as factor of uniform distribution
            use_relu_routing: Use ReLU (ReMoE) vs softmax routing
            enable_spatial_bias: Learn positional biases for experts
            max_seq_len: Maximum sequence length for spatial bias
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_slices = num_slices
        self.capacity_factor = capacity_factor
        self.use_relu_routing = use_relu_routing
        self.enable_spatial_bias = enable_spatial_bias

        # Create nested experts
        self.experts = nn.ModuleList([
            NestedExpert(hidden_dim, intermediate_dim, num_slices, dropout)
            for _ in range(num_experts)
        ])

        # Router projection
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

        # Spatial positional bias (learnable)
        # Each expert learns which positions it prefers
        if enable_spatial_bias:
            self.spatial_bias = nn.Parameter(
                torch.zeros(num_experts, max_seq_len)
            )
        else:
            self.spatial_bias = None

        # Slice selector (predicts which slice to use based on difficulty)
        self.slice_predictor = nn.Linear(hidden_dim, num_slices)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.slice_predictor.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.slice_predictor.bias)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with Expert Choice routing.

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
            # Create position indices if not provided
            if positions is None:
                positions = torch.arange(seq_len, device=x.device)
                positions = positions.unsqueeze(0).expand(batch_size, -1)

            # Flatten positions
            pos_flat = positions.reshape(-1)  # [num_tokens]
            pos_flat = pos_flat.clamp(0, self.spatial_bias.shape[1] - 1)

            # Add spatial bias: [num_tokens, num_experts]
            spatial_bias = self.spatial_bias[:, pos_flat].T
            router_logits = router_logits + spatial_bias

        # Predict slice difficulty
        slice_logits = self.slice_predictor(x_flat)  # [num_tokens, num_slices]
        slice_probs = F.softmax(slice_logits, dim=-1)
        slice_indices = slice_probs.argmax(dim=-1)  # [num_tokens]

        # Expert Choice routing: softmax over TOKENS for each expert
        # This means each expert selects which tokens to process
        if self.use_relu_routing:
            # ReMoE: ReLU activation for fully differentiable routing
            router_weights = F.relu(router_logits)
            # Normalize per expert
            router_weights = router_weights / (router_weights.sum(dim=0, keepdim=True) + 1e-6)
        else:
            # Standard softmax over tokens (Expert Choice)
            router_weights = F.softmax(router_logits, dim=0)  # Softmax over tokens!

        # Capacity per expert
        capacity = int(num_tokens * self.capacity_factor / self.num_experts)
        top_k = min(capacity, num_tokens)

        # Batch topk for all experts at once: [num_experts, top_k]
        top_weights_all, top_indices_all = router_weights.T.topk(top_k, dim=-1)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # During training, use full capacity (no nested slices) for speed
        # Nested slices can be enabled for inference
        use_nested = not self.training and self.num_slices > 1

        if use_nested:
            # Slower path with nested slices for inference
            for expert_idx in range(self.num_experts):
                top_indices = top_indices_all[expert_idx]
                top_weights = top_weights_all[expert_idx]

                expert_input = x_flat[top_indices]
                expert_slice_indices = slice_indices[top_indices]

                expert_output = torch.zeros_like(expert_input)

                for slice_idx in range(self.num_slices):
                    slice_mask = expert_slice_indices == slice_idx
                    if slice_mask.any():
                        slice_input = expert_input[slice_mask]
                        slice_output = self.experts[expert_idx](slice_input, slice_idx)
                        expert_output[slice_mask] = slice_output.to(expert_output.dtype)

                weighted_output = expert_output * top_weights.unsqueeze(-1)
                output.index_add_(0, top_indices, weighted_output)
        else:
            # Fast path: process all experts with full capacity
            # Gather inputs for all experts: [num_experts, top_k, hidden_dim]
            expert_inputs = x_flat[top_indices_all]

            # Process each expert (can't fully vectorize due to different weights)
            for expert_idx in range(self.num_experts):
                expert_input = expert_inputs[expert_idx]  # [top_k, hidden_dim]
                expert_output = self.experts[expert_idx](expert_input, slice_idx=None)  # Full capacity

                # Weight and scatter
                weighted_output = expert_output * top_weights_all[expert_idx].unsqueeze(-1)
                output.index_add_(0, top_indices_all[expert_idx], weighted_output.to(output.dtype))

        # Reshape back
        output = rearrange(output, "(b s) d -> b s d", b=batch_size, s=seq_len)

        # Auxiliary info for logging
        aux_info = {
            "expert_counts": torch.tensor([top_k] * self.num_experts, device=x.device),
            "router_entropy": -(router_weights * (router_weights + 1e-8).log()).sum(),
            "slice_distribution": slice_probs.mean(dim=0),
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
        attention_type: str = "deltanet",  # "deltanet" or "mla"
        kv_latent_dim: int = 64,
        dropout: float = 0.0,
    ):
        """
        Args:
            hidden_dim: Model dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head
            num_experts: Number of experts
            intermediate_dim: Expert FFN dimension (default: 4 * hidden_dim)
            num_slices: Number of nested slices
            capacity_factor: Expert capacity factor
            attention_type: Type of attention to use
            kv_latent_dim: Latent dim for MLA attention
            dropout: Dropout probability
        """
        super().__init__()

        if intermediate_dim is None:
            intermediate_dim = 4 * hidden_dim

        # Attention layer
        self.norm1 = nn.LayerNorm(hidden_dim)

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
        self.norm2 = nn.LayerNorm(hidden_dim)
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
        """
        Args:
            x: Input [batch, seq_len, hidden_dim]
            attention_mask: Optional mask
            positions: Optional position indices

        Returns:
            output: [batch, seq_len, hidden_dim]
            aux_info: MoE routing statistics
        """
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
