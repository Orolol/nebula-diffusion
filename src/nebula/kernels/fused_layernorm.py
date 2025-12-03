"""Fused LayerNorm with Triton kernel and PyTorch fallback."""

import torch
import torch.nn as nn

# Try to import Triton
TRITON_AVAILABLE = False
if torch.cuda.is_available():
    try:
        import triton
        import triton.language as tl

        TRITON_AVAILABLE = True
    except ImportError:
        pass


if TRITON_AVAILABLE:

    @triton.jit
    def _layer_norm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        stride,
        hidden_dim,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for LayerNorm.

        Each program handles one row (one token's hidden dimension).
        """
        row_idx = tl.program_id(0)
        row_start = row_idx * stride

        # Load the row
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim

        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)

        # Compute mean
        mean = tl.sum(x, axis=0) / hidden_dim

        # Compute variance
        x_centered = x - mean
        var = tl.sum(x_centered * x_centered, axis=0) / hidden_dim

        # Normalize
        inv_std = 1.0 / tl.sqrt(var + eps)
        x_norm = x_centered * inv_std

        # Load weight and bias
        weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)

        # Apply affine transformation
        out = x_norm * weight + bias

        # Store result
        tl.store(out_ptr + row_start + offsets, out, mask=mask)

    def triton_layer_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Apply LayerNorm using Triton kernel."""
        assert x.is_cuda, "Triton LayerNorm requires CUDA tensor"

        # Flatten to 2D: [num_rows, hidden_dim]
        orig_shape = x.shape
        hidden_dim = x.shape[-1]
        x_2d = x.view(-1, hidden_dim)
        num_rows = x_2d.shape[0]

        out = torch.empty_like(x_2d)

        # Determine block size (must be power of 2 and >= hidden_dim)
        BLOCK_SIZE = triton.next_power_of_2(hidden_dim)

        grid = (num_rows,)

        _layer_norm_kernel[grid](
            x_2d,
            weight,
            bias,
            out,
            hidden_dim,  # stride
            hidden_dim,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return out.view(orig_shape)


def pytorch_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """PyTorch fallback for LayerNorm."""
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def fused_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """LayerNorm with automatic backend selection.

    Uses Triton kernel if CUDA is available, otherwise falls back to PyTorch.
    """
    if TRITON_AVAILABLE and x.is_cuda:
        return triton_layer_norm(x, weight, bias, eps)
    return pytorch_layer_norm(x, weight, bias, eps)


class FusedLayerNorm(nn.Module):
    """LayerNorm module with Triton/PyTorch backend."""

    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_layer_norm(x, self.weight, self.bias, self.eps)
