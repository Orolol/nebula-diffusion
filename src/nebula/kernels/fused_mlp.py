"""Fused GELU activation with Triton kernel and PyTorch fallback."""

import math

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
    def _gelu_kernel(
        x_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for GELU activation.

        GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)

        # GELU approximation (tanh version)
        # sqrt(2/pi) ≈ 0.7978845608
        x_cubed = x * x * x
        inner = 0.7978845608 * (x + 0.044715 * x_cubed)
        tanh_inner = tl.math.tanh(inner)
        out = x * 0.5 * (1.0 + tanh_inner)

        tl.store(out_ptr + offsets, out, mask=mask)

    def triton_gelu(x: torch.Tensor) -> torch.Tensor:
        """Apply GELU using Triton kernel."""
        assert x.is_cuda, "Triton GELU requires CUDA tensor"

        out = torch.empty_like(x)
        n_elements = x.numel()

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        _gelu_kernel[grid](
            x.view(-1),
            out.view(-1),
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return out


def pytorch_gelu(x: torch.Tensor) -> torch.Tensor:
    """PyTorch fallback for GELU activation."""
    return torch.nn.functional.gelu(x)


def fused_gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU activation with automatic backend selection.

    Uses Triton kernel if CUDA is available, otherwise falls back to PyTorch.
    """
    if TRITON_AVAILABLE and x.is_cuda:
        return triton_gelu(x)
    return pytorch_gelu(x)


class FusedGELU(nn.Module):
    """GELU activation module with Triton/PyTorch backend."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_gelu(x)
