"""Triton kernels with PyTorch fallbacks."""

import torch

# Check if CUDA and Triton are available
TRITON_AVAILABLE = False
CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        import triton

        TRITON_AVAILABLE = True
    except ImportError:
        pass

from .fused_mlp import fused_gelu, FusedGELU
from .fused_layernorm import fused_layer_norm, FusedLayerNorm

__all__ = [
    "TRITON_AVAILABLE",
    "CUDA_AVAILABLE",
    "fused_gelu",
    "FusedGELU",
    "fused_layer_norm",
    "FusedLayerNorm",
]
