#!/usr/bin/env python3
"""Profile the model to find bottlenecks."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

from nebula.config import Config, ModelConfig, TrainingConfig, DiffusionConfig, DataConfig
from nebula.model import HybridDiffusionTransformer

def main():
    device = torch.device("cuda")

    # Load checkpoint to get config
    torch.serialization.add_safe_globals([Config, ModelConfig, TrainingConfig, DiffusionConfig, DataConfig])
    checkpoint = torch.load("checkpoints/base/checkpoint_10000.pt", map_location=device)
    config = checkpoint.get("config", Config())

    # Create model
    model = HybridDiffusionTransformer(config.model)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in checkpoint["model_state_dict"].items()})
    model = model.to(device)
    model.eval()

    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Config: hidden_dim={config.model.hidden_dim}, layers={config.model.num_layers}")
    print(f"MoE: {config.model.num_experts} experts")

    # Test sequence
    seq_len = 256
    batch_size = 1
    x = torch.randint(0, config.model.vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    print("\nWarming up...")
    with torch.inference_mode():
        for _ in range(5):
            _ = model.generate_forward(x)
    torch.cuda.synchronize()

    # Time individual components
    print("\n=== Timing individual forward passes ===")
    import time

    with torch.inference_mode():
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = model.generate_forward(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"generate_forward: {elapsed/10*1000:.1f}ms per call")

    # Profile with PyTorch profiler
    print("\n=== PyTorch Profiler ===")
    with torch.inference_mode():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            for _ in range(3):
                _ = model.generate_forward(x)
                torch.cuda.synchronize()

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Profile layer by layer
    print("\n=== Layer-by-layer timing ===")
    with torch.inference_mode():
        # Embedding
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            emb = model.token_embed(x)
        torch.cuda.synchronize()
        print(f"Embedding: {(time.time()-start)/10*1000:.2f}ms")

        # Positions
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Each layer
        h = emb
        for i, layer in enumerate(model.layers):
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                h_out, _ = layer(h, None, positions)
            torch.cuda.synchronize()
            layer_time = (time.time() - start) / 10 * 1000

            layer_type = "DeltaNet" if layer.attention_type == "deltanet" else "MLA"
            has_moe = "MoE" if layer.use_moe else "FFN"
            print(f"Layer {i:2d} ({layer_type:8s} + {has_moe}): {layer_time:.2f}ms")
            h = h_out

        # Final norm + LM head
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            out = model.lm_head(model.final_norm(h))
        torch.cuda.synchronize()
        print(f"Final norm + LM head: {(time.time()-start)/10*1000:.2f}ms")


if __name__ == "__main__":
    main()
