#!/usr/bin/env python3
"""Generation script for Nebula Diffusion.

Uses block-by-block generation with Dilated Unmasking Scheduler (DUS).
"""

import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nebula.config import Config, ModelConfig
from nebula.data import DiffusionTokenizer
from nebula.model import HybridDiffusionTransformer
from nebula.generation import BlockDiffusionSampler
from nebula.training.utils import get_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Generate text with Nebula Diffusion")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional prompt to condition generation",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=16,
        help="Number of blocks to generate",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--steps-per-block",
        type=int,
        default=16,
        help="Diffusion steps per block",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling (0 to disable)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling threshold (1.0 to disable)",
    )
    parser.add_argument(
        "--use-dus",
        action="store_true",
        default=True,
        help="Use Dilated Unmasking Scheduler",
    )
    parser.add_argument(
        "--no-dus",
        action="store_true",
        help="Disable Dilated Unmasking Scheduler",
    )
    parser.add_argument(
        "--dus-groups",
        type=int,
        default=3,
        help="Number of DUS dilated groups",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Handle DUS flag
    use_dus = args.use_dus and not args.no_dus

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Get config from checkpoint
    config = checkpoint.get("config")
    if config is None:
        print("Warning: Config not found in checkpoint, using defaults")
        config = Config()

    # Create tokenizer
    tokenizer = DiffusionTokenizer()

    # Create model
    model = HybridDiffusionTransformer(config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded (step {checkpoint['step']})")

    # Print architecture info
    layer_types = config.model.get_layer_types()
    moe_layers = config.model.get_moe_layers()
    print(f"  DeltaNet/MLA: {layer_types.count('deltanet')}/{layer_types.count('mla')}")
    print(f"  MoE layers: {len(moe_layers)}")
    print(f"  Block size: {config.model.block_size}")

    # Create sampler
    sampler = BlockDiffusionSampler(
        model=model,
        mask_token_id=config.model.mask_token_id,
        block_size=config.model.block_size,
        num_steps_per_block=args.steps_per_block,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_dus=use_dus,
        dus_num_groups=args.dus_groups,
    )

    # Prepare prompt if provided
    prompt_ids = None
    if args.prompt:
        prompt_tokens = tokenizer.encode(args.prompt)
        prompt_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        print(f"Prompt: {args.prompt}")
        print(f"Prompt tokens: {len(prompt_tokens)}")

    # Calculate total sequence length
    total_len = args.num_blocks * config.model.block_size

    # Generate
    print(f"\nGenerating {args.num_samples} sample(s)...")
    print(f"  Blocks: {args.num_blocks} x {config.model.block_size} = {total_len} tokens")
    print(f"  Steps per block: {args.steps_per_block}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Top-p: {args.top_p}")
    print(f"  DUS enabled: {use_dus}")
    if use_dus:
        print(f"  DUS groups: {args.dus_groups}")
    print("-" * 50)

    for i in range(args.num_samples):
        generated_ids = sampler.generate(
            num_blocks=args.num_blocks,
            batch_size=1,
            prompt_ids=prompt_ids,
            show_progress=True,
        )

        # Decode
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        print(f"\n=== Sample {i + 1} ===")
        print(text)
        print("=" * 50)


if __name__ == "__main__":
    main()
