#!/usr/bin/env python3
"""Training script for Nebula Diffusion.

Trains the hybrid architecture:
- 75% Gated DeltaNet (LION) + 25% MLA
- Expert Choice MoE with nested experts
- Block Diffusion training
- MTP auxiliary loss
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nebula.config import Config, ModelConfig, TrainingConfig, DiffusionConfig, DataConfig
from nebula.data import DiffusionTokenizer, create_dataloader
from nebula.model import HybridDiffusionTransformer
from nebula.model.block_diffusion import BlockDiffusion
from nebula.training import Trainer, set_seed, get_device
from nebula.training.utils import format_number

# Enable TF32 for faster training on Ampere+ GPUs
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message="Online softmax is disabled.*")



def load_config(config_path: str | None) -> Config:
    """Load configuration from YAML file or use defaults."""
    if config_path and Path(config_path).exists():
        yaml_config = OmegaConf.load(config_path)

        # Build config from YAML
        model_cfg = ModelConfig(**yaml_config.get("model", {}))
        training_cfg = TrainingConfig(**yaml_config.get("training", {}))
        diffusion_cfg = DiffusionConfig(**yaml_config.get("diffusion", {}))
        data_cfg = DataConfig(**yaml_config.get("data", {}))

        config = Config(
            model=model_cfg,
            training=training_cfg,
            diffusion=diffusion_cfg,
            data=data_cfg,
            seed=yaml_config.get("seed", 42),
            device=yaml_config.get("device", "auto"),
        )
    else:
        # Use defaults (tiny config)
        config = Config()

    return config


def main():
    parser = argparse.ArgumentParser(description="Train Nebula Diffusion model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max training steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Override WandB project name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--no-moe",
        action="store_true",
        help="Disable MoE layers",
    )
    parser.add_argument(
        "--no-mtp",
        action="store_true",
        help="Disable MTP auxiliary loss",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (PyTorch 2.0+)",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply command line overrides
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.wandb_project:
        config.training.wandb_project = args.wandb_project
    if args.seed:
        config.seed = args.seed
    if args.no_moe:
        config.model.use_moe = False
    if args.no_mtp:
        config.model.use_mtp = False
    if args.compile:
        config.training.compile = True
    if args.compile_mode:
        config.training.compile_mode = args.compile_mode

    # Set seed
    set_seed(config.seed)

    # Get device
    device = get_device(config.device)
    print(f"Using device: {device}")

    # Create tokenizer
    print("Loading tokenizer...")
    tokenizer = DiffusionTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create dataloader
    print("Creating dataloader...")
    train_loader = create_dataloader(
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_seq_len=config.data.max_seq_len,
        dataset_name=config.data.dataset_name,
        dataset_config=config.data.dataset_config,
        num_workers=4,
        prefetch_factor=4,
    )

    # Create hybrid model
    print("Creating hybrid model...")
    model = HybridDiffusionTransformer(config.model)
    num_params = model.count_parameters()
    print(f"Model parameters: {format_number(num_params)}")

    # Print architecture info
    layer_types = config.model.get_layer_types()
    moe_layers = config.model.get_moe_layers()
    deltanet_count = layer_types.count("deltanet")
    mla_count = layer_types.count("mla")
    print(f"  DeltaNet layers: {deltanet_count} ({100*deltanet_count/len(layer_types):.0f}%)")
    print(f"  MLA layers: {mla_count} ({100*mla_count/len(layer_types):.0f}%)")
    print(f"  MoE layers: {len(moe_layers)} (indices: {moe_layers})")
    print(f"  Block size: {config.model.block_size}")
    print(f"  MTP enabled: {config.model.use_mtp}")

    # Apply torch.compile if enabled
    if config.training.compile:
        if hasattr(torch, "compile"):
            print(f"Compiling model with mode='{config.training.compile_mode}'...")
            model = torch.compile(model, mode=config.training.compile_mode)
            print("Model compiled successfully")
        else:
            print("Warning: torch.compile not available (requires PyTorch 2.0+)")

    # Create block diffusion process
    block_diffusion = BlockDiffusion(
        mask_token_id=config.model.mask_token_id,
        vocab_size=config.model.vocab_size,
        block_size=config.model.block_size,
    )

    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        block_diffusion=block_diffusion,
        train_loader=train_loader,
        config=config,
        device=device,
    )

    # Start training
    print("\n" + "=" * 50)
    print("Starting training...")
    print(f"  Max steps: {config.training.max_steps}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Mixed precision: {config.training.mixed_precision}")
    print(f"  torch.compile: {config.training.compile} ({config.training.compile_mode})")
    print("=" * 50 + "\n")

    trainer.train()


if __name__ == "__main__":
    main()
