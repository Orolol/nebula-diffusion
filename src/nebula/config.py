"""Configuration dataclasses for Nebula Diffusion.

Hybrid architecture configuration:
- 75% Gated DeltaNet (LION) + 25% Full Attention with MLA
- Expert Choice MoE with nested experts
- Block Diffusion with DUS generation
- MTP head for auxiliary training
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    vocab_size: int = 50258  # GPT-2 (50257) + [MASK] token
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 2
    head_dim: int = 64
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    max_seq_len: int = 128
    mask_token_id: int = 50257

    # Hybrid attention configuration
    # Ratio of layers using Gated DeltaNet vs MLA (full attention)
    deltanet_ratio: float = 0.75  # 75% DeltaNet, 25% MLA

    # MLA (Multi-head Latent Attention) configuration
    kv_latent_dim: int = 32  # Low-rank KV compression dimension

    # MoE configuration
    use_moe: bool = True
    num_experts: int = 4
    num_expert_slices: int = 4  # Nested expert slices
    moe_capacity_factor: float = 1.25
    moe_every_n_layers: int = 2  # MoE on every Nth layer

    # Block diffusion configuration
    block_size: int = 8

    # MTP (Multi-Token Prediction) configuration
    use_mtp: bool = True
    mtp_num_tokens: int = 2
    mtp_loss_weight: float = 0.1

    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"

    def get_layer_types(self) -> List[str]:
        """Get attention type for each layer.

        Returns list of "deltanet" or "mla" for each layer.
        Distributes MLA layers evenly across the model.
        """
        num_mla = int(self.num_layers * (1 - self.deltanet_ratio))
        num_mla = max(1, num_mla) if self.num_layers > 1 else 0

        # Distribute MLA layers evenly
        layer_types = ["deltanet"] * self.num_layers

        if num_mla > 0:
            # Place MLA layers at regular intervals
            interval = self.num_layers / (num_mla + 1)
            for i in range(num_mla):
                mla_idx = int((i + 1) * interval)
                mla_idx = min(mla_idx, self.num_layers - 1)
                layer_types[mla_idx] = "mla"

        return layer_types

    def get_moe_layers(self) -> List[int]:
        """Get which layers should have MoE."""
        if not self.use_moe:
            return []
        return list(range(self.moe_every_n_layers - 1, self.num_layers, self.moe_every_n_layers))


@dataclass
class DiffusionConfig:
    """Diffusion process configuration."""

    num_diffusion_steps: int = 32  # Steps for generation
    min_masking_ratio: float = 0.0
    max_masking_ratio: float = 1.0

    # DUS (Dilated Unmasking Scheduler) configuration
    dus_num_groups: int = 3  # Dilation factor
    dus_confidence_threshold: float = 0.5
    dus_use_kl: bool = True
    dus_kl_threshold: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""

    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 10000
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"
    gradient_checkpointing: bool = False
    grad_clip: float = 1.0

    # torch.compile options (PyTorch 2.0+)
    compile: bool = False  # Enable torch.compile
    compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"

    # Logging
    log_every: int = 10
    save_every: int = 1000
    eval_every: int = 500

    # WandB
    wandb_project: str = "nebula-diffusion"
    wandb_run_name: Optional[str] = None

    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"


@dataclass
class DataConfig:
    """Data loading configuration."""

    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    max_seq_len: int = 128
    num_workers: int = 0  # 0 for streaming datasets
    prefetch_factor: int = 2


@dataclass
class Config:
    """Main configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu"

    def __post_init__(self):
        # Sync seq_len between model and data
        if self.data.max_seq_len != self.model.max_seq_len:
            self.data.max_seq_len = self.model.max_seq_len
