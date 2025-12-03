"""Hybrid Transformer for Nebula Diffusion.

Combines:
- 75% Gated DeltaNet (LION bidirectional) layers
- 25% Full Attention with MLA compression layers
- Expert Choice MoE with nested experts on select layers
- MTP head for multi-token prediction

This architecture follows the spec from aidocs/doc.md:
- Hybrid attention for O(1) memory complexity + strong recall
- Spatial MoE for heterogeneous parallel token prediction
- Block diffusion for KV cache compatibility
"""

from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from .embeddings import TokenEmbedding, SinusoidalPositionalEncoding
from .gated_deltanet import GatedDeltaNet
from .mla_attention import MultiHeadLatentAttention
from .moe import ExpertChoiceMoE
from .mtp_head import MultiTokenPredictionHead


class HybridTransformerBlock(nn.Module):
    """Transformer block with hybrid attention (DeltaNet or MLA) and optional MoE."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        attention_type: str = "deltanet",  # "deltanet" or "mla"
        kv_latent_dim: int = 64,
        mlp_ratio: float = 4.0,
        use_moe: bool = False,
        num_experts: int = 8,
        num_expert_slices: int = 4,
        moe_capacity_factor: float = 1.25,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.attention_type = attention_type
        self.use_moe = use_moe

        # Pre-norm for attention
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Attention layer (DeltaNet or MLA)
        if attention_type == "deltanet":
            self.attn = GatedDeltaNet(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
            )
        else:
            self.attn = MultiHeadLatentAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                kv_latent_dim=kv_latent_dim,
                dropout=dropout,
            )

        # Pre-norm for FFN
        self.norm2 = nn.LayerNorm(hidden_dim)

        # FFN: MoE or standard MLP
        if use_moe:
            intermediate_dim = int(hidden_dim * mlp_ratio)
            self.ffn = ExpertChoiceMoE(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_experts=num_experts,
                num_slices=num_expert_slices,
                capacity_factor=moe_capacity_factor,
                dropout=dropout,
            )
        else:
            intermediate_dim = int(hidden_dim * mlp_ratio)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, hidden_dim),
                nn.Dropout(dropout),
            )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask
            positions: Optional position indices for MoE

        Returns:
            output: [batch, seq_len, hidden_dim]
            aux_info: Dictionary with attention/MoE statistics
        """
        aux_info = {}

        # Attention
        if self.attention_type == "deltanet":
            attn_out, _ = self.attn(self.norm1(x))
        else:
            attn_out, _ = self.attn(self.norm1(x), attention_mask)

        x = x + attn_out

        # FFN
        normed = self.norm2(x)
        if self.use_moe:
            ffn_out, moe_info = self.ffn(normed, positions)
            aux_info["moe"] = moe_info
        else:
            ffn_out = self.ffn(normed)

        x = x + ffn_out

        return x, aux_info


class HybridDiffusionTransformer(nn.Module):
    """Main hybrid transformer for masked diffusion.

    Architecture:
    - Token embeddings + Sinusoidal positional encoding
    - N transformer blocks with hybrid attention:
      - 75% Gated DeltaNet (LION bidirectional) for O(1) memory
      - 25% Full Attention with MLA compression for strong recall
    - Expert Choice MoE on select layers
    - Final layer norm
    - LM head + MTP head
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embed = TokenEmbedding(config.vocab_size, config.hidden_dim)
        self.pos_embed = SinusoidalPositionalEncoding(
            config.hidden_dim,
            config.max_seq_len,
            dropout=config.dropout,
        )

        # Get layer configurations
        layer_types = config.get_layer_types()
        moe_layers = config.get_moe_layers()

        # Build transformer blocks
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_layers):
            block = HybridTransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                attention_type=layer_types[layer_idx],
                kv_latent_dim=config.kv_latent_dim,
                mlp_ratio=config.mlp_ratio,
                use_moe=layer_idx in moe_layers,
                num_experts=config.num_experts,
                num_expert_slices=config.num_expert_slices,
                moe_capacity_factor=config.moe_capacity_factor,
                dropout=config.dropout,
            )
            self.layers.append(block)

        # Output
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # MTP head (optional)
        self.mtp_head = None
        if config.use_mtp:
            self.mtp_head = MultiTokenPredictionHead(
                hidden_dim=config.hidden_dim,
                vocab_size=config.vocab_size,
                num_tokens=config.mtp_num_tokens,
            )

        # Initialize
        self._init_weights()

        # Gradient checkpointing flag
        self._gradient_checkpointing = False

    def _init_weights(self):
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: Token IDs [batch, seq_len], may contain [MASK] tokens
            attention_mask: Optional attention mask [batch, seq_len]
            target_ids: Optional targets for MTP loss [batch, seq_len]

        Returns:
            Dictionary with:
                - logits: [batch, seq_len, vocab_size]
                - hidden_states: [batch, seq_len, hidden_dim]
                - mtp_loss: Optional MTP auxiliary loss
                - aux_info: Layer statistics (MoE routing, etc.)
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens and add positional encoding
        x = self.token_embed(input_ids)
        x = self.pos_embed(x)

        # Create position indices for MoE spatial bias
        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        # Pass through transformer blocks
        aux_infos = []

        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x, aux = checkpoint(
                    layer, x, attention_mask, positions,
                    use_reentrant=False
                )
            else:
                x, aux = layer(x, attention_mask, positions)
            aux_infos.append(aux)

        # Final norm
        hidden_states = self.final_norm(x)

        # LM head
        logits = self.lm_head(hidden_states)

        result = {
            "logits": logits,
            "hidden_states": hidden_states,
            "aux_info": aux_infos,
        }

        # MTP auxiliary loss
        if self.mtp_head is not None and target_ids is not None:
            _, mtp_loss = self.mtp_head(hidden_states, target_ids)
            result["mtp_loss"] = mtp_loss * self.config.mtp_loss_weight

        return result

    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convenience method to get just logits."""
        return self.forward(input_ids)["logits"]

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_info(self) -> List[Dict[str, Any]]:
        """Get information about each layer's configuration."""
        layer_types = self.config.get_layer_types()
        moe_layers = self.config.get_moe_layers()

        info = []
        for i in range(self.config.num_layers):
            info.append({
                "layer": i,
                "attention_type": layer_types[i],
                "has_moe": i in moe_layers,
            })
        return info


# Backwards compatibility alias
DiffusionTransformer = HybridDiffusionTransformer
