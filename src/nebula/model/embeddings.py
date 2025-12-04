"""Token and positional embeddings."""

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token embedding layer."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.hidden_dim = hidden_dim

        # Initialize with small values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]

        Returns:
            embeddings: [batch_size, seq_len, hidden_dim]
        """
        input_ids = input_ids.clone()
        return self.embedding(input_ids)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encodings.

    Following "Attention Is All You Need" (Vaswani et al., 2017).
    """

    def __init__(self, hidden_dim: int, max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but should be saved/loaded)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_seq_len, hidden_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]

        Returns:
            Tensor with positional encoding added: [batch_size, seq_len, hidden_dim]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encodings."""

    def __init__(self, hidden_dim: int, max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]

        Returns:
            Tensor with positional encoding added: [batch_size, seq_len, hidden_dim]
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.embedding(positions)
        return self.dropout(x)
