"""
Conditioning modules for financial diffusion models.

Provides various ways to condition the diffusion model on:
- Historical price/return sequences
- Technical indicator values
- Realized volatility
- Market regime indicators
"""

from typing import Optional

import torch
import torch.nn as nn


class HistoryEncoder(nn.Module):
    """
    Encode historical price/return sequences.

    Uses a bidirectional LSTM to capture temporal patterns
    in the conditioning history.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Number of features per timestep
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Project bidirectional output to hidden_dim
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode historical sequence.

        Args:
            x: History tensor, shape (batch, seq_len, input_dim)

        Returns:
            Encoded history, shape (batch, hidden_dim)
        """
        # LSTM encoding
        output, (h_n, _) = self.lstm(x)

        # Use the concatenated final hidden states from both directions
        # h_n shape: (num_layers * 2, batch, hidden_dim)
        forward_h = h_n[-2]  # Last layer, forward
        backward_h = h_n[-1]  # Last layer, backward
        combined = torch.cat([forward_h, backward_h], dim=-1)

        # Project to hidden_dim
        return self.proj(combined)


class SignalEncoder(nn.Module):
    """
    Encode technical indicator signals.

    Simple MLP to transform signal features into embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Number of signal features
            hidden_dim: Output embedding dimension
            num_layers: Number of MLP layers
            dropout: Dropout probability
        """
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode signals.

        Args:
            x: Signal tensor, shape (batch, input_dim) or (batch, seq_len, input_dim)

        Returns:
            Encoded signals, shape (batch, hidden_dim)
        """
        # If sequence input, use the last timestep
        if x.dim() == 3:
            x = x[:, -1, :]

        return self.mlp(x)


class VolatilityEncoder(nn.Module):
    """
    Encode realized volatility features.

    Volatility is a key conditioning feature for financial prediction.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
    ):
        """
        Args:
            input_dim: Number of volatility features (e.g., 1 for single vol)
            hidden_dim: Output embedding dimension
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode volatility.

        Args:
            x: Volatility tensor, shape (batch, input_dim)

        Returns:
            Encoded volatility, shape (batch, hidden_dim)
        """
        return self.encoder(x)


class FinancialConditioner(nn.Module):
    """
    Combined conditioning module for financial diffusion models.

    Fuses multiple conditioning sources:
    - Historical price/return sequences
    - Technical indicator signals
    - Realized volatility
    - Optional additional features

    Supports different fusion strategies:
    - Concatenation + MLP
    - Additive combination
    - Cross-attention
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        history_dim: int = None,
        signal_dim: int = None,
        volatility_dim: int = None,
        use_history: bool = True,
        use_signals: bool = True,
        use_volatility: bool = True,
        fusion_type: str = "concat",  # "concat", "add", "attention"
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize the conditioner.

        Args:
            hidden_dim: Output conditioning dimension
            history_dim: Dimension of history features (per timestep)
            signal_dim: Dimension of signal features
            volatility_dim: Dimension of volatility features
            use_history: Whether to use history conditioning
            use_signals: Whether to use signal conditioning
            use_volatility: Whether to use volatility conditioning
            fusion_type: How to fuse different condition types
            num_lstm_layers: Number of LSTM layers for history encoder
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_history = use_history
        self.use_signals = use_signals
        self.use_volatility = use_volatility
        self.fusion_type = fusion_type

        # Build encoders for each conditioning type
        self.encoders = nn.ModuleDict()
        num_sources = 0

        if use_history and history_dim is not None:
            self.encoders["history"] = HistoryEncoder(
                input_dim=history_dim,
                hidden_dim=hidden_dim,
                num_layers=num_lstm_layers,
                dropout=dropout,
            )
            num_sources += 1

        if use_signals and signal_dim is not None:
            self.encoders["signals"] = SignalEncoder(
                input_dim=signal_dim,
                hidden_dim=hidden_dim,
                num_layers=2,
                dropout=dropout,
            )
            num_sources += 1

        if use_volatility and volatility_dim is not None:
            self.encoders["volatility"] = VolatilityEncoder(
                input_dim=volatility_dim,
                hidden_dim=hidden_dim,
            )
            num_sources += 1

        self.num_sources = num_sources

        # Build fusion layer
        if fusion_type == "concat" and num_sources > 1:
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * num_sources, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif fusion_type == "attention" and num_sources > 1:
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True,
            )
            self.fusion_norm = nn.LayerNorm(hidden_dim)
            self.fusion = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.fusion = nn.Identity()

    def forward(
        self,
        history: Optional[torch.Tensor] = None,
        signals: Optional[torch.Tensor] = None,
        volatility: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode and fuse all condition types.

        Args:
            history: Historical data, shape (batch, seq_len, history_dim)
            signals: Technical signals, shape (batch, signal_dim)
            volatility: Realized volatility, shape (batch, volatility_dim)

        Returns:
            Fused conditioning vector, shape (batch, hidden_dim)
        """
        embeddings = []

        if self.use_history and history is not None and "history" in self.encoders:
            h_emb = self.encoders["history"](history)
            embeddings.append(h_emb)

        if self.use_signals and signals is not None and "signals" in self.encoders:
            s_emb = self.encoders["signals"](signals)
            embeddings.append(s_emb)

        if self.use_volatility and volatility is not None and "volatility" in self.encoders:
            v_emb = self.encoders["volatility"](volatility)
            embeddings.append(v_emb)

        if len(embeddings) == 0:
            raise ValueError("At least one conditioning input must be provided")

        if len(embeddings) == 1:
            return embeddings[0]

        # Fuse embeddings
        if self.fusion_type == "concat":
            combined = torch.cat(embeddings, dim=-1)
            return self.fusion(combined)

        elif self.fusion_type == "add":
            return sum(embeddings) / len(embeddings)

        elif self.fusion_type == "attention":
            # Stack embeddings as sequence for attention
            stacked = torch.stack(embeddings, dim=1)  # (batch, num_sources, hidden)
            attended, _ = self.fusion_attention(stacked, stacked, stacked)
            attended = self.fusion_norm(attended + stacked)
            # Pool by averaging
            pooled = attended.mean(dim=1)
            return self.fusion(pooled)

        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")


class FiLMConditioning(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) conditioning.

    Applies affine transformation to features based on conditioning:
        output = gamma * features + beta

    Useful for conditioning intermediate layers of the denoiser.
    """

    def __init__(
        self,
        condition_dim: int,
        feature_dim: int,
    ):
        """
        Args:
            condition_dim: Dimension of conditioning vector
            feature_dim: Dimension of features to modulate
        """
        super().__init__()

        self.gamma_proj = nn.Linear(condition_dim, feature_dim)
        self.beta_proj = nn.Linear(condition_dim, feature_dim)

        # Initialize gamma close to 1 and beta close to 0
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply FiLM conditioning.

        Args:
            features: Features to modulate, shape (batch, ..., feature_dim)
            condition: Conditioning vector, shape (batch, condition_dim)

        Returns:
            Modulated features, same shape as input
        """
        gamma = self.gamma_proj(condition)
        beta = self.beta_proj(condition)

        # Expand gamma/beta for broadcasting
        while gamma.dim() < features.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return gamma * features + beta
