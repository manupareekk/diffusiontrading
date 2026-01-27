"""
Causal temporal convolution networks for diffusion models.

These networks are specifically designed to prevent lookahead bias
by using causal convolutions that only attend to past timesteps.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for diffusion timesteps.

    Maps discrete timesteps to continuous embeddings using
    sine and cosine functions at different frequencies.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: Tensor of shape (batch_size,) with timestep values

        Returns:
            Embeddings of shape (batch_size, dim)
        """
        device = timesteps.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution.

    Ensures that the output at time t only depends on inputs
    at times <= t, preventing lookahead bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Output tensor of shape (batch, out_channels, seq_len)
        """
        out = self.conv(x)
        # Remove the future-looking padding
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class DilatedCausalConvBlock(nn.Module):
    """
    Dilated causal convolution block with gated activation.

    Uses gated activation (tanh * sigmoid) similar to WaveNet.
    Includes residual connection and layer normalization.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.causal_conv = CausalConv1d(
            channels,
            channels * 2,  # Double for gating
            kernel_size,
            dilation=dilation,
        )
        self.layer_norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

        # 1x1 convolution for residual
        self.residual_conv = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, seq_len, channels)

        Returns:
            Output of shape (batch, seq_len, channels)
        """
        residual = x

        # (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)

        # Causal convolution with gating
        x = self.causal_conv(x)
        gate, filter_ = x.chunk(2, dim=1)
        x = torch.tanh(filter_) * torch.sigmoid(gate)

        # Residual connection
        x = self.residual_conv(x)

        # (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)

        # Layer norm and residual
        x = self.layer_norm(x + residual)
        x = self.dropout(x)

        return x


class TemporalConvStack(nn.Module):
    """
    Stack of dilated causal convolution blocks.

    Dilation increases exponentially (1, 2, 4, 8, ...) to capture
    long-range dependencies while maintaining causality.
    """

    def __init__(
        self,
        channels: int,
        num_layers: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            DilatedCausalConvBlock(
                channels,
                kernel_size,
                dilation=2 ** (i % 8),  # Reset dilation every 8 layers
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, seq_len, channels)

        Returns:
            Output of shape (batch, seq_len, channels)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class TemporalConvDenoiser(nn.Module):
    """
    Denoising network using causal temporal convolutions.

    Architecture:
    1. Input projection (features -> hidden_dim)
    2. Time embedding (sinusoidal + MLP)
    3. Optional condition embedding
    4. Stack of dilated causal conv blocks
    5. Self-attention layer
    6. Output projection (hidden_dim -> features)

    The causal nature ensures no lookahead bias during training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.1,
        time_embedding_dim: int = 64,
        condition_dim: Optional[int] = None,
        num_attention_heads: int = 4,
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension
            num_layers: Number of conv blocks
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
            time_embedding_dim: Dimension for time embeddings
            condition_dim: Dimension of conditioning input (None if no conditioning)
            num_attention_heads: Number of attention heads
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embedding_dim),
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Condition projection (optional)
        self.condition_proj = None
        if condition_dim is not None:
            self.condition_proj = nn.Sequential(
                nn.Linear(condition_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # Temporal convolution stack
        self.conv_stack = TemporalConvStack(
            hidden_dim,
            num_layers,
            kernel_size,
            dropout,
        )

        # Self-attention for capturing global patterns
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict noise given noisy input and timestep.

        Args:
            x: Noisy input of shape (batch, seq_len, input_dim)
            timestep: Diffusion timesteps of shape (batch,)
            condition: Optional conditioning of shape (batch, seq_len, condition_dim)

        Returns:
            Predicted noise of shape (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Time embedding
        t_emb = self.time_mlp(timestep)  # (batch, hidden_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, hidden_dim)

        # Input projection
        h = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # Add time embedding
        h = h + t_emb

        # Add condition embedding if present
        if condition is not None and self.condition_proj is not None:
            c_emb = self.condition_proj(condition)
            
            # Handle sequence length mismatch between input and condition
            # If condition sequence length != input sequence length, perform global pooling on condition
            if c_emb.shape[1] != h.shape[1]:
                # Global Average Pooling: (B, T_cond, H) -> (B, H)
                c_emb = c_emb.mean(dim=1)
                # Reshape for broadcasting: (B, H) -> (B, 1, H)
                c_emb = c_emb.unsqueeze(1)
                
            h = h + c_emb

        # Temporal convolutions
        h = self.conv_stack(h)

        # Causal self-attention (with mask to prevent attending to future)
        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
            diagonal=1
        )
        attn_out, _ = self.attention(h, h, h, attn_mask=attn_mask)
        h = self.attention_norm(h + attn_out)

        # Output projection
        output = self.output_proj(h)

        return output


class SimpleMLPDenoiser(nn.Module):
    """
    Simple MLP-based denoiser for baseline comparison.

    Flattens the sequence and uses fully connected layers.
    Less sophisticated but useful for debugging and ablation studies.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_embedding_dim: int = 64,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        flat_dim = input_dim * seq_len

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embedding_dim),
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.GELU(),
        )

        # MLP layers
        layers = [nn.Linear(flat_dim + hidden_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, flat_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, seq_len, input_dim)
            timestep: Timesteps of shape (batch,)
            condition: Ignored in this simple version

        Returns:
            Output of shape (batch, seq_len, input_dim)
        """
        batch_size = x.shape[0]

        # Flatten input
        x_flat = x.view(batch_size, -1)

        # Time embedding
        t_emb = self.time_mlp(timestep)

        # Concatenate and process
        h = torch.cat([x_flat, t_emb], dim=-1)
        h = self.mlp(h)

        # Reshape to original
        return h.view(batch_size, self.seq_len, self.input_dim)
