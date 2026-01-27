#!/usr/bin/env python3
"""
Train a diffusion model for financial time series prediction.

Usage:
    python scripts/train_diffusion.py --data artifacts/data/raw/SPY_5m.parquet
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from loguru import logger

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config.settings import settings


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--window-size", type=int, default=64, help="Context window size")
    parser.add_argument("--prediction-horizon", type=int, default=5, help="Prediction horizon")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.data}...")
    data = pd.read_parquet(args.data)
    logger.info(f"Loaded {len(data)} bars")

    # Import after path setup
    from src.data.dataset import FinancialTimeSeriesDataset, DatasetConfig
    from src.data.splits import TimeSeriesSplit
    from src.models.diffusion.ddpm import DDPM
    from src.models.networks.temporal_conv import TemporalConvDenoiser
    from src.models.schedulers.noise_schedule import create_noise_schedule

    # Create dataset config
    config = DatasetConfig(
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon,
        features_to_predict=[3],  # Close price index
        normalize=True,
    )

    # Split data
    splitter = TimeSeriesSplit(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        purge_window=10,
        embargo_window=5,
    )
    train_idx, val_idx, test_idx = splitter.split(len(data))

    # Use only OHLCV columns
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    data_values = data[ohlcv_cols].values

    train_data = data_values[train_idx]
    val_data = data_values[val_idx]

    logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

    # Create datasets
    train_dataset = FinancialTimeSeriesDataset(
        data=train_data,
        config=config,
        is_train=True,
    )

    val_dataset = FinancialTimeSeriesDataset(
        data=val_data,
        config=config,
        normalizer=train_dataset.normalizer,
        is_train=False,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Create model
    denoiser = TemporalConvDenoiser(
        input_dim=len(ohlcv_cols),
        hidden_dim=128,
        num_layers=8,
        kernel_size=3,
        dropout=0.1,
        time_embedding_dim=64,
        condition_dim=None,
    )

    noise_schedule = create_noise_schedule(
        schedule_type="linear",
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
    )

    model = DDPM(
        denoiser=denoiser,
        noise_schedule=noise_schedule,
        num_timesteps=1000,
        prediction_type="epsilon",
        loss_type="mse",
    ).to(args.device)

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    # Training loop
    best_val_loss = float("inf")
    checkpoint_dir = Path(args.checkpoint_dir or settings.model.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            target = batch["target"].to(args.device)

            optimizer.zero_grad()
            loss = model(target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                target = batch["target"].to(args.device)
                loss = model(target)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step()

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "normalizer_state": train_dataset.get_normalizer_state(),
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")

    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")

    return 0


if __name__ == "__main__":
    exit(main())
