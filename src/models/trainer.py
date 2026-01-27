"""
Training utilities for diffusion models.

Provides a PyTorch Lightning-based trainer with proper
logging, checkpointing, and early stopping.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
except ImportError:
    pl = None

from loguru import logger
from ..utils.device import get_device


class DiffusionLightningModule(pl.LightningModule if pl else object):
    """
    PyTorch Lightning module for training diffusion models.

    Wraps a DDPM model with training/validation logic.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        ema_decay: float = 0.9999,
        use_ema: bool = True,
    ):
        """
        Initialize the Lightning module.

        Args:
            model: DDPM model
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            warmup_steps: Number of warmup steps for learning rate
            ema_decay: Decay rate for EMA
            use_ema: Whether to use EMA for evaluation
        """
        if pl is None:
            raise ImportError("pytorch_lightning is required for training")

        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.ema_decay = ema_decay
        self.use_ema = use_ema

        # EMA model
        if use_ema:
            self.ema_model = self._create_ema_model()

        self.save_hyperparameters(ignore=["model"])

    def _create_ema_model(self) -> nn.Module:
        """Create a copy of the model for EMA."""
        import copy
        ema = copy.deepcopy(self.model)
        for param in ema.parameters():
            param.requires_grad = False
        return ema

    def _update_ema(self):
        """Update EMA model parameters."""
        if not self.use_ema:
            return

        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """Forward pass returns the loss."""
        return self.model(x, condition)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        x = batch["target"]
        condition = batch.get("history", None)

        loss = self(x, condition)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Update EMA
        self._update_ema()

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Validation step using EMA model if available."""
        x = batch["target"]
        condition = batch.get("history", None)

        model = self.ema_model if self.use_ema else self.model
        loss = model(x, condition)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Linear warmup then cosine decay
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_save_checkpoint(self, checkpoint):
        """Save EMA model in checkpoint."""
        if self.use_ema:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        """Load EMA model from checkpoint."""
        if self.use_ema and "ema_state_dict" in checkpoint:
            self.ema_model.load_state_dict(checkpoint["ema_state_dict"])


def create_trainer(
    max_epochs: int = 100,
    checkpoint_dir: Path = None,
    early_stopping_patience: int = 10,
    accelerator: str = "auto",
    devices: int = 1,
    log_every_n_steps: int = 10,
    val_check_interval: float = 1.0,
    gradient_clip_val: float = 1.0,
) -> "pl.Trainer":
    """
    Create a PyTorch Lightning trainer with standard callbacks.

    Args:
        max_epochs: Maximum training epochs
        checkpoint_dir: Directory for checkpoints
        early_stopping_patience: Patience for early stopping
        accelerator: Accelerator type ("cpu", "gpu", "auto")
        devices: Number of devices
        log_every_n_steps: Logging frequency
        val_check_interval: Validation check interval
        gradient_clip_val: Gradient clipping value

    Returns:
        PyTorch Lightning Trainer
    """
    if pl is None:
        raise ImportError("pytorch_lightning is required")

    callbacks = []

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=early_stopping_patience,
        mode="min",
        verbose=True,
    )
    callbacks.append(early_stopping)

    # Checkpointing
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="ddpm-{epoch:02d}-{val_loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=checkpoint_dir.parent if checkpoint_dir else "logs",
        name="diffusion_training",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        gradient_clip_val=gradient_clip_val,
        enable_progress_bar=True,
    )

    return trainer


def train_diffusion_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = 100,
    learning_rate: float = 1e-4,
    checkpoint_dir: Path = None,
    early_stopping_patience: int = 10,
    use_ema: bool = True,
) -> nn.Module:
    """
    Train a diffusion model.

    Convenience function that sets up the Lightning module and trainer.

    Args:
        model: DDPM model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        max_epochs: Maximum epochs
        learning_rate: Learning rate
        checkpoint_dir: Checkpoint directory
        early_stopping_patience: Early stopping patience
        use_ema: Whether to use EMA

    Returns:
        Trained model (or EMA model if use_ema=True)
    """
    if pl is None:
        raise ImportError("pytorch_lightning is required")

    # Create Lightning module
    lightning_module = DiffusionLightningModule(
        model=model,
        learning_rate=learning_rate,
        use_ema=use_ema,
    )

    # Create trainer
    trainer = create_trainer(
        max_epochs=max_epochs,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=early_stopping_patience,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(lightning_module, train_loader, val_loader)

    # Return EMA model if available
    if use_ema:
        return lightning_module.ema_model
    return lightning_module.model


class SimpleTrainer:
    """
    Simple training loop without PyTorch Lightning.

    Useful for debugging or when Lightning is not available.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        device: str = "auto",
    ):
        self.model = model
        self.device = get_device(device)
        self.model.to(self.device)

        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.optimizer = optimizer

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            x = batch["target"].to(self.device)
            condition = batch.get("history")
            if condition is not None:
                condition = condition.to(self.device)

            self.optimizer.zero_grad()
            loss = self.model(x, condition)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in val_loader:
            x = batch["target"].to(self.device)
            condition = batch.get("history")
            if condition is not None:
                condition = condition.to(self.device)

            loss = self.model(x, condition)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
    ) -> dict:
        """
        Full training loop.

        Returns:
            Dictionary with training history
        """
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(max_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            logger.info(f"Epoch {epoch + 1}/{max_epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        return history
