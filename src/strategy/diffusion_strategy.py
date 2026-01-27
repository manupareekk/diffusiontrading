"""
Diffusion model-based trading strategy.

Uses the diffusion model to predict future return distributions
and generates trading signals based on expected returns and
prediction confidence.
"""

from typing import Optional

import numpy as np
import pandas as pd
import torch

from .base import BaseStrategy, Position, Signal, StrategyConfig
from ..utils.device import get_device
from ..data.features import ModelConditioner


class SignalBasedStrategy:
    """Simple RSI-based trading strategy for backtesting."""

    def __init__(
        self,
        rsi_period: int = 9,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
    ):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def compute_rsi(self, prices: pd.Series) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals: 1=buy, -1=sell, 0=hold."""
        rsi = self.compute_rsi(data['close'])

        signals = pd.Series(0, index=data.index)
        signals[rsi < self.rsi_oversold] = 1   # Buy signal
        signals[rsi > self.rsi_overbought] = -1  # Sell signal

        return signals


class DiffusionStrategyConfig(StrategyConfig):
    """Configuration specific to diffusion strategy."""
    num_samples: int = 100  # Monte Carlo samples
    prediction_horizon: int = 5
    prob_threshold: float = 0.6  # Min probability for trade
    var_percentile: float = 5  # VaR percentile


class DiffusionPredictionStrategy(BaseStrategy):
    """
    Trading strategy based on diffusion model predictions.

    Uses the diffusion model to:
    1. Generate distribution of future returns via Monte Carlo
    2. Estimate probability of positive/negative returns
    3. Size positions based on confidence and expected return
    4. Set stop-loss/take-profit based on prediction distribution
    """

    def __init__(
        self,
        model: torch.nn.Module,
        window_size: int = 64,
        normalizer=None,
        device: str = "auto",
        config: DiffusionStrategyConfig = None,
    ):
        """
        Initialize the diffusion strategy.

        Args:
            model: Trained DDPM model
            window_size: Historical window size for conditioning
            normalizer: Data normalizer (from training)
            device: Compute device
            config: Strategy configuration
        """
        super().__init__(config or DiffusionStrategyConfig())
        self.model = model
        self.window_size = window_size
        self.normalizer = normalizer
        self.normalizer = normalizer
        self.device = get_device(device)
        self.conditioner = ModelConditioner()

        self.model.eval()
        self.model.to(device)

    def generate_signals(
        self,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> list[Signal]:
        """
        Generate trading signals from market data.

        Args:
            data: DataFrame with OHLCV and features
            symbol: Asset symbol

        Returns:
            List of Signal objects
        """
        self.validate_data(data)

        signals = []
        current_time = data.index[-1]
        current_price = data["close"].iloc[-1]

        # Get prediction distribution
        prediction = self.predict_return_distribution(data)

        if prediction is None:
            return signals

        # Determine signal based on prediction
        prob_positive = prediction["prob_positive"]
        prob_negative = prediction["prob_negative"]
        expected_return = prediction["mean"]
        var_95 = prediction["var_95"]

        # Generate signal if probability exceeds threshold
        if prob_positive > self.config.prob_threshold and expected_return > self.config.min_expected_return:
            position = Position.LONG
            confidence = prob_positive
            allocation = self._calculate_allocation(expected_return, prediction["std"], confidence)

        elif prob_negative > self.config.prob_threshold and expected_return < -self.config.min_expected_return:
            position = Position.SHORT
            confidence = prob_negative
            allocation = -self._calculate_allocation(abs(expected_return), prediction["std"], confidence)

        else:
            # No clear signal
            return signals

        # Calculate stop-loss and take-profit from distribution
        stop_loss = self._calculate_stop_from_distribution(
            current_price, position, prediction
        )
        take_profit = self._calculate_tp_from_distribution(
            current_price, position, prediction
        )

        signal = Signal(
            timestamp=current_time,
            symbol=symbol,
            position=position,
            confidence=confidence,
            target_allocation=allocation,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "expected_return": expected_return,
                "return_std": prediction["std"],
                "prob_positive": prob_positive,
                "prob_negative": prob_negative,
                "var_95": var_95,
                "prediction_horizon": self.config.prediction_horizon,
            },
        )

        signals.append(signal)
        return signals

    def predict_return_distribution(
        self,
        data: pd.DataFrame,
    ) -> Optional[dict]:
        """
        Generate Monte Carlo samples of future returns.

        Args:
            data: Historical market data

        Returns:
            Dictionary with distribution statistics
        """
        # Prepare input
        history = self._prepare_input(data)
        if history is None:
            return None

        # Generate samples
        samples = self._generate_samples(history)
        if samples is None:
            return None

        # Compute statistics
        return {
            "samples": samples,
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "median": float(np.median(samples)),
            "prob_positive": float(np.mean(samples > 0)),
            "prob_negative": float(np.mean(samples < 0)),
            "var_95": float(np.percentile(samples, 5)),  # 5th percentile
            "var_99": float(np.percentile(samples, 1)),
            "percentile_25": float(np.percentile(samples, 25)),
            "percentile_75": float(np.percentile(samples, 75)),
        }

    def _prepare_input(self, data: pd.DataFrame) -> Optional[torch.Tensor]:
        """Prepare model input from data."""
        # Get RSI feature
        rsi_tensor = self.conditioner.get_condition_tensor(data)
        rsi_window = rsi_tensor[-self.window_size:]

        # Extract features (close prices for simple case)
        features = data[["close"]].values[-self.window_size:]

        if len(features) < self.window_size:
            return None

        # Normalize if normalizer is available
        if self.normalizer is not None:
            features = self.normalizer.transform(features)
            if isinstance(features, pd.DataFrame):
                features = features.values

        # Combine features (Price + RSI)
        # Check shapes: features is (T, 1), rsi_window is (T, 1)
        combined = np.hstack([features, rsi_window])

        # Convert to tensor
        tensor = torch.from_numpy(combined.astype(np.float32))
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        tensor = tensor.to(self.device)

        return tensor

    def _generate_samples(self, history: torch.Tensor) -> Optional[np.ndarray]:
        """Generate Monte Carlo samples from the model."""
        try:
            with torch.no_grad():
                samples = []
                batch_size = 10  # Generate in batches for efficiency

                for _ in range(self.config.num_samples // batch_size):
                    # Expand history for batch
                    history_batch = history.expand(batch_size, -1, -1)

                    # Sample from model
                    output = self.model.sample(
                        shape=(batch_size, self.config.prediction_horizon, 1),
                        condition=history_batch,
                    )

                    # Sum over horizon for total return
                    total_return = output.sum(dim=1).squeeze(-1)
                    samples.append(total_return.cpu().numpy())

                samples = np.concatenate(samples)

                # Denormalize if needed
                if self.normalizer is not None:
                    # For returns, we might need different handling
                    pass

                return samples

        except Exception as e:
            return None

    def _calculate_allocation(
        self,
        expected_return: float,
        return_std: float,
        confidence: float,
    ) -> float:
        """
        Calculate position allocation based on Kelly criterion.

        Uses fractional Kelly for more conservative sizing.
        """
        if return_std == 0:
            return 0.0

        # Simplified Kelly: f = mu / sigma^2
        kelly = expected_return / (return_std ** 2) if return_std > 0 else 0

        # Apply fractional Kelly (half Kelly)
        fractional_kelly = kelly * 0.5

        # Scale by confidence
        allocation = fractional_kelly * confidence

        # Clamp to max position size
        allocation = max(-self.config.max_position_size, min(self.config.max_position_size, allocation))

        return allocation

    def _calculate_stop_from_distribution(
        self,
        current_price: float,
        position: Position,
        prediction: dict,
    ) -> float:
        """Calculate stop-loss from prediction distribution."""
        # Use VaR as stop-loss reference
        var_95 = prediction["var_95"]

        if position == Position.LONG:
            # Stop at 95% VaR level
            return current_price * (1 + var_95)
        elif position == Position.SHORT:
            # For short, use upper percentile
            upper_95 = prediction["percentile_75"] + 1.5 * prediction["std"]
            return current_price * (1 + upper_95)

        return current_price

    def _calculate_tp_from_distribution(
        self,
        current_price: float,
        position: Position,
        prediction: dict,
    ) -> float:
        """Calculate take-profit from prediction distribution."""
        expected = prediction["mean"]

        if position == Position.LONG:
            # Target at expected return
            return current_price * (1 + expected)
        elif position == Position.SHORT:
            return current_price * (1 + expected)  # expected is negative

        return current_price

    def get_required_lookback(self) -> int:
        """Return required historical data points."""
        return self.window_size

    def get_required_features(self) -> list[str]:
        """Return required data columns."""
        return ["open", "high", "low", "close", "volume"]


class EnsembleDiffusionStrategy(DiffusionPredictionStrategy):
    """
    Ensemble of multiple diffusion models for more robust predictions.

    Combines predictions from multiple models trained with different
    seeds or architectures.
    """

    def __init__(
        self,
        models: list[torch.nn.Module],
        window_size: int = 64,
        normalizer=None,
        device: str = "auto",
        config: DiffusionStrategyConfig = None,
    ):
        """
        Initialize ensemble strategy.

        Args:
            models: List of trained DDPM models
            window_size: Historical window size
            normalizer: Data normalizer
            device: Compute device
            config: Strategy configuration
        """
        # Initialize with first model
        super().__init__(models[0], window_size, normalizer, device, config)
        self.models = models

        for model in self.models:
            model.eval()
            model.to(device)

    def _generate_samples(self, history: torch.Tensor) -> Optional[np.ndarray]:
        """Generate samples from all models in the ensemble."""
        all_samples = []

        samples_per_model = self.config.num_samples // len(self.models)

        for model in self.models:
            try:
                with torch.no_grad():
                    samples = []

                    for _ in range(samples_per_model // 10):
                        history_batch = history.expand(10, -1, -1)

                        output = model.sample(
                            shape=(10, self.config.prediction_horizon, 1),
                            condition=history_batch,
                        )

                        total_return = output.sum(dim=1).squeeze(-1)
                        samples.append(total_return.cpu().numpy())

                    all_samples.extend(np.concatenate(samples))

            except Exception:
                continue

        if not all_samples:
            return None

        return np.array(all_samples)
