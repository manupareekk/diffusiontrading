"""
Band-based diffusion trading strategy.

Generates trading signals when price crosses predicted bands.
"""

from typing import Optional, Dict
import numpy as np
import pandas as pd
import torch

from .diffusion_strategy import DiffusionPredictionStrategy, DiffusionStrategyConfig
from .base import Position, Signal


class BandBasedDiffusionStrategy(DiffusionPredictionStrategy):
    """
    Band-based trading strategy using diffusion model predictions.
    
    Uses prediction distributions to create upper/lower bands.
    Generates buy signal when price < lower band (oversold).
    Generates sell signal when price > upper band (overbought).
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        window_size: int = 64,
        normalizer=None,
        device: str = "auto",
        config: DiffusionStrategyConfig = None,
        prediction_horizon: int = 5,
        band_type: str = "percentile",  # "percentile" or "std"
        band_width: str = "1std",  # "1std", "2std", "percentile_5_95", "percentile_10_90"
    ):
        """
        Initialize band-based strategy.
        
        Args:
            model: Trained DDPM model
            window_size: Historical window size
            normalizer: Data normalizer
            device: Compute device
            config: Strategy configuration
            prediction_horizon: Steps ahead to predict
            band_type: "percentile" or "std" based bands
            band_width: Width of bands
        """
        super().__init__(model, window_size, normalizer, device, config)
        self.prediction_horizon = prediction_horizon
        self.band_type = band_type
        self.band_width = band_width
        
        # Update config prediction horizon
        self.config.prediction_horizon = prediction_horizon
        
    def calculate_bands(self, prediction: Dict) -> tuple[float, float, float]:
        """
        Calculate upper and lower bands from prediction distribution.
        
        Args:
            prediction: Dict with samples, mean, std, percentiles
            
        Returns:
            Tuple of (lower_band, mean_prediction, upper_band)
        """
        samples = prediction["samples"]
        
        if self.band_type == "std":
            # Standard deviation based bands (Bollinger-style)
            mean = prediction["mean"]
            std = prediction["std"]
            
            if self.band_width == "1std":
                lower = mean - std
                upper = mean + std
            elif self.band_width == "2std":
                lower = mean - 2 * std
                upper = mean + 2 * std
            else:
                lower = mean - std
                upper = mean + std
                
        else:  # percentile-based
            if self.band_width == "percentile_5_95":
                lower = prediction["var_95"]  # 5th percentile
                upper = float(np.percentile(samples, 95))
            elif self.band_width == "percentile_10_90":
                lower = float(np.percentile(samples, 10))
                upper = float(np.percentile(samples, 90))
            else:  # default to 5-95
                lower = prediction["var_95"]
                upper = float(np.percentile(samples, 95))
                
        mean = prediction["mean"]
        return lower, mean, upper
        
    def generate_signals(
        self,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> list[Signal]:
        """
        Generate trading signals based on band crossings.
        
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
            
        # Calculate bands
        lower_band, mean_pred, upper_band = self.calculate_bands(prediction)
        
        # Convert from returns to prices
        # Predictions are cumulative returns over horizon
        predicted_lower_price = current_price * (1 + lower_band)
        predicted_mean_price = current_price * (1 + mean_pred)
        predicted_upper_price = current_price * (1 + upper_band)
        
        # Generate signal based on current price vs predicted bands
        # If current price is below lower band → oversold → BUY
        # If current price is above upper band → overbought → SELL
        
        position = None
        confidence = 0.0
        
        # Use a threshold to avoid whipsaws
        band_range = predicted_upper_price - predicted_lower_price
        threshold = band_range * 0.1  # 10% of band width
        
        if current_price < (predicted_lower_price - threshold):
            # Price significantly below predicted range → oversold
            position = Position.LONG
            # Confidence based on how far below
            distance = (predicted_lower_price - current_price) / band_range
            confidence = min(0.9, 0.5 + distance)
            
        elif current_price > (predicted_upper_price + threshold):
            # Price significantly above predicted range → overbought
            position = Position.SHORT
            # Confidence based on how far above
            distance = (current_price - predicted_upper_price) / band_range
            confidence = min(0.9, 0.5 + distance)
            
        if position is None:
            return signals
            
        # Calculate allocation
        expected_return = mean_pred
        return_std = prediction["std"]
        allocation = self._calculate_allocation(
            abs(expected_return), return_std, confidence
        )
        
        if position == Position.SHORT:
            allocation = -allocation
            
        # Calculate stop-loss and take-profit
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
                "return_std": return_std,
                "lower_band": lower_band,
                "upper_band": upper_band,
                "predicted_lower_price": predicted_lower_price,
                "predicted_mean_price": predicted_mean_price,
                "predicted_upper_price": predicted_upper_price,
                "current_price": current_price,
                "band_type": self.band_type,
                "band_width": self.band_width,
                "prediction_horizon": self.prediction_horizon,
            },
        )
        
        signals.append(signal)
        return signals
        
    def get_prediction_bands(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Get prediction bands for visualization.
        
        Args:
            data: Historical market data
            
        Returns:
            Dictionary with band information
        """
        prediction = self.predict_return_distribution(data)
        if prediction is None:
            return None
            
        current_price = data["close"].iloc[-1]
        lower_band, mean_pred, upper_band = self.calculate_bands(prediction)
        
        return {
            "lower_band": lower_band,
            "mean_prediction": mean_pred,
            "upper_band": upper_band,
            "lower_price": current_price * (1 + lower_band),
            "mean_price": current_price * (1 + mean_pred),
            "upper_price": current_price * (1 + upper_band),
            "current_price": current_price,
        }
