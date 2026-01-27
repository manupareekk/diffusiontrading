"""
Data normalization utilities for financial time series.

Provides various normalization methods that are essential for
neural network training. All normalizers support fit/transform
pattern to prevent lookahead bias.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class NormalizerState:
    """Stores normalization parameters for later use."""

    method: str
    params: dict = field(default_factory=dict)
    is_fitted: bool = False


class BaseNormalizer(ABC):
    """
    Abstract base class for normalizers.

    All normalizers follow the fit/transform pattern to ensure
    test data is normalized using only training data statistics.
    """

    def __init__(self):
        self.state = NormalizerState(method=self.__class__.__name__)

    @abstractmethod
    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> "BaseNormalizer":
        """
        Fit the normalizer on training data.

        Args:
            data: Training data to compute normalization parameters

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform data using fitted parameters.

        Args:
            data: Data to normalize

        Returns:
            Normalized data
        """
        pass

    @abstractmethod
    def inverse_transform(
        self, data: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Reverse the normalization.

        Args:
            data: Normalized data

        Returns:
            Original scale data
        """
        pass

    def fit_transform(
        self, data: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)

    def _ensure_fitted(self) -> None:
        """Raise error if not fitted."""
        if not self.state.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")


class ZScoreNormalizer(BaseNormalizer):
    """
    Z-score (standard) normalization: (x - mean) / std

    Transforms data to have zero mean and unit variance.
    Good for data that is approximately normally distributed.
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: Small constant to prevent division by zero
        """
        super().__init__()
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> "ZScoreNormalizer":
        """Compute mean and std from data."""
        arr = self._to_array(data)
        self.mean_ = np.mean(arr, axis=0)
        self.std_ = np.std(arr, axis=0) + self.eps

        self.state.params = {
            "mean": self.mean_.tolist() if isinstance(self.mean_, np.ndarray) else self.mean_,
            "std": self.std_.tolist() if isinstance(self.std_, np.ndarray) else self.std_,
        }
        self.state.is_fitted = True
        return self

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Apply z-score normalization."""
        self._ensure_fitted()
        arr = self._to_array(data)
        normalized = (arr - self.mean_) / self.std_
        return self._restore_type(normalized, data)

    def inverse_transform(
        self, data: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Reverse z-score normalization."""
        self._ensure_fitted()
        arr = self._to_array(data)
        original = arr * self.std_ + self.mean_
        return self._restore_type(original, data)

    @staticmethod
    def _to_array(data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            return data.values
        return np.asarray(data)

    @staticmethod
    def _restore_type(
        arr: np.ndarray, original: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        if isinstance(original, pd.DataFrame):
            return pd.DataFrame(arr, index=original.index, columns=original.columns)
        return arr


class MinMaxNormalizer(BaseNormalizer):
    """
    Min-max normalization: (x - min) / (max - min)

    Scales data to a fixed range, typically [0, 1] or [-1, 1].
    Good for data with known bounds.
    """

    def __init__(self, feature_range: tuple = (0, 1), eps: float = 1e-8):
        """
        Args:
            feature_range: Target range (min, max)
            eps: Small constant to prevent division by zero
        """
        super().__init__()
        self.feature_range = feature_range
        self.eps = eps
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None

    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> "MinMaxNormalizer":
        """Compute min and max from data."""
        arr = self._to_array(data)
        self.min_ = np.min(arr, axis=0)
        self.max_ = np.max(arr, axis=0)

        self.state.params = {
            "min": self.min_.tolist() if isinstance(self.min_, np.ndarray) else self.min_,
            "max": self.max_.tolist() if isinstance(self.max_, np.ndarray) else self.max_,
            "feature_range": self.feature_range,
        }
        self.state.is_fitted = True
        return self

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Apply min-max normalization."""
        self._ensure_fitted()
        arr = self._to_array(data)

        # Scale to [0, 1]
        range_ = self.max_ - self.min_ + self.eps
        scaled = (arr - self.min_) / range_

        # Scale to target range
        min_range, max_range = self.feature_range
        normalized = scaled * (max_range - min_range) + min_range

        return self._restore_type(normalized, data)

    def inverse_transform(
        self, data: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Reverse min-max normalization."""
        self._ensure_fitted()
        arr = self._to_array(data)

        min_range, max_range = self.feature_range
        scaled = (arr - min_range) / (max_range - min_range)

        range_ = self.max_ - self.min_ + self.eps
        original = scaled * range_ + self.min_

        return self._restore_type(original, data)

    @staticmethod
    def _to_array(data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            return data.values
        return np.asarray(data)

    @staticmethod
    def _restore_type(
        arr: np.ndarray, original: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        if isinstance(original, pd.DataFrame):
            return pd.DataFrame(arr, index=original.index, columns=original.columns)
        return arr


class LogReturnNormalizer(BaseNormalizer):
    """
    Log return transformation: log(x_t / x_{t-1})

    Standard normalization for financial price data.
    Transforms prices to log returns which are:
    - Approximately stationary
    - Time-additive
    - Symmetric around zero
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: Small constant to prevent log(0)
        """
        super().__init__()
        self.eps = eps
        self.first_value_: Optional[np.ndarray] = None

    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> "LogReturnNormalizer":
        """Store first value for inverse transform."""
        arr = self._to_array(data)
        self.first_value_ = arr[0].copy()

        self.state.params = {
            "first_value": (
                self.first_value_.tolist()
                if isinstance(self.first_value_, np.ndarray)
                else self.first_value_
            ),
        }
        self.state.is_fitted = True
        return self

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Compute log returns."""
        self._ensure_fitted()
        arr = self._to_array(data)

        # Compute log returns: log(p_t / p_{t-1})
        log_returns = np.log(arr[1:] / (arr[:-1] + self.eps) + self.eps)

        # Prepend a zero for the first value to maintain shape
        log_returns = np.vstack([np.zeros((1, arr.shape[1])), log_returns])

        return self._restore_type(log_returns, data)

    def inverse_transform(
        self, data: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Convert log returns back to prices."""
        self._ensure_fitted()
        arr = self._to_array(data)

        # Reconstruct prices from log returns
        cumulative_returns = np.exp(np.cumsum(arr, axis=0))
        prices = self.first_value_ * cumulative_returns

        return self._restore_type(prices, data)

    @staticmethod
    def _to_array(data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            return data.values
        return np.asarray(data)

    @staticmethod
    def _restore_type(
        arr: np.ndarray, original: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        if isinstance(original, pd.DataFrame):
            return pd.DataFrame(arr, index=original.index, columns=original.columns)
        return arr


class RobustNormalizer(BaseNormalizer):
    """
    Robust normalization using median and IQR.

    (x - median) / IQR

    Less sensitive to outliers than z-score normalization.
    Good for financial data which often has fat tails.
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: Small constant to prevent division by zero
        """
        super().__init__()
        self.eps = eps
        self.median_: Optional[np.ndarray] = None
        self.iqr_: Optional[np.ndarray] = None

    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> "RobustNormalizer":
        """Compute median and IQR from data."""
        arr = self._to_array(data)
        self.median_ = np.median(arr, axis=0)
        q75 = np.percentile(arr, 75, axis=0)
        q25 = np.percentile(arr, 25, axis=0)
        self.iqr_ = q75 - q25 + self.eps

        self.state.params = {
            "median": (
                self.median_.tolist() if isinstance(self.median_, np.ndarray) else self.median_
            ),
            "iqr": self.iqr_.tolist() if isinstance(self.iqr_, np.ndarray) else self.iqr_,
        }
        self.state.is_fitted = True
        return self

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Apply robust normalization."""
        self._ensure_fitted()
        arr = self._to_array(data)
        normalized = (arr - self.median_) / self.iqr_
        return self._restore_type(normalized, data)

    def inverse_transform(
        self, data: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Reverse robust normalization."""
        self._ensure_fitted()
        arr = self._to_array(data)
        original = arr * self.iqr_ + self.median_
        return self._restore_type(original, data)

    @staticmethod
    def _to_array(data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            return data.values
        return np.asarray(data)

    @staticmethod
    def _restore_type(
        arr: np.ndarray, original: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        if isinstance(original, pd.DataFrame):
            return pd.DataFrame(arr, index=original.index, columns=original.columns)
        return arr


def create_normalizer(method: str, **kwargs) -> BaseNormalizer:
    """
    Factory function to create normalizers.

    Args:
        method: Normalization method ('zscore', 'minmax', 'logreturns', 'robust')
        **kwargs: Additional arguments for the normalizer

    Returns:
        Normalizer instance
    """
    normalizers = {
        "zscore": ZScoreNormalizer,
        "minmax": MinMaxNormalizer,
        "logreturns": LogReturnNormalizer,
        "robust": RobustNormalizer,
    }

    if method not in normalizers:
        raise ValueError(f"Unknown normalization method: {method}. Available: {list(normalizers.keys())}")

    return normalizers[method](**kwargs)
