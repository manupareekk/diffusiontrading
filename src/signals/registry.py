"""
Signal registry for dynamic signal registration and discovery.

Provides a unified interface for computing technical and fundamental
signals with automatic registration and validation.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import pandas as pd
from loguru import logger


@dataclass
class SignalMetadata:
    """Metadata describing a registered signal."""

    name: str
    category: str  # "momentum", "trend", "volatility", "volume", "fundamental"
    lookback_period: int  # Minimum history required
    output_columns: list[str]  # Names of output columns
    description: str = ""
    default_params: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.output_columns:
            self.output_columns = [self.name]


# Type alias for signal functions
SignalFunction = Callable[[pd.DataFrame, Optional[dict]], pd.DataFrame]


class SignalRegistry:
    """
    Registry pattern for signal discovery and management.

    Allows dynamic registration of new signals via decorator
    and provides a unified interface for signal computation.

    Usage:
        @SignalRegistry.register(SignalMetadata(...))
        def my_signal(data: pd.DataFrame, params: dict = None) -> pd.DataFrame:
            ...

        # Later:
        result = SignalRegistry.compute("my_signal", data, params)
    """

    _signals: dict[str, tuple[SignalFunction, SignalMetadata]] = {}

    @classmethod
    def register(cls, metadata: SignalMetadata):
        """
        Decorator to register a signal function.

        Args:
            metadata: SignalMetadata describing the signal

        Returns:
            Decorator function
        """

        def decorator(func: SignalFunction) -> SignalFunction:
            cls._signals[metadata.name] = (func, metadata)
            logger.debug(f"Registered signal: {metadata.name}")
            return func

        return decorator

    @classmethod
    def compute(
        cls,
        name: str,
        data: pd.DataFrame,
        params: dict = None,
    ) -> pd.DataFrame:
        """
        Compute a registered signal.

        Args:
            name: Name of the registered signal
            data: Input DataFrame with OHLCV data
            params: Optional parameters to override defaults

        Returns:
            DataFrame with signal values

        Raises:
            KeyError: If signal is not registered
            ValueError: If data is insufficient
        """
        if name not in cls._signals:
            raise KeyError(
                f"Signal '{name}' not registered. "
                f"Available: {list(cls._signals.keys())}"
            )

        func, metadata = cls._signals[name]

        # Validate data length
        if len(data) < metadata.lookback_period:
            raise ValueError(
                f"Signal '{name}' requires at least {metadata.lookback_period} "
                f"data points, got {len(data)}"
            )

        # Merge with default params
        final_params = {**metadata.default_params, **(params or {})}

        try:
            result = func(data, final_params)
            return result
        except Exception as e:
            logger.error(f"Error computing signal '{name}': {e}")
            raise

    @classmethod
    def compute_all(
        cls,
        data: pd.DataFrame,
        signal_configs: list[dict],
    ) -> pd.DataFrame:
        """
        Compute multiple signals and concatenate results.

        Args:
            data: Input DataFrame with OHLCV data
            signal_configs: List of signal configurations, each containing:
                - name: Signal name
                - params: Optional parameters dict

        Returns:
            DataFrame with all signal values

        Example:
            signals = SignalRegistry.compute_all(data, [
                {"name": "rsi", "params": {"period": 14}},
                {"name": "macd"},
                {"name": "bollinger"},
            ])
        """
        results = []

        for config in signal_configs:
            name = config["name"]
            params = config.get("params", None)

            try:
                result = cls.compute(name, data, params)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to compute signal '{name}': {e}")
                continue

        if not results:
            return pd.DataFrame(index=data.index)

        return pd.concat(results, axis=1)

    @classmethod
    def get_metadata(cls, name: str) -> SignalMetadata:
        """Get metadata for a registered signal."""
        if name not in cls._signals:
            raise KeyError(f"Signal '{name}' not registered")
        return cls._signals[name][1]

    @classmethod
    def list_signals(cls, category: str = None) -> list[str]:
        """
        List registered signals.

        Args:
            category: Optional category filter

        Returns:
            List of signal names
        """
        if category is None:
            return list(cls._signals.keys())

        return [
            name
            for name, (_, metadata) in cls._signals.items()
            if metadata.category == category
        ]

    @classmethod
    def get_all_metadata(cls) -> dict[str, SignalMetadata]:
        """Get metadata for all registered signals."""
        return {name: meta for name, (_, meta) in cls._signals.items()}

    @classmethod
    def get_total_lookback(cls, signal_names: list[str]) -> int:
        """
        Get the maximum lookback period for a list of signals.

        Useful for determining minimum data requirements.

        Args:
            signal_names: List of signal names

        Returns:
            Maximum lookback period
        """
        max_lookback = 0
        for name in signal_names:
            if name in cls._signals:
                _, metadata = cls._signals[name]
                max_lookback = max(max_lookback, metadata.lookback_period)
        return max_lookback

    @classmethod
    def clear(cls):
        """Clear all registered signals (useful for testing)."""
        cls._signals.clear()


class SignalPipeline:
    """
    Pipeline for computing multiple signals in sequence.

    Provides a more structured way to define signal computation
    workflows with validation and error handling.
    """

    def __init__(self, signal_configs: list[dict] = None):
        """
        Initialize the pipeline.

        Args:
            signal_configs: List of signal configurations
        """
        self.signal_configs = signal_configs or []

    def add_signal(
        self,
        name: str,
        params: dict = None,
    ) -> "SignalPipeline":
        """
        Add a signal to the pipeline.

        Args:
            name: Signal name
            params: Signal parameters

        Returns:
            self for method chaining
        """
        self.signal_configs.append({"name": name, "params": params})
        return self

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all signals in the pipeline.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with all signal values
        """
        return SignalRegistry.compute_all(data, self.signal_configs)

    def get_required_lookback(self) -> int:
        """Get minimum data requirements for this pipeline."""
        names = [config["name"] for config in self.signal_configs]
        return SignalRegistry.get_total_lookback(names)

    def validate(self) -> list[str]:
        """
        Validate the pipeline configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        for config in self.signal_configs:
            name = config["name"]
            if name not in SignalRegistry._signals:
                errors.append(f"Unknown signal: {name}")
        return errors

    @classmethod
    def from_config(cls, config: list[dict]) -> "SignalPipeline":
        """Create pipeline from configuration dict."""
        return cls(signal_configs=config)


# Default signal configurations for intraday trading
INTRADAY_SIGNAL_CONFIG = [
    {"name": "rsi", "params": {"period": 9}},
    {"name": "macd", "params": {"fast": 8, "slow": 17, "signal": 9}},
    {"name": "stochastic", "params": {"k_period": 9, "d_period": 3}},
    {"name": "bollinger", "params": {"period": 20, "std": 2.0}},
    {"name": "atr", "params": {"period": 14}},
    {"name": "vwap"},
    {"name": "obv"},
]

# Default signal configurations for daily trading
DAILY_SIGNAL_CONFIG = [
    {"name": "rsi", "params": {"period": 14}},
    {"name": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}},
    {"name": "stochastic", "params": {"k_period": 14, "d_period": 3}},
    {"name": "bollinger", "params": {"period": 20, "std": 2.0}},
    {"name": "atr", "params": {"period": 14}},
    {"name": "sma", "params": {"period": 50}},
    {"name": "ema", "params": {"span": 20}},
]
