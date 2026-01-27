"""
Abstract base class for data providers.

All data providers (yfinance, Alpaca, Polygon) must implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd


@dataclass
class DataRequest:
    """Request specification for fetching market data."""

    symbols: list[str]
    start_date: datetime
    end_date: datetime
    interval: str = "5m"  # 1m, 2m, 5m, 15m, 30m, 1h, 1d
    include_premarket: bool = False
    include_afterhours: bool = False
    adjust_prices: bool = True  # Adjust for splits/dividends

    def __post_init__(self):
        if isinstance(self.symbols, str):
            self.symbols = [self.symbols]


@dataclass
class DataResponse:
    """Response containing fetched market data."""

    data: pd.DataFrame
    symbols: list[str]
    interval: str
    start_date: datetime
    end_date: datetime
    metadata: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.data is None or len(self.data) == 0

    @property
    def num_bars(self) -> int:
        return len(self.data) if self.data is not None else 0


class BaseDataProvider(ABC):
    """
    Abstract base class for market data providers.

    All providers must implement:
    - fetch_ohlcv: Get OHLCV (Open, High, Low, Close, Volume) data
    - get_rate_limits: Return API rate limit information
    """

    @abstractmethod
    def fetch_ohlcv(self, request: DataRequest) -> DataResponse:
        """
        Fetch OHLCV data for given symbols and date range.

        Args:
            request: DataRequest specifying symbols, dates, and interval

        Returns:
            DataResponse containing the market data

        Raises:
            ValueError: If request parameters are invalid
            ConnectionError: If API connection fails
        """
        pass

    @abstractmethod
    def get_rate_limits(self) -> dict:
        """
        Return API rate limit information.

        Returns:
            dict with keys like 'requests_per_minute', 'requests_per_day'
        """
        pass

    @abstractmethod
    def get_available_intervals(self) -> list[str]:
        """Return list of supported data intervals."""
        pass

    def validate_request(self, request: DataRequest) -> None:
        """
        Validate a data request.

        Args:
            request: DataRequest to validate

        Raises:
            ValueError: If request is invalid
        """
        if not request.symbols:
            raise ValueError("At least one symbol is required")

        if request.start_date >= request.end_date:
            raise ValueError("start_date must be before end_date")

        available_intervals = self.get_available_intervals()
        if request.interval not in available_intervals:
            raise ValueError(
                f"Invalid interval '{request.interval}'. "
                f"Available: {available_intervals}"
            )

    @staticmethod
    def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to lowercase.

        Args:
            df: DataFrame with potentially mixed-case columns

        Returns:
            DataFrame with lowercase column names
        """
        df.columns = df.columns.str.lower()

        # Ensure required columns exist
        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df[required]
