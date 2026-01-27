"""
Yahoo Finance data provider implementation.

Provides intraday and daily OHLCV data via the yfinance library.
Includes caching to respect rate limits and improve performance.
"""

import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from .base import BaseDataProvider, DataRequest, DataResponse


class YFinanceProvider(BaseDataProvider):
    """
    Yahoo Finance data provider.

    Limitations for free tier:
    - 1m data: max 7 days history
    - 2m data: max 60 days history
    - 5m-15m data: max 60 days history
    - 30m-1h data: max 730 days history
    - 1d data: full history available

    Implements local caching to reduce API calls.
    """

    # Interval to max history mapping (in days)
    INTERVAL_LIMITS = {
        "1m": 7,
        "2m": 60,
        "5m": 60,
        "15m": 60,
        "30m": 730,
        "60m": 730,
        "1h": 730,
        "1d": 10000,  # Effectively unlimited
    }

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        cache_expiry_hours: int = 24,
        use_cache: bool = True,
    ):
        """
        Initialize the YFinance provider.

        Args:
            cache_dir: Directory for caching data (None = no caching)
            cache_expiry_hours: Hours before cache expires
            use_cache: Whether to use caching
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_expiry_hours = cache_expiry_hours
        self.use_cache = use_cache and cache_dir is not None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_rate_limits(self) -> dict:
        """Return yfinance rate limit info (unofficial API, limits are soft)."""
        return {
            "requests_per_minute": 60,
            "requests_per_day": 2000,
            "note": "yfinance has soft rate limits; excessive use may result in IP blocking",
        }

    def get_available_intervals(self) -> list[str]:
        """Return available data intervals."""
        return list(self.INTERVAL_LIMITS.keys())

    def fetch_ohlcv(self, request: DataRequest) -> DataResponse:
        """
        Fetch OHLCV data from Yahoo Finance.

        Args:
            request: DataRequest with symbols, dates, interval

        Returns:
            DataResponse with market data
        """
        self.validate_request(request)
        self._validate_date_range(request)

        # Check cache first
        if self.use_cache:
            cached = self._load_from_cache(request)
            if cached is not None:
                logger.debug(f"Loaded {len(cached)} bars from cache")
                return DataResponse(
                    data=cached,
                    symbols=request.symbols,
                    interval=request.interval,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    metadata={"source": "cache"},
                )

        # Fetch from API
        logger.info(
            f"Fetching {request.symbols} from {request.start_date} to {request.end_date} "
            f"({request.interval})"
        )

        try:
            all_data = []

            for symbol in request.symbols:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=request.start_date,
                    end=request.end_date,
                    interval=request.interval,
                    prepost=request.include_premarket or request.include_afterhours,
                    auto_adjust=request.adjust_prices,
                )

                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    continue

                # Standardize columns
                df = self.standardize_columns(df)

                # Add symbol column for multi-symbol support
                df["symbol"] = symbol

                all_data.append(df)

            if not all_data:
                return DataResponse(
                    data=pd.DataFrame(),
                    symbols=request.symbols,
                    interval=request.interval,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    metadata={"source": "api", "error": "no_data"},
                )

            # Combine all symbols
            combined = pd.concat(all_data)

            # Filter to market hours if requested
            if not request.include_premarket and not request.include_afterhours:
                combined = self._filter_market_hours(combined)

            # Cache the result
            if self.use_cache:
                self._save_to_cache(request, combined)

            return DataResponse(
                data=combined,
                symbols=request.symbols,
                interval=request.interval,
                start_date=request.start_date,
                end_date=request.end_date,
                metadata={"source": "api"},
            )

        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise ConnectionError(f"Yahoo Finance API error: {e}") from e

    def _validate_date_range(self, request: DataRequest) -> None:
        """Validate that date range is within limits for the interval."""
        max_days = self.INTERVAL_LIMITS.get(request.interval, 7)
        requested_days = (request.end_date - request.start_date).days

        if requested_days > max_days:
            raise ValueError(
                f"Date range ({requested_days} days) exceeds maximum "
                f"({max_days} days) for interval '{request.interval}'"
            )

    def _filter_market_hours(
        self,
        df: pd.DataFrame,
        market_open: str = "09:30",
        market_close: str = "16:00",
    ) -> pd.DataFrame:
        """
        Filter DataFrame to regular market hours only.

        Args:
            df: DataFrame with datetime index
            market_open: Market open time (HH:MM)
            market_close: Market close time (HH:MM)

        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        # Parse market hours
        open_hour, open_min = map(int, market_open.split(":"))
        close_hour, close_min = map(int, market_close.split(":"))

        open_time = df.index.hour * 60 + df.index.minute
        market_open_mins = open_hour * 60 + open_min
        market_close_mins = close_hour * 60 + close_min

        mask = (open_time >= market_open_mins) & (open_time < market_close_mins)

        return df[mask]

    def _get_cache_key(self, request: DataRequest) -> str:
        """Generate a unique cache key for the request."""
        key_data = f"{sorted(request.symbols)}_{request.start_date}_{request.end_date}_{request.interval}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, request: DataRequest) -> Path:
        """Get the cache file path for a request."""
        cache_key = self._get_cache_key(request)
        return self.cache_dir / f"{cache_key}.parquet"

    def _load_from_cache(self, request: DataRequest) -> Optional[pd.DataFrame]:
        """Load data from cache if valid."""
        if not self.cache_dir:
            return None

        cache_path = self._get_cache_path(request)

        if not cache_path.exists():
            return None

        # Check expiry
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if file_age > timedelta(hours=self.cache_expiry_hours):
            logger.debug(f"Cache expired for {cache_path}")
            cache_path.unlink()
            return None

        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_to_cache(self, request: DataRequest, data: pd.DataFrame) -> None:
        """Save data to cache."""
        if not self.cache_dir or data.empty:
            return

        cache_path = self._get_cache_path(request)

        try:
            data.to_parquet(cache_path)
            logger.debug(f"Cached {len(data)} bars to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def fetch_latest(
        self,
        symbols: list[str],
        interval: str = "5m",
        lookback_bars: int = 100,
    ) -> DataResponse:
        """
        Fetch the most recent data for given symbols.

        Convenience method for getting latest bars.

        Args:
            symbols: List of ticker symbols
            interval: Data interval
            lookback_bars: Number of bars to fetch

        Returns:
            DataResponse with recent data
        """
        # Calculate appropriate date range
        interval_minutes = self._interval_to_minutes(interval)
        total_minutes = lookback_bars * interval_minutes

        # Add buffer for weekends/holidays
        total_days = max(1, (total_minutes // (6 * 60)) + 2)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=total_days)

        request = DataRequest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )

        response = self.fetch_ohlcv(request)

        # Trim to requested number of bars
        if not response.is_empty:
            response.data = response.data.tail(lookback_bars * len(symbols))

        return response

    @staticmethod
    def _interval_to_minutes(interval: str) -> int:
        """Convert interval string to minutes."""
        mapping = {
            "1m": 1,
            "2m": 2,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "60m": 60,
            "1h": 60,
            "1d": 390,  # Trading day
        }
        return mapping.get(interval, 5)
