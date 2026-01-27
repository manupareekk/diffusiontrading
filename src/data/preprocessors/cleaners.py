"""
Data cleaning utilities for financial time series.

Handles missing data, outliers, and data quality issues
that are common in market data.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger


class MissingDataMethod(Enum):
    """Methods for handling missing data."""

    DROP = "drop"  # Remove rows with missing values
    FORWARD_FILL = "ffill"  # Forward fill (use previous value)
    BACKWARD_FILL = "bfill"  # Backward fill (use next value)
    INTERPOLATE = "interpolate"  # Linear interpolation
    ZERO = "zero"  # Fill with zero


class OutlierMethod(Enum):
    """Methods for handling outliers."""

    CLIP = "clip"  # Clip to threshold
    REMOVE = "remove"  # Remove outlier rows
    WINSORIZE = "winsorize"  # Winsorize (replace with percentile value)


@dataclass
class CleaningReport:
    """Report of cleaning operations performed."""

    original_rows: int
    final_rows: int
    missing_filled: int
    outliers_handled: int
    duplicates_removed: int
    gaps_detected: int

    @property
    def rows_removed(self) -> int:
        return self.original_rows - self.final_rows

    @property
    def data_loss_pct(self) -> float:
        if self.original_rows == 0:
            return 0.0
        return (self.rows_removed / self.original_rows) * 100


class DataCleaner:
    """
    Comprehensive data cleaning for financial time series.

    Handles:
    - Missing values
    - Outliers
    - Duplicates
    - Time gaps
    - Data type issues
    """

    def __init__(
        self,
        missing_method: MissingDataMethod = MissingDataMethod.FORWARD_FILL,
        outlier_method: OutlierMethod = OutlierMethod.CLIP,
        outlier_threshold: float = 5.0,  # Standard deviations
        max_missing_pct: float = 0.05,
        max_gap_minutes: int = 30,
    ):
        """
        Initialize the data cleaner.

        Args:
            missing_method: Method to handle missing values
            outlier_method: Method to handle outliers
            outlier_threshold: Number of standard deviations for outlier detection
            max_missing_pct: Maximum allowed percentage of missing data
            max_gap_minutes: Maximum allowed time gap in minutes
        """
        self.missing_method = missing_method
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.max_missing_pct = max_missing_pct
        self.max_gap_minutes = max_gap_minutes

    def clean(
        self,
        data: pd.DataFrame,
        price_columns: list[str] = None,
        volume_column: str = "volume",
    ) -> tuple[pd.DataFrame, CleaningReport]:
        """
        Clean the data and return cleaning report.

        Args:
            data: DataFrame with OHLCV data
            price_columns: Columns containing price data (default: ohlc)
            volume_column: Column containing volume data

        Returns:
            Tuple of (cleaned DataFrame, CleaningReport)
        """
        if data.empty:
            return data, CleaningReport(0, 0, 0, 0, 0, 0)

        price_columns = price_columns or ["open", "high", "low", "close"]
        original_rows = len(data)

        df = data.copy()

        # Track metrics
        missing_filled = 0
        outliers_handled = 0
        duplicates_removed = 0
        gaps_detected = 0

        # 1. Remove duplicates
        dup_count = df.index.duplicated().sum()
        if dup_count > 0:
            df = df[~df.index.duplicated(keep="first")]
            duplicates_removed = dup_count
            logger.debug(f"Removed {dup_count} duplicate rows")

        # 2. Sort by index
        df = df.sort_index()

        # 3. Detect time gaps
        if isinstance(df.index, pd.DatetimeIndex):
            gaps_detected = self._detect_gaps(df)

        # 4. Handle missing values
        missing_before = df.isnull().sum().sum()
        df = self._handle_missing(df)
        missing_filled = missing_before - df.isnull().sum().sum()

        # 5. Validate OHLC relationships
        df = self._validate_ohlc(df, price_columns)

        # 6. Handle outliers in price columns
        for col in price_columns:
            if col in df.columns:
                outliers = self._detect_outliers(df[col])
                outliers_handled += outliers.sum()
                df = self._handle_outliers(df, col)

        # 7. Handle outliers in volume
        if volume_column in df.columns:
            outliers = self._detect_outliers(df[volume_column])
            outliers_handled += outliers.sum()
            df = self._handle_outliers(df, volume_column)

        # 8. Final missing check
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > self.max_missing_pct:
            logger.warning(
                f"High missing data percentage: {missing_pct:.2%} > {self.max_missing_pct:.2%}"
            )

        # 9. Drop any remaining NaN rows
        df = df.dropna()

        report = CleaningReport(
            original_rows=original_rows,
            final_rows=len(df),
            missing_filled=missing_filled,
            outliers_handled=outliers_handled,
            duplicates_removed=duplicates_removed,
            gaps_detected=gaps_detected,
        )

        return df, report

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on configured method."""
        if self.missing_method == MissingDataMethod.DROP:
            return df.dropna()
        elif self.missing_method == MissingDataMethod.FORWARD_FILL:
            return df.ffill()
        elif self.missing_method == MissingDataMethod.BACKWARD_FILL:
            return df.bfill()
        elif self.missing_method == MissingDataMethod.INTERPOLATE:
            return df.interpolate(method="linear")
        elif self.missing_method == MissingDataMethod.ZERO:
            return df.fillna(0)
        else:
            return df

    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using z-score method."""
        if series.std() == 0:
            return pd.Series(False, index=series.index)

        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > self.outlier_threshold

    def _handle_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Handle outliers in a specific column."""
        if column not in df.columns:
            return df

        series = df[column]
        outliers = self._detect_outliers(series)

        if not outliers.any():
            return df

        if self.outlier_method == OutlierMethod.CLIP:
            # Clip to mean ± threshold * std
            lower = series.mean() - self.outlier_threshold * series.std()
            upper = series.mean() + self.outlier_threshold * series.std()
            df[column] = series.clip(lower=lower, upper=upper)

        elif self.outlier_method == OutlierMethod.REMOVE:
            df = df[~outliers]

        elif self.outlier_method == OutlierMethod.WINSORIZE:
            # Replace with percentile values
            lower_pct = (1 - 0.99) / 2 * 100
            upper_pct = 100 - lower_pct
            lower = np.percentile(series, lower_pct)
            upper = np.percentile(series, upper_pct)
            df[column] = series.clip(lower=lower, upper=upper)

        return df

    def _detect_gaps(self, df: pd.DataFrame) -> int:
        """Detect time gaps in the data."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return 0

        time_diffs = df.index.to_series().diff()
        max_gap = pd.Timedelta(minutes=self.max_gap_minutes)

        gaps = time_diffs > max_gap
        num_gaps = gaps.sum()

        if num_gaps > 0:
            logger.debug(f"Detected {num_gaps} time gaps > {self.max_gap_minutes} minutes")

        return num_gaps

    def _validate_ohlc(
        self, df: pd.DataFrame, price_columns: list[str]
    ) -> pd.DataFrame:
        """
        Validate OHLC price relationships.

        Ensures:
        - High >= max(Open, Close)
        - Low <= min(Open, Close)
        - All prices > 0
        """
        if not all(col in df.columns for col in ["open", "high", "low", "close"]):
            return df

        # Ensure high >= open, close
        df["high"] = df[["high", "open", "close"]].max(axis=1)

        # Ensure low <= open, close
        df["low"] = df[["low", "open", "close"]].min(axis=1)

        # Ensure all prices > 0
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].clip(lower=0.0001)

        return df


def detect_stale_data(
    df: pd.DataFrame,
    columns: list[str] = None,
    max_unchanged_periods: int = 10,
) -> pd.Series:
    """
    Detect stale/unchanged data points.

    In real market data, prices rarely stay exactly the same
    for many consecutive periods.

    Args:
        df: DataFrame with price data
        columns: Columns to check (default: close)
        max_unchanged_periods: Max consecutive unchanged values

    Returns:
        Boolean Series marking potentially stale data
    """
    columns = columns or ["close"]
    stale_mask = pd.Series(False, index=df.index)

    for col in columns:
        if col not in df.columns:
            continue

        # Count consecutive unchanged values
        unchanged = df[col] == df[col].shift(1)
        streak = unchanged.groupby((~unchanged).cumsum()).cumsum()

        stale_mask |= streak >= max_unchanged_periods

    return stale_mask


def compute_data_quality_score(df: pd.DataFrame) -> dict:
    """
    Compute a data quality score.

    Returns metrics about data quality including:
    - Completeness (% non-null)
    - Consistency (OHLC relationships valid)
    - Timeliness (gaps in data)

    Args:
        df: DataFrame to assess

    Returns:
        Dictionary with quality metrics
    """
    metrics = {}

    # Completeness
    total_cells = len(df) * len(df.columns)
    non_null = df.count().sum()
    metrics["completeness"] = non_null / total_cells if total_cells > 0 else 0

    # Consistency (OHLC relationships)
    if all(col in df.columns for col in ["open", "high", "low", "close"]):
        high_valid = (df["high"] >= df[["open", "close"]].max(axis=1)).mean()
        low_valid = (df["low"] <= df[["open", "close"]].min(axis=1)).mean()
        metrics["ohlc_consistency"] = (high_valid + low_valid) / 2
    else:
        metrics["ohlc_consistency"] = None

    # Timeliness (check for gaps)
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        time_diffs = df.index.to_series().diff().dropna()
        median_diff = time_diffs.median()
        gaps = (time_diffs > 2 * median_diff).sum()
        metrics["gap_count"] = gaps
        metrics["timeliness"] = 1 - (gaps / len(df))
    else:
        metrics["gap_count"] = None
        metrics["timeliness"] = None

    # Overall score
    scores = [v for v in [metrics["completeness"], metrics.get("ohlc_consistency"), metrics.get("timeliness")] if v is not None]
    metrics["overall_score"] = sum(scores) / len(scores) if scores else 0

    return metrics
