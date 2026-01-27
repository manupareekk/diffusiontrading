"""
Robustness testing for trading strategies.

Tests how strategy performance degrades under various perturbations.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class RobustnessResult:
    """Results from robustness testing."""
    baseline_metrics: Dict[str, float]
    perturbed_metrics: pd.DataFrame
    degradation_scores: Dict[str, float]
    robustness_score: float  # 0-1, higher is more robust


class RobustnessAnalyzer:
    """
    Test strategy robustness to various perturbations.

    Perturbation types:
    - Noise injection: Add Gaussian noise to prices
    - Missing data: Random missing bars
    - Outliers: Inject extreme price movements
    - Regime shift: Change volatility or trend
    """

    def __init__(
        self,
        evaluation_func: Callable[[pd.DataFrame], Dict[str, float]],
        seed: int = 42,
    ):
        """
        Initialize the analyzer.

        Args:
            evaluation_func: Function that takes data and returns metrics
            seed: Random seed
        """
        self.evaluation_func = evaluation_func
        self.rng = np.random.default_rng(seed)

    def run_full_analysis(
        self,
        data: pd.DataFrame,
        noise_levels: List[float] = None,
        missing_rates: List[float] = None,
        n_trials: int = 10,
        primary_metric: str = "sharpe_ratio",
    ) -> RobustnessResult:
        """
        Run comprehensive robustness analysis.

        Args:
            data: Original data
            noise_levels: List of noise standard deviations to test
            missing_rates: List of missing data rates to test
            n_trials: Number of trials per condition
            primary_metric: Metric to use for robustness scoring

        Returns:
            RobustnessResult with all analysis
        """
        noise_levels = noise_levels or [0.001, 0.005, 0.01, 0.02]
        missing_rates = missing_rates or [0.01, 0.05, 0.10]

        # Get baseline
        logger.info("Computing baseline metrics...")
        baseline = self.evaluation_func(data)

        # Run perturbation tests
        results = []

        # Noise injection tests
        logger.info("Running noise injection tests...")
        for level in noise_levels:
            for trial in range(n_trials):
                perturbed = self._add_noise(data, level)
                try:
                    metrics = self.evaluation_func(perturbed)
                    results.append({
                        "perturbation": "noise",
                        "level": level,
                        "trial": trial,
                        **metrics
                    })
                except Exception as e:
                    logger.warning(f"Noise test failed: {e}")

        # Missing data tests
        logger.info("Running missing data tests...")
        for rate in missing_rates:
            for trial in range(n_trials):
                perturbed = self._add_missing(data, rate)
                try:
                    metrics = self.evaluation_func(perturbed)
                    results.append({
                        "perturbation": "missing",
                        "level": rate,
                        "trial": trial,
                        **metrics
                    })
                except Exception as e:
                    logger.warning(f"Missing data test failed: {e}")

        # Outlier tests
        logger.info("Running outlier tests...")
        for rate in [0.001, 0.005, 0.01]:
            for trial in range(n_trials):
                perturbed = self._add_outliers(data, rate)
                try:
                    metrics = self.evaluation_func(perturbed)
                    results.append({
                        "perturbation": "outliers",
                        "level": rate,
                        "trial": trial,
                        **metrics
                    })
                except Exception as e:
                    logger.warning(f"Outlier test failed: {e}")

        # Compile results
        results_df = pd.DataFrame(results)

        # Compute degradation scores
        degradation = self._compute_degradation(baseline, results_df, primary_metric)

        # Compute overall robustness score
        if primary_metric in results_df.columns and primary_metric in baseline:
            baseline_val = baseline[primary_metric]
            if baseline_val != 0:
                mean_perturbed = results_df[primary_metric].mean()
                robustness_score = max(0, min(1, mean_perturbed / baseline_val))
            else:
                robustness_score = 0.5
        else:
            robustness_score = 0.5

        return RobustnessResult(
            baseline_metrics=baseline,
            perturbed_metrics=results_df,
            degradation_scores=degradation,
            robustness_score=robustness_score,
        )

    def noise_injection_test(
        self,
        data: pd.DataFrame,
        noise_levels: List[float] = None,
        n_trials: int = 10,
    ) -> pd.DataFrame:
        """
        Test performance under different noise levels.

        Args:
            data: Original data
            noise_levels: Noise standard deviations as fraction of price
            n_trials: Number of trials per level

        Returns:
            DataFrame with metrics for each noise level
        """
        noise_levels = noise_levels or [0.001, 0.005, 0.01, 0.02, 0.05]
        results = []

        for level in noise_levels:
            for trial in range(n_trials):
                perturbed = self._add_noise(data, level)
                try:
                    metrics = self.evaluation_func(perturbed)
                    results.append({
                        "noise_level": level,
                        "trial": trial,
                        **metrics
                    })
                except Exception as e:
                    logger.warning(f"Trial failed at noise={level}: {e}")

        return pd.DataFrame(results)

    def _add_noise(
        self,
        data: pd.DataFrame,
        noise_std: float,
        price_cols: List[str] = None,
    ) -> pd.DataFrame:
        """
        Add Gaussian noise to price columns.

        Args:
            data: Original data
            noise_std: Noise standard deviation as fraction of price
            price_cols: Columns to add noise to

        Returns:
            Perturbed data
        """
        price_cols = price_cols or ["open", "high", "low", "close"]
        df = data.copy()

        for col in price_cols:
            if col in df.columns:
                noise = self.rng.normal(0, noise_std, len(df))
                df[col] = df[col] * (1 + noise)

        # Ensure OHLC consistency
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            df["high"] = df[["high", "open", "close"]].max(axis=1)
            df["low"] = df[["low", "open", "close"]].min(axis=1)

        return df

    def _add_missing(
        self,
        data: pd.DataFrame,
        missing_rate: float,
    ) -> pd.DataFrame:
        """
        Randomly remove bars from data.

        Args:
            data: Original data
            missing_rate: Fraction of bars to remove

        Returns:
            Data with missing bars
        """
        df = data.copy()
        n_missing = int(len(df) * missing_rate)

        if n_missing > 0:
            drop_indices = self.rng.choice(df.index, n_missing, replace=False)
            df = df.drop(drop_indices)

        return df

    def _add_outliers(
        self,
        data: pd.DataFrame,
        outlier_rate: float,
        outlier_magnitude: float = 5.0,
    ) -> pd.DataFrame:
        """
        Inject outliers (extreme moves) into price data.

        Args:
            data: Original data
            outlier_rate: Fraction of bars to make outliers
            outlier_magnitude: How many standard deviations for outliers

        Returns:
            Data with injected outliers
        """
        df = data.copy()
        n_outliers = int(len(df) * outlier_rate)

        if n_outliers > 0 and "close" in df.columns:
            returns = df["close"].pct_change()
            std = returns.std()

            outlier_indices = self.rng.choice(
                df.index[1:], n_outliers, replace=False
            )

            for idx in outlier_indices:
                direction = self.rng.choice([-1, 1])
                shock = direction * outlier_magnitude * std

                # Apply shock to close price
                df.loc[idx, "close"] = df.loc[idx, "close"] * (1 + shock)

                # Adjust high/low
                if "high" in df.columns:
                    df.loc[idx, "high"] = max(
                        df.loc[idx, "high"], df.loc[idx, "close"]
                    )
                if "low" in df.columns:
                    df.loc[idx, "low"] = min(
                        df.loc[idx, "low"], df.loc[idx, "close"]
                    )

        return df

    def _compute_degradation(
        self,
        baseline: Dict[str, float],
        results: pd.DataFrame,
        primary_metric: str,
    ) -> Dict[str, float]:
        """Compute degradation scores for each perturbation type."""
        if primary_metric not in baseline or primary_metric not in results.columns:
            return {}

        baseline_val = baseline[primary_metric]
        if baseline_val == 0:
            return {}

        degradation = {}
        for perturbation in results["perturbation"].unique():
            subset = results[results["perturbation"] == perturbation]
            mean_val = subset[primary_metric].mean()
            degradation[perturbation] = (baseline_val - mean_val) / abs(baseline_val)

        return degradation


def regime_analysis(
    data: pd.DataFrame,
    evaluation_func: Callable[[pd.DataFrame], Dict[str, float]],
    volatility_quantiles: List[float] = None,
) -> pd.DataFrame:
    """
    Analyze strategy performance across different market regimes.

    Args:
        data: Historical data
        evaluation_func: Function that evaluates strategy on data
        volatility_quantiles: Quantiles for regime classification

    Returns:
        DataFrame with metrics for each regime
    """
    volatility_quantiles = volatility_quantiles or [0.25, 0.5, 0.75]

    # Compute rolling volatility
    returns = data["close"].pct_change().dropna()
    rolling_vol = returns.rolling(window=20).std()

    # Classify regimes
    vol_quantiles = rolling_vol.quantile(volatility_quantiles)

    results = []

    # Low volatility
    low_vol_mask = rolling_vol <= vol_quantiles.iloc[0]
    if low_vol_mask.sum() > 50:
        low_vol_data = data.loc[low_vol_mask.index[low_vol_mask]]
        try:
            metrics = evaluation_func(low_vol_data)
            results.append({"regime": "low_volatility", **metrics})
        except Exception:
            pass

    # Medium volatility
    med_vol_mask = (rolling_vol > vol_quantiles.iloc[0]) & (rolling_vol <= vol_quantiles.iloc[1])
    if med_vol_mask.sum() > 50:
        med_vol_data = data.loc[med_vol_mask.index[med_vol_mask]]
        try:
            metrics = evaluation_func(med_vol_data)
            results.append({"regime": "medium_volatility", **metrics})
        except Exception:
            pass

    # High volatility
    high_vol_mask = rolling_vol > vol_quantiles.iloc[1]
    if high_vol_mask.sum() > 50:
        high_vol_data = data.loc[high_vol_mask.index[high_vol_mask]]
        try:
            metrics = evaluation_func(high_vol_data)
            results.append({"regime": "high_volatility", **metrics})
        except Exception:
            pass

    return pd.DataFrame(results)
