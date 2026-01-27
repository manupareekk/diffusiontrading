"""
Monte Carlo simulation for strategy performance estimation.

Uses bootstrap and synthetic data generation to estimate
confidence intervals and tail risks.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    mean: float
    std: float
    median: float
    percentile_5: float
    percentile_95: float
    samples: np.ndarray

    def confidence_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval at given level."""
        alpha = (1 - level) / 2
        lower = np.percentile(self.samples, alpha * 100)
        upper = np.percentile(self.samples, (1 - alpha) * 100)
        return (lower, upper)


@dataclass
class FullMonteCarloResult:
    """Full Monte Carlo simulation results."""
    metric_results: Dict[str, MonteCarloResult]
    return_distribution: np.ndarray
    max_drawdown_distribution: np.ndarray
    var_95: float
    cvar_95: float


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy analysis.

    Provides:
    - Bootstrap simulation for confidence intervals
    - Synthetic path generation
    - Tail risk analysis (VaR, CVaR)
    - Stress testing scenarios
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        seed: int = 42,
    ):
        """
        Initialize the simulator.

        Args:
            n_simulations: Number of Monte Carlo simulations
            seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)

    def bootstrap_returns(
        self,
        returns: np.ndarray,
        block_size: int = 20,
        horizon: int = None,
    ) -> MonteCarloResult:
        """
        Block bootstrap simulation of returns.

        Uses block bootstrap to preserve autocorrelation structure
        typical in financial returns.

        Args:
            returns: Array of historical returns
            block_size: Size of blocks for resampling
            horizon: Simulation horizon (default: same as input)

        Returns:
            MonteCarloResult with distribution of total returns
        """
        horizon = horizon or len(returns)
        n_blocks = (horizon + block_size - 1) // block_size

        simulated_total_returns = []

        for _ in range(self.n_simulations):
            # Sample block start indices
            max_start = len(returns) - block_size
            if max_start <= 0:
                # Fall back to regular bootstrap
                sampled_returns = self.rng.choice(returns, size=horizon, replace=True)
            else:
                block_starts = self.rng.integers(0, max_start + 1, size=n_blocks)

                # Construct simulated returns from blocks
                blocks = [returns[start:start + block_size] for start in block_starts]
                sampled_returns = np.concatenate(blocks)[:horizon]

            # Compute total return
            total_return = np.prod(1 + sampled_returns) - 1
            simulated_total_returns.append(total_return)

        samples = np.array(simulated_total_returns)

        return MonteCarloResult(
            mean=samples.mean(),
            std=samples.std(),
            median=np.median(samples),
            percentile_5=np.percentile(samples, 5),
            percentile_95=np.percentile(samples, 95),
            samples=samples,
        )

    def bootstrap_metric(
        self,
        data: np.ndarray,
        metric_func: Callable[[np.ndarray], float],
        block_size: int = 20,
    ) -> MonteCarloResult:
        """
        Bootstrap any metric function.

        Args:
            data: Input data array
            metric_func: Function that computes a single metric value
            block_size: Size of blocks for resampling

        Returns:
            MonteCarloResult for the metric
        """
        samples = []

        for _ in range(self.n_simulations):
            # Block bootstrap resample
            n_blocks = (len(data) + block_size - 1) // block_size
            max_start = max(0, len(data) - block_size)

            if max_start <= 0:
                resampled = self.rng.choice(data, size=len(data), replace=True)
            else:
                block_starts = self.rng.integers(0, max_start + 1, size=n_blocks)
                blocks = [data[start:start + block_size] for start in block_starts]
                resampled = np.concatenate(blocks)[:len(data)]

            try:
                value = metric_func(resampled)
                samples.append(value)
            except Exception:
                pass

        samples = np.array(samples)

        return MonteCarloResult(
            mean=samples.mean(),
            std=samples.std(),
            median=np.median(samples),
            percentile_5=np.percentile(samples, 5),
            percentile_95=np.percentile(samples, 95),
            samples=samples,
        )

    def run_full_simulation(
        self,
        returns: np.ndarray,
        block_size: int = 20,
    ) -> FullMonteCarloResult:
        """
        Run comprehensive Monte Carlo simulation.

        Args:
            returns: Array of historical returns
            block_size: Block size for bootstrap

        Returns:
            FullMonteCarloResult with all analyses
        """
        # Bootstrap total returns
        total_return_result = self.bootstrap_returns(returns, block_size)

        # Bootstrap Sharpe ratio
        def compute_sharpe(r):
            if r.std() == 0:
                return 0
            return r.mean() / r.std() * np.sqrt(252)

        sharpe_result = self.bootstrap_metric(returns, compute_sharpe, block_size)

        # Simulate drawdown distribution
        max_drawdowns = []
        for _ in range(self.n_simulations):
            # Generate simulated path
            n_blocks = (len(returns) + block_size - 1) // block_size
            max_start = max(0, len(returns) - block_size)

            if max_start <= 0:
                sim_returns = self.rng.choice(returns, size=len(returns), replace=True)
            else:
                block_starts = self.rng.integers(0, max_start + 1, size=n_blocks)
                blocks = [returns[start:start + block_size] for start in block_starts]
                sim_returns = np.concatenate(blocks)[:len(returns)]

            # Compute equity curve and max drawdown
            equity = np.cumprod(1 + sim_returns)
            running_max = np.maximum.accumulate(equity)
            drawdowns = (equity - running_max) / running_max
            max_drawdowns.append(drawdowns.min())

        max_dd_array = np.array(max_drawdowns)

        # Compute VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95

        return FullMonteCarloResult(
            metric_results={
                "total_return": total_return_result,
                "sharpe_ratio": sharpe_result,
            },
            return_distribution=total_return_result.samples,
            max_drawdown_distribution=max_dd_array,
            var_95=var_95,
            cvar_95=cvar_95,
        )

    def stress_test(
        self,
        returns: np.ndarray,
        scenarios: Dict[str, Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Dict[str, MonteCarloResult]:
        """
        Run stress test scenarios.

        Args:
            returns: Historical returns
            scenarios: Dict of scenario name -> transformation function

        Returns:
            Dict of scenario name -> MonteCarloResult
        """
        if scenarios is None:
            scenarios = {
                "2x_volatility": lambda r: r * 2,
                "3x_volatility": lambda r: r * 3,
                "negative_skew": lambda r: np.where(
                    r < 0, r * 1.5, r
                ),
                "fat_tails": lambda r: np.where(
                    np.abs(r) > r.std() * 2, r * 1.5, r
                ),
                "trend_reversal": lambda r: -r * 0.5 + r.mean(),
            }

        results = {}

        for name, transform in scenarios.items():
            try:
                stressed_returns = transform(returns.copy())
                result = self.bootstrap_returns(stressed_returns)
                results[name] = result
            except Exception as e:
                logger.warning(f"Stress scenario '{name}' failed: {e}")

        return results


def compute_probability_of_loss(
    returns: np.ndarray,
    n_simulations: int = 1000,
    horizon_days: int = 252,
    seed: int = 42,
) -> float:
    """
    Compute probability of negative total return.

    Args:
        returns: Historical daily returns
        n_simulations: Number of simulations
        horizon_days: Investment horizon in days
        seed: Random seed

    Returns:
        Probability of loss (0-1)
    """
    simulator = MonteCarloSimulator(n_simulations=n_simulations, seed=seed)
    result = simulator.bootstrap_returns(returns, block_size=20, horizon=horizon_days)
    return (result.samples < 0).mean()


def compute_expected_shortfall(
    returns: np.ndarray,
    confidence_level: float = 0.95,
) -> float:
    """
    Compute Expected Shortfall (CVaR).

    Expected Shortfall is the expected loss given that loss exceeds VaR.

    Args:
        returns: Historical returns
        confidence_level: Confidence level (e.g., 0.95)

    Returns:
        Expected shortfall (negative value indicating loss)
    """
    var = np.percentile(returns, (1 - confidence_level) * 100)
    tail_returns = returns[returns <= var]

    if len(tail_returns) == 0:
        return var

    return tail_returns.mean()
