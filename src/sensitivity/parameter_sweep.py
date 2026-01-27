"""
Parameter sweep for sensitivity analysis.

Systematically test how strategy performance varies with parameter changes.
"""

import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ParameterSpec:
    """
    Specification for a single parameter to sweep.
    """
    name: str
    values: List[Any] = None
    min_val: float = None
    max_val: float = None
    num_points: int = 10
    log_scale: bool = False
    param_type: str = "discrete"  # "discrete" or "continuous"

    def __post_init__(self):
        if self.values is not None:
            self.param_type = "discrete"
        elif self.min_val is not None and self.max_val is not None:
            self.param_type = "continuous"
            self._generate_values()
        else:
            raise ValueError("Must provide either 'values' or 'min_val' and 'max_val'")

    def _generate_values(self):
        """Generate values for continuous parameter."""
        if self.log_scale:
            self.values = np.geomspace(self.min_val, self.max_val, self.num_points).tolist()
        else:
            self.values = np.linspace(self.min_val, self.max_val, self.num_points).tolist()


@dataclass
class SweepResult:
    """
    Result of a parameter sweep.
    """
    parameters: pd.DataFrame
    metrics: pd.DataFrame
    best_params: Dict[str, Any]
    best_metrics: Dict[str, float]
    sensitivity_scores: Dict[str, float] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Combine parameters and metrics into single DataFrame."""
        return pd.concat([self.parameters, self.metrics], axis=1)


class ParameterSweep:
    """
    Systematic parameter sweep for sensitivity analysis.

    Supports:
    - Grid search: Test all combinations
    - Random search: Sample from parameter space
    - Sobol sequences: Quasi-random for better coverage
    """

    def __init__(
        self,
        parameter_specs: List[ParameterSpec],
        evaluation_func: Callable[[Dict[str, Any]], Dict[str, float]],
        search_type: str = "grid",
        n_samples: int = 100,
        n_jobs: int = 1,
        seed: int = 42,
    ):
        """
        Initialize the parameter sweep.

        Args:
            parameter_specs: List of ParameterSpec objects defining the search space
            evaluation_func: Function that takes params dict, returns metrics dict
            search_type: "grid", "random", or "sobol"
            n_samples: Number of samples for random/sobol search
            n_jobs: Number of parallel workers (-1 for all cores)
            seed: Random seed for reproducibility
        """
        self.parameter_specs = parameter_specs
        self.evaluation_func = evaluation_func
        self.search_type = search_type
        self.n_samples = n_samples
        self.n_jobs = n_jobs
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _generate_grid_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search."""
        param_names = [spec.name for spec in self.parameter_specs]
        param_values = [spec.values for spec in self.parameter_specs]

        combinations = []
        for combo in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combo)))

        return combinations

    def _generate_random_samples(self) -> List[Dict[str, Any]]:
        """Generate random samples from parameter space."""
        samples = []

        for _ in range(self.n_samples):
            sample = {}
            for spec in self.parameter_specs:
                idx = self.rng.integers(0, len(spec.values))
                sample[spec.name] = spec.values[idx]
            samples.append(sample)

        return samples

    def _generate_sobol_samples(self) -> List[Dict[str, Any]]:
        """Generate quasi-random Sobol sequence samples."""
        try:
            from scipy.stats.qmc import Sobol
        except ImportError:
            logger.warning("scipy.stats.qmc not available, falling back to random")
            return self._generate_random_samples()

        n_dims = len(self.parameter_specs)
        sobol = Sobol(d=n_dims, seed=self.seed)
        unit_samples = sobol.random(self.n_samples)

        samples = []
        for unit_sample in unit_samples:
            sample = {}
            for i, spec in enumerate(self.parameter_specs):
                # Map unit interval to parameter values
                idx = int(unit_sample[i] * len(spec.values))
                idx = min(idx, len(spec.values) - 1)
                sample[spec.name] = spec.values[idx]
            samples.append(sample)

        return samples

    def run(self, metric_to_optimize: str = "sharpe_ratio") -> SweepResult:
        """
        Execute the parameter sweep.

        Args:
            metric_to_optimize: Which metric to use for finding best params

        Returns:
            SweepResult with all results and analysis
        """
        # Generate parameter combinations
        if self.search_type == "grid":
            combinations = self._generate_grid_combinations()
        elif self.search_type == "random":
            combinations = self._generate_random_samples()
        elif self.search_type == "sobol":
            combinations = self._generate_sobol_samples()
        else:
            raise ValueError(f"Unknown search type: {self.search_type}")

        logger.info(f"Running {len(combinations)} parameter combinations...")

        # Execute evaluations
        results = []

        if self.n_jobs == 1:
            # Sequential execution
            for i, params in enumerate(combinations):
                try:
                    metrics = self.evaluation_func(params)
                    results.append({"params": params, "metrics": metrics})
                except Exception as e:
                    logger.warning(f"Evaluation failed for {params}: {e}")
                    results.append({"params": params, "metrics": {}})

                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(combinations)} evaluations")
        else:
            # Parallel execution
            n_workers = self.n_jobs if self.n_jobs > 0 else None
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(self.evaluation_func, params): params
                    for params in combinations
                }

                for i, future in enumerate(as_completed(futures)):
                    params = futures[future]
                    try:
                        metrics = future.result()
                        results.append({"params": params, "metrics": metrics})
                    except Exception as e:
                        logger.warning(f"Evaluation failed for {params}: {e}")
                        results.append({"params": params, "metrics": {}})

                    if (i + 1) % 10 == 0:
                        logger.info(f"Completed {i + 1}/{len(combinations)} evaluations")

        # Convert to DataFrames
        params_df = pd.DataFrame([r["params"] for r in results])
        metrics_df = pd.DataFrame([r["metrics"] for r in results])

        # Find best parameters
        if metric_to_optimize in metrics_df.columns:
            best_idx = metrics_df[metric_to_optimize].idxmax()
            best_params = results[best_idx]["params"]
            best_metrics = results[best_idx]["metrics"]
        else:
            best_params = {}
            best_metrics = {}

        # Compute sensitivity scores
        sensitivity_scores = self._compute_sensitivity_scores(
            params_df, metrics_df, metric_to_optimize
        )

        return SweepResult(
            parameters=params_df,
            metrics=metrics_df,
            best_params=best_params,
            best_metrics=best_metrics,
            sensitivity_scores=sensitivity_scores,
        )

    def _compute_sensitivity_scores(
        self,
        params_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        target_metric: str,
    ) -> Dict[str, float]:
        """
        Compute sensitivity scores for each parameter.

        Higher score = metric is more sensitive to this parameter.
        Uses correlation coefficient as a simple sensitivity measure.
        """
        if target_metric not in metrics_df.columns:
            return {}

        target = metrics_df[target_metric].dropna()
        scores = {}

        for param_name in params_df.columns:
            param_values = params_df.loc[target.index, param_name]

            # Convert to numeric if possible
            try:
                param_numeric = pd.to_numeric(param_values)
                corr = param_numeric.corr(target)
                scores[param_name] = abs(corr) if not np.isnan(corr) else 0.0
            except (ValueError, TypeError):
                # Non-numeric parameter, use variance of means
                means = target.groupby(param_values).mean()
                scores[param_name] = means.std() / (target.std() + 1e-8)

        return scores


def create_default_sweep(
    strategy_class,
    data: pd.DataFrame,
    param_ranges: Dict[str, List[Any]],
    n_jobs: int = 1,
) -> ParameterSweep:
    """
    Create a parameter sweep with sensible defaults.

    Args:
        strategy_class: Strategy class to instantiate
        data: Historical data for backtesting
        param_ranges: Dictionary mapping param names to value lists
        n_jobs: Number of parallel workers

    Returns:
        Configured ParameterSweep instance
    """
    from ..backtesting.engine import BacktestEngine, BacktestConfig

    # Create parameter specs
    specs = [
        ParameterSpec(name=name, values=values)
        for name, values in param_ranges.items()
    ]

    def evaluate(params: Dict[str, Any]) -> Dict[str, float]:
        """Evaluation function for the sweep."""
        try:
            strategy = strategy_class(**params)
            engine = BacktestEngine(strategy, BacktestConfig())
            result = engine.run(data)
            return result.metrics
        except Exception as e:
            return {"error": str(e)}

    return ParameterSweep(
        parameter_specs=specs,
        evaluation_func=evaluate,
        search_type="grid" if len(specs) <= 3 else "random",
        n_samples=100,
        n_jobs=n_jobs,
    )
