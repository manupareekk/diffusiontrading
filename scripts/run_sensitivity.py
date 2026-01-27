#!/usr/bin/env python3
"""
Run sensitivity analysis on trading strategy.

Usage:
    python scripts/run_sensitivity.py --data artifacts/data/raw/SPY_5m.parquet --type parameter_sweep
    python scripts/run_sensitivity.py --data artifacts/data/raw/SPY_5m.parquet --type robustness
    python scripts/run_sensitivity.py --data artifacts/data/raw/SPY_5m.parquet --type monte_carlo
"""

import argparse
from pathlib import Path

from loguru import logger

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


def run_parameter_sweep(data: pd.DataFrame, output_dir: Path):
    """Run parameter sweep analysis."""
    from src.sensitivity.parameter_sweep import ParameterSweep, ParameterSpec
    from src.backtesting.engine import BacktestEngine, BacktestConfig
    from src.strategy.diffusion_strategy import SignalBasedStrategy

    logger.info("Running parameter sweep analysis...")

    # Define parameter grid
    params = [
        ParameterSpec("rsi_period", [5, 7, 9, 11, 14]),
        ParameterSpec("rsi_oversold", [25, 30, 35]),
        ParameterSpec("rsi_overbought", [65, 70, 75]),
    ]

    # Define evaluation function
    def evaluate_strategy(param_dict: dict) -> dict:
        strategy = SignalBasedStrategy(
            rsi_period=param_dict["rsi_period"],
            rsi_oversold=param_dict["rsi_oversold"],
            rsi_overbought=param_dict["rsi_overbought"],
        )

        config = BacktestConfig(
            initial_capital=100000,
            commission_rate=0.001,
            slippage_bps=5.0,
        )

        engine = BacktestEngine(strategy, config)
        result = engine.run(data)

        return result.metrics

    # Run sweep
    sweep = ParameterSweep(params)
    results = sweep.run_sweep(evaluate_strategy, method="grid", n_jobs=1)

    # Save results
    output_file = output_dir / "parameter_sweep_results.csv"
    results.to_dataframe().to_csv(output_file, index=False)
    logger.info(f"Parameter sweep results saved to {output_file}")

    # Print best configuration
    best_idx = np.argmax([r["sharpe_ratio"] for r in results.results if "sharpe_ratio" in r])
    best_params = results.parameter_sets[best_idx]
    best_metrics = results.results[best_idx]

    print("\nBest Configuration:")
    print("-" * 40)
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\nSharpe Ratio: {best_metrics.get('sharpe_ratio', 'N/A'):.4f}")
    print(f"Total Return: {best_metrics.get('total_return', 'N/A'):.4f}")

    return results


def run_robustness_analysis(data: pd.DataFrame, output_dir: Path):
    """Run robustness analysis."""
    from src.sensitivity.robustness import RobustnessAnalyzer
    from src.backtesting.engine import BacktestEngine, BacktestConfig
    from src.strategy.diffusion_strategy import SignalBasedStrategy

    logger.info("Running robustness analysis...")

    strategy = SignalBasedStrategy(
        rsi_period=9,
        rsi_oversold=30,
        rsi_overbought=70,
    )

    config = BacktestConfig(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_bps=5.0,
    )

    # Define evaluation function that returns a metric
    def evaluate_fn(perturbed_data: pd.DataFrame) -> float:
        engine = BacktestEngine(strategy, config)
        result = engine.run(perturbed_data)
        return result.metrics.get("sharpe_ratio", 0.0)

    analyzer = RobustnessAnalyzer(seed=42)

    # Run noise injection test
    print("\n" + "=" * 50)
    print("NOISE INJECTION TEST")
    print("=" * 50)

    noise_results = analyzer.noise_injection_test(
        data,
        evaluate_fn,
        noise_levels=[0.0, 0.001, 0.002, 0.005, 0.01],
        n_trials=5,
    )

    for noise_level, metrics in noise_results.items():
        print(f"\nNoise Level: {noise_level}")
        print(f"  Mean Sharpe: {metrics['mean']:.4f}")
        print(f"  Std Sharpe:  {metrics['std']:.4f}")
        print(f"  Degradation: {metrics['degradation_pct']:.2f}%")

    # Run missing data test
    print("\n" + "=" * 50)
    print("MISSING DATA TEST")
    print("=" * 50)

    missing_results = analyzer.missing_data_test(
        data,
        evaluate_fn,
        missing_rates=[0.0, 0.01, 0.02, 0.05],
        n_trials=5,
    )

    for missing_rate, metrics in missing_results.items():
        print(f"\nMissing Rate: {missing_rate}")
        print(f"  Mean Sharpe: {metrics['mean']:.4f}")
        print(f"  Std Sharpe:  {metrics['std']:.4f}")
        print(f"  Degradation: {metrics['degradation_pct']:.2f}%")

    # Save summary
    summary_file = output_dir / "robustness_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Robustness Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write("Noise Injection Test:\n")
        for noise_level, metrics in noise_results.items():
            f.write(f"  {noise_level}: Sharpe={metrics['mean']:.4f} ± {metrics['std']:.4f}\n")
        f.write("\nMissing Data Test:\n")
        for missing_rate, metrics in missing_results.items():
            f.write(f"  {missing_rate}: Sharpe={metrics['mean']:.4f} ± {metrics['std']:.4f}\n")

    logger.info(f"Robustness summary saved to {summary_file}")

    return {"noise": noise_results, "missing": missing_results}


def run_monte_carlo_analysis(data: pd.DataFrame, output_dir: Path):
    """Run Monte Carlo simulation."""
    from src.sensitivity.monte_carlo import (
        MonteCarloSimulator,
        compute_probability_of_loss,
        compute_expected_shortfall,
    )
    from src.backtesting.engine import BacktestEngine, BacktestConfig
    from src.strategy.diffusion_strategy import SignalBasedStrategy

    logger.info("Running Monte Carlo simulation...")

    # First run a backtest to get returns
    strategy = SignalBasedStrategy(
        rsi_period=9,
        rsi_oversold=30,
        rsi_overbought=70,
    )

    config = BacktestConfig(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_bps=5.0,
    )

    engine = BacktestEngine(strategy, config)
    result = engine.run(data)

    # Get strategy returns from equity curve
    equity = result.equity_curve
    returns = equity.pct_change().dropna().values

    # Run Monte Carlo simulation
    simulator = MonteCarloSimulator(n_simulations=1000, seed=42)
    mc_result = simulator.run_full_simulation(returns)

    print("\n" + "=" * 50)
    print("MONTE CARLO SIMULATION RESULTS")
    print("=" * 50)

    print("\nTotal Return Distribution:")
    tr = mc_result.metric_results["total_return"]
    print(f"  Mean: {tr.mean:.4f}")
    print(f"  Median: {tr.median:.4f}")
    print(f"  Std: {tr.std:.4f}")
    print(f"  5th percentile: {tr.percentile_5:.4f}")
    print(f"  95th percentile: {tr.percentile_95:.4f}")

    print("\nSharpe Ratio Distribution:")
    sr = mc_result.metric_results["sharpe_ratio"]
    print(f"  Mean: {sr.mean:.4f}")
    print(f"  Median: {sr.median:.4f}")
    print(f"  Std: {sr.std:.4f}")
    print(f"  5th percentile: {sr.percentile_5:.4f}")
    print(f"  95th percentile: {sr.percentile_95:.4f}")

    print("\nMax Drawdown Distribution:")
    mdd = mc_result.max_drawdown_distribution
    print(f"  Mean: {np.mean(mdd):.4f}")
    print(f"  Worst 5%: {np.percentile(mdd, 5):.4f}")

    print("\nRisk Metrics:")
    print(f"  VaR (95%): {mc_result.var_95:.4f}")
    print(f"  CVaR (95%): {mc_result.cvar_95:.4f}")

    # Probability of loss
    prob_loss = compute_probability_of_loss(returns, n_simulations=1000, horizon_days=252)
    print(f"  Probability of Loss (1 year): {prob_loss:.2%}")

    # Run stress tests
    print("\n" + "=" * 50)
    print("STRESS TEST SCENARIOS")
    print("=" * 50)

    stress_results = simulator.stress_test(returns)
    for scenario, result in stress_results.items():
        print(f"\n{scenario}:")
        print(f"  Expected Return: {result.mean:.4f}")
        print(f"  95% CI: [{result.percentile_5:.4f}, {result.percentile_95:.4f}]")

    # Save results
    results_file = output_dir / "monte_carlo_results.csv"
    results_df = pd.DataFrame({
        "metric": ["total_return", "sharpe_ratio", "var_95", "cvar_95", "prob_loss"],
        "value": [tr.mean, sr.mean, mc_result.var_95, mc_result.cvar_95, prob_loss],
    })
    results_df.to_csv(results_file, index=False)
    logger.info(f"Monte Carlo results saved to {results_file}")

    # Save return distribution
    dist_file = output_dir / "return_distribution.csv"
    pd.DataFrame({"simulated_return": mc_result.return_distribution}).to_csv(dist_file, index=False)

    return mc_result


def main():
    parser = argparse.ArgumentParser(description="Run sensitivity analysis")
    parser.add_argument("--data", type=str, required=True, help="Path to data")
    parser.add_argument("--type", type=str, default="all",
                       choices=["parameter_sweep", "robustness", "monte_carlo", "all"],
                       help="Type of analysis")
    parser.add_argument("--output", type=str, default="artifacts/sensitivity",
                       help="Output directory")
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.data}...")
    data = pd.read_parquet(args.data)
    logger.info(f"Loaded {len(data)} bars")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analyses
    if args.type == "parameter_sweep" or args.type == "all":
        run_parameter_sweep(data, output_dir)

    if args.type == "robustness" or args.type == "all":
        run_robustness_analysis(data, output_dir)

    if args.type == "monte_carlo" or args.type == "all":
        run_monte_carlo_analysis(data, output_dir)

    print("\n" + "=" * 50)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"Results saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
