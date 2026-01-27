#!/usr/bin/env python3
"""
Run backtest with a trading strategy.

Usage:
    python scripts/run_backtest.py --data artifacts/data/raw/SPY_5m.parquet --strategy signal
"""

import argparse
from pathlib import Path

from loguru import logger

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--data", type=str, required=True, help="Path to data")
    parser.add_argument("--strategy", type=str, default="signal",
                       choices=["signal", "diffusion", "ensemble"],
                       help="Strategy type")
    parser.add_argument("--model", type=str, help="Path to trained diffusion model")
    parser.add_argument("--initial-capital", type=float, default=100000)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--output", type=str, help="Output file for results")
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.data}...")
    data = pd.read_parquet(args.data)
    logger.info(f"Loaded {len(data)} bars from {data.index.min()} to {data.index.max()}")

    # Import after path setup
    from src.backtesting.engine import BacktestEngine, BacktestConfig
    from src.strategy.base import BaseStrategy

    # Create strategy based on type
    if args.strategy == "signal":
        from src.strategy.diffusion_strategy import SignalBasedStrategy
        strategy = SignalBasedStrategy(
            rsi_period=9,
            rsi_oversold=30,
            rsi_overbought=70,
        )
    elif args.strategy == "diffusion":
        if not args.model:
            logger.error("--model required for diffusion strategy")
            return 1
        from src.strategy.diffusion_strategy import DiffusionPredictionStrategy
        import torch

        # Load model
        checkpoint = torch.load(args.model, map_location="cpu")
        # Create and load model (simplified - would need full reconstruction)
        logger.warning("Diffusion strategy requires model reconstruction - using placeholder")
        strategy = SignalBasedStrategy()  # Fallback
    else:  # ensemble
        from src.strategy.diffusion_strategy import SignalBasedStrategy
        strategy = SignalBasedStrategy(
            rsi_period=9,
            rsi_oversold=30,
            rsi_overbought=70,
        )

    # Create backtest config
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        commission_rate=args.commission,
        slippage_bps=5.0,
        slippage_model="percentage",
        fill_assumption="close",
        allow_shorting=True,
    )

    # Run backtest
    logger.info("Running backtest...")
    engine = BacktestEngine(strategy, config)
    result = engine.run(data)

    # Print results
    print("\n" + result.summary())

    # Print additional metrics
    print("\nDetailed Metrics:")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Print trade statistics
    if result.trades:
        print(f"\nTrade Statistics:")
        print(f"  Total trades: {len(result.trades)}")
        winning = [t for t in result.trades if t.pnl > 0]
        losing = [t for t in result.trades if t.pnl <= 0]
        print(f"  Winning trades: {len(winning)}")
        print(f"  Losing trades: {len(losing)}")
        if winning:
            print(f"  Average win: ${sum(t.pnl for t in winning) / len(winning):,.2f}")
        if losing:
            print(f"  Average loss: ${sum(t.pnl for t in losing) / len(losing):,.2f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save equity curve
        result.equity_curve.to_csv(output_path.with_suffix(".equity.csv"))

        # Save trades
        if result.trades:
            trades_df = pd.DataFrame([
                {
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl": t.pnl,
                    "return_pct": t.return_pct,
                }
                for t in result.trades
            ])
            trades_df.to_csv(output_path.with_suffix(".trades.csv"), index=False)

        logger.info(f"Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
