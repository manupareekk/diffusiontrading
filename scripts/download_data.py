#!/usr/bin/env python3
"""
Download market data for backtesting.

Usage:
    python scripts/download_data.py --symbol SPY --interval 5m --days 60
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.providers.yfinance_provider import YFinanceProvider, DataRequest
from config.settings import settings


def main():
    parser = argparse.ArgumentParser(description="Download market data")
    parser.add_argument("--symbol", type=str, default="SPY", help="Ticker symbol")
    parser.add_argument("--symbols", type=str, nargs="+", help="Multiple symbols")
    parser.add_argument("--interval", type=str, default="5m", help="Data interval")
    parser.add_argument("--days", type=int, default=60, help="Number of days to download")
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()

    # Setup symbols
    symbols = args.symbols or [args.symbol]

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    # Create provider
    provider = YFinanceProvider(
        cache_dir=settings.data.cache_dir,
        cache_expiry_hours=settings.data.cache_expiry_hours,
        use_cache=settings.data.use_cache,
    )

    # Create request
    request = DataRequest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval=args.interval,
    )

    # Fetch data
    logger.info(f"Downloading {symbols} from {start_date.date()} to {end_date.date()}...")
    response = provider.fetch_ohlcv(request)

    if response.is_empty:
        logger.error("No data returned!")
        return 1

    logger.info(f"Downloaded {response.num_bars} bars")

    # Save to file
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = settings.data.cache_dir / f"{'_'.join(symbols)}_{args.interval}.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    response.data.to_parquet(output_path)
    logger.info(f"Saved to {output_path}")

    # Print summary
    print("\nData Summary:")
    print(f"  Symbols: {symbols}")
    print(f"  Interval: {args.interval}")
    print(f"  Date range: {response.data.index.min()} to {response.data.index.max()}")
    print(f"  Number of bars: {len(response.data)}")
    print(f"  Columns: {list(response.data.columns)}")

    return 0


if __name__ == "__main__":
    exit(main())
