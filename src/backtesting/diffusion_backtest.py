"""
Backtesting helpers for diffusion strategies.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.strategy.band_diffusion_strategy import BandBasedDiffusionStrategy
from src.strategy.diffusion_strategy import DiffusionPredictionStrategy
from src.data.features import ModelConditioner
from src.utils.device import get_device


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""
    initial_capital: float = 100000
    commission: float = 0.001
    slippage_bps: float = 5.0
    fast_mode: bool = True
    fast_mode_bars: int = 100


@dataclass
class BacktestResult:
    """Backtest results."""
    equity_curve: pd.Series
    trades: List[Dict]
    metrics: Dict[str, float]


def run_diffusion_backtest(
    data: pd.DataFrame,
    model: Any,
    model_config: Dict,
    strategy_type: str,
    config: BacktestConfig,
    band_type: str = "percentile",
    band_width: str = "percentile_5_95",
    num_samples: int = 100,
    progress_callback=None,
) -> BacktestResult:
    """
    Run backtest using diffusion model strategy.
    
    Args:
        data: Historical market data
        model: Trained diffusion model
        model_config: Model configuration dict
        strategy_type: "Diffusion - Probability" or "Diffusion - Bands"
        config: Backtest configuration
        band_type: "percentile" or "std" (for band strategy)
        band_width: Band width specification
        num_samples: Number of Monte Carlo samples
        progress_callback: Optional callback for progress updates
        
    Returns:
        BacktestResult with equity curve, trades, and metrics
    """
    # Select data range (Fast Mode or Full)
    backtest_data = data.copy()
    if config.fast_mode:
        backtest_data = data.iloc[-config.fast_mode_bars:].copy()
    
    # Get model config parameters
    window_size = model_config.get('window_size', 50)
    prediction_horizon = model_config.get('prediction_horizon', 5)
    
    # Setup device
    device = get_device("auto")
    
    # Log device to terminal and UI
    print(f"[BACKTEST] Using device: {device.upper()}")
    if progress_callback:
        progress_callback(0.0, f"Initialized on {device.upper()}")
    
    model.to(device)
    model.eval()
    
    # Create strategy
    if "Bands" in strategy_type:
        strategy = BandBasedDiffusionStrategy(
            model=model,
            window_size=window_size,
            device=device,
            prediction_horizon=prediction_horizon,
            band_type=band_type,
            band_width=band_width,
        )
    else:  # Probability-based
        strategy = DiffusionPredictionStrategy(
            model=model,
            window_size=window_size,
            device=device,
        )
        strategy.config.prediction_horizon = prediction_horizon
    
    # Configure num_samples
    strategy.config.num_samples = num_samples
    
    # Add RSI features
    conditioner = ModelConditioner()
    backtest_data = conditioner.add_features(backtest_data)
    
    # Initialize backtest state
    cash = config.initial_capital
    position = 0  # 0 = flat, 1 = long
    equity_history = []
    trades = []
    entry_price = 0
    entry_time = None
    shares = 0
    
    # Rolling window backtest
    start_idx = max(window_size, 0)
    total_steps = len(backtest_data) - start_idx
    
    for i in range(start_idx, len(backtest_data)):
        # Update progress
        if progress_callback and i % 10 == 0:
            progress = (i - start_idx + 1) / total_steps
            progress_callback(progress, f"Bar {i - start_idx + 1}/{total_steps}")
        
        # Get historical window
        window_data = backtest_data.iloc[max(0, i-window_size):i+1]
        
        if len(window_data) < window_size:
            continue
        
        price = backtest_data['close'].iloc[i]
        time = backtest_data.index[i]
        
        # Generate signal using strategy
        signals = strategy.generate_signals(window_data)
        
        if signals:
            signal = signals[0]
            signal_direction = signal.position.value  # 1=LONG, -1=SHORT, 0=FLAT
        else:
            signal_direction = 0
        
        # Execute signals
        if signal_direction == 1 and position == 0:  # Buy
            position = 1
            entry_price = price * (1 + config.slippage_bps/10000)
            shares = (cash * (1 - config.commission)) / entry_price
            cash = 0
            entry_time = time
            
        elif signal_direction == -1 and position == 1:  # Sell
            exit_price = price * (1 - config.slippage_bps/10000)
            cash = shares * exit_price * (1 - config.commission)
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': shares * (exit_price - entry_price),
                'return_pct': (exit_price - entry_price) / entry_price,
                'direction': 1
            })
            position = 0
            shares = 0
        
        # Calculate equity
        if position == 1:
            equity = shares * price
        else:
            equity = cash if cash > 0 else config.initial_capital
        equity_history.append((time, equity))
    
    # Build results
    equity_df = pd.DataFrame(equity_history, columns=['time', 'equity'])
    equity_df.set_index('time', inplace=True)
    equity_curve = equity_df['equity']
    
    # Calculate metrics
    returns = equity_curve.pct_change().dropna()
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1 if len(equity_curve) > 0 else 0
    
    # Sharpe ratio
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 78)  # 78 5-min bars/day
    else:
        sharpe = 0
    
    # Max drawdown
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
    
    # Win rate
    if trades:
        wins = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = wins / len(trades)
    else:
        win_rate = 0
    
    metrics = {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'total_trades': len(trades),
    }
    
    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        metrics=metrics
    )
