"""
Event-driven backtesting engine.

Simulates historical trading with realistic execution,
ensuring no lookahead bias.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..strategy.base import BaseStrategy, Position, Signal


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    symbol: str
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float
    commission: float
    slippage: float
    exit_reason: str = "signal"  # signal, stop_loss, take_profit, timeout


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_bps: float = 5.0  # 5 basis points
    slippage_model: str = "percentage"  # percentage, fixed
    fill_assumption: str = "close"  # open, close, vwap
    allow_shorting: bool = True
    margin_requirement: float = 0.5
    max_position_pct: float = 1.0  # Max % of capital per position


@dataclass
class BacktestResult:
    """Container for backtest results."""
    equity_curve: pd.Series
    returns: pd.Series
    trades: list[Trade]
    metrics: dict
    config: BacktestConfig
    signals: list[Signal] = field(default_factory=list)

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=" * 50,
            "BACKTEST SUMMARY",
            "=" * 50,
            f"Initial Capital: ${self.config.initial_capital:,.2f}",
            f"Final Equity: ${self.equity_curve.iloc[-1]:,.2f}",
            f"Total Return: {self.metrics['total_return']:.2%}",
            f"Annualized Return: {self.metrics.get('annualized_return', 0):.2%}",
            f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.3f}",
            f"Max Drawdown: {self.metrics['max_drawdown']:.2%}",
            f"Win Rate: {self.metrics['win_rate']:.2%}",
            f"Number of Trades: {len(self.trades)}",
            "=" * 50,
        ]
        return "\n".join(lines)


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Key features:
    - No lookahead bias (chronological iteration)
    - Realistic execution modeling (slippage, commissions)
    - Position tracking
    - Trade logging
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        config: BacktestConfig = None,
    ):
        """
        Initialize the backtesting engine.

        Args:
            strategy: Trading strategy to test
            config: Backtesting configuration
        """
        self.strategy = strategy
        self.config = config or BacktestConfig()

        # State
        self.cash = self.config.initial_capital
        self.positions: dict[str, dict] = {}  # symbol -> position info
        self.equity_history: list[tuple[pd.Timestamp, float]] = []
        self.trades: list[Trade] = []
        self.all_signals: list[Signal] = []

    def run(
        self,
        data: pd.DataFrame,
        symbol: str = "ASSET",
        start_idx: int = None,
        end_idx: int = None,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with OHLCV data
            symbol: Asset symbol
            start_idx: Starting index (default: after lookback)
            end_idx: Ending index (default: end of data)

        Returns:
            BacktestResult with performance metrics
        """
        # Reset state
        self._reset()

        # Determine start index
        lookback = self.strategy.get_required_lookback()
        if start_idx is None:
            start_idx = lookback
        if end_idx is None:
            end_idx = len(data)

        logger.info(f"Running backtest from index {start_idx} to {end_idx}")

        # Main loop - iterate chronologically
        for i in range(start_idx, end_idx):
            current_time = data.index[i]
            current_bar = data.iloc[i]

            # Get historical data up to current point (NO LOOKAHEAD)
            historical_data = data.iloc[:i + 1]

            # Check stop-loss and take-profit for existing positions
            self._check_exits(current_bar, current_time, symbol)

            # Generate signals from historical data only
            try:
                signals = self.strategy.generate_signals(historical_data, symbol)
                self.all_signals.extend(signals)
            except Exception as e:
                logger.debug(f"Signal generation failed at {current_time}: {e}")
                signals = []

            # Execute signals
            for signal in signals:
                self._execute_signal(signal, current_bar)

            # Update equity
            equity = self._calculate_equity(current_bar)
            self.equity_history.append((current_time, equity))

        # Close any remaining positions at end
        final_bar = data.iloc[-1]
        self._close_all_positions(final_bar, data.index[-1], "end_of_backtest")

        # Calculate results
        return self._generate_result()

    def _reset(self):
        """Reset engine state for new backtest."""
        self.cash = self.config.initial_capital
        self.positions = {}
        self.equity_history = []
        self.trades = []
        self.all_signals = []

    def _execute_signal(self, signal: Signal, current_bar: pd.Series):
        """Execute a trading signal."""
        symbol = signal.symbol

        # Get fill price based on assumption
        fill_price = self._get_fill_price(current_bar)

        # Apply slippage
        slippage = self._calculate_slippage(fill_price, signal.position)
        fill_price += slippage

        # Check if we have an existing position
        existing = self.positions.get(symbol)

        if signal.position == Position.FLAT:
            # Close position
            if existing:
                self._close_position(
                    symbol, fill_price, signal.timestamp, "signal"
                )
            return

        # Calculate position size
        equity = self._calculate_equity(current_bar)
        position_value = abs(signal.target_allocation) * equity * self.config.max_position_pct

        # Check if we need to close existing position first
        if existing:
            if (existing["direction"] == 1 and signal.position == Position.SHORT) or \
               (existing["direction"] == -1 and signal.position == Position.LONG):
                # Reverse position
                self._close_position(symbol, fill_price, signal.timestamp, "signal")

        # Open new position
        direction = 1 if signal.position == Position.LONG else -1

        if direction == -1 and not self.config.allow_shorting:
            return

        # Calculate quantity
        quantity = position_value / fill_price

        # Calculate commission
        commission = position_value * self.config.commission_rate

        # Check if we have enough cash
        required_cash = position_value + commission
        if direction == -1:
            required_cash = position_value * self.config.margin_requirement + commission

        if required_cash > self.cash:
            # Reduce position size to fit available capital
            available = self.cash - commission
            position_value = available
            quantity = position_value / fill_price

        if quantity <= 0:
            return

        # Execute
        self.cash -= commission
        if direction == 1:
            self.cash -= position_value
        else:
            # For shorts, margin is held
            self.cash -= position_value * self.config.margin_requirement

        self.positions[symbol] = {
            "direction": direction,
            "quantity": quantity,
            "entry_price": fill_price,
            "entry_time": signal.timestamp,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "commission": commission,
            "slippage": abs(slippage) * quantity,
        }

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_time: pd.Timestamp,
        exit_reason: str,
    ):
        """Close an existing position."""
        position = self.positions.get(symbol)
        if not position:
            return

        direction = position["direction"]
        quantity = position["quantity"]
        entry_price = position["entry_price"]

        # Calculate P&L
        if direction == 1:  # Long
            pnl = (exit_price - entry_price) * quantity
        else:  # Short
            pnl = (entry_price - exit_price) * quantity

        # Exit commission
        exit_value = exit_price * quantity
        exit_commission = exit_value * self.config.commission_rate
        pnl -= exit_commission

        # Update cash
        self.cash += exit_value + pnl
        if direction == -1:
            # Return margin
            self.cash += entry_price * quantity * self.config.margin_requirement

        # Record trade
        trade = Trade(
            entry_time=position["entry_time"],
            exit_time=exit_time,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            return_pct=(exit_price / entry_price - 1) * direction,
            commission=position["commission"] + exit_commission,
            slippage=position["slippage"],
            exit_reason=exit_reason,
        )
        self.trades.append(trade)

        # Remove position
        del self.positions[symbol]

    def _check_exits(
        self,
        current_bar: pd.Series,
        current_time: pd.Timestamp,
        symbol: str,
    ):
        """Check stop-loss and take-profit levels."""
        position = self.positions.get(symbol)
        if not position:
            return

        high = current_bar["high"]
        low = current_bar["low"]
        close = current_bar["close"]

        direction = position["direction"]
        stop_loss = position.get("stop_loss")
        take_profit = position.get("take_profit")

        # Check stop-loss
        if stop_loss:
            if direction == 1 and low <= stop_loss:
                self._close_position(symbol, stop_loss, current_time, "stop_loss")
                return
            elif direction == -1 and high >= stop_loss:
                self._close_position(symbol, stop_loss, current_time, "stop_loss")
                return

        # Check take-profit
        if take_profit:
            if direction == 1 and high >= take_profit:
                self._close_position(symbol, take_profit, current_time, "take_profit")
                return
            elif direction == -1 and low <= take_profit:
                self._close_position(symbol, take_profit, current_time, "take_profit")
                return

    def _close_all_positions(
        self,
        current_bar: pd.Series,
        current_time: pd.Timestamp,
        reason: str,
    ):
        """Close all open positions."""
        for symbol in list(self.positions.keys()):
            fill_price = self._get_fill_price(current_bar)
            self._close_position(symbol, fill_price, current_time, reason)

    def _get_fill_price(self, bar: pd.Series) -> float:
        """Get fill price based on configuration."""
        if self.config.fill_assumption == "open":
            return bar["open"]
        elif self.config.fill_assumption == "close":
            return bar["close"]
        elif self.config.fill_assumption == "vwap":
            if "vwap" in bar:
                return bar["vwap"]
            # Approximate VWAP
            return (bar["high"] + bar["low"] + bar["close"]) / 3
        return bar["close"]

    def _calculate_slippage(self, price: float, position: Position) -> float:
        """Calculate slippage."""
        slippage_pct = self.config.slippage_bps / 10000

        if self.config.slippage_model == "percentage":
            slippage = price * slippage_pct
        else:
            slippage = self.config.slippage_bps / 100  # Fixed amount

        # Slippage always works against us
        if position == Position.LONG:
            return slippage  # Pay more
        elif position == Position.SHORT:
            return -slippage  # Receive less
        return 0

    def _calculate_equity(self, current_bar: pd.Series) -> float:
        """Calculate current total equity."""
        equity = self.cash

        for symbol, position in self.positions.items():
            current_price = current_bar["close"]
            direction = position["direction"]
            quantity = position["quantity"]
            entry_price = position["entry_price"]

            if direction == 1:
                # Long: current value
                equity += current_price * quantity
            else:
                # Short: margin + unrealized P&L
                unrealized_pnl = (entry_price - current_price) * quantity
                equity += entry_price * quantity * self.config.margin_requirement + unrealized_pnl

        return equity

    def _generate_result(self) -> BacktestResult:
        """Generate backtest results."""
        # Create equity curve
        equity_df = pd.DataFrame(
            self.equity_history, columns=["timestamp", "equity"]
        )
        equity_df.set_index("timestamp", inplace=True)
        equity_curve = equity_df["equity"]

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        # Calculate metrics
        metrics = self._calculate_metrics(equity_curve, returns)

        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            trades=self.trades,
            metrics=metrics,
            config=self.config,
            signals=self.all_signals,
        )

    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        returns: pd.Series,
    ) -> dict:
        """Calculate performance metrics."""
        metrics = {}

        # Total return
        metrics["total_return"] = (
            equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        )

        # Annualized return (assuming 252 trading days)
        n_days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if n_days > 0:
            metrics["annualized_return"] = (
                (1 + metrics["total_return"]) ** (365 / n_days) - 1
            )
        else:
            metrics["annualized_return"] = 0

        # Volatility
        if len(returns) > 1:
            metrics["volatility"] = returns.std() * np.sqrt(252)
        else:
            metrics["volatility"] = 0

        # Sharpe ratio
        if metrics["volatility"] > 0:
            metrics["sharpe_ratio"] = (
                metrics["annualized_return"] / metrics["volatility"]
            )
        else:
            metrics["sharpe_ratio"] = 0

        # Max drawdown
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        metrics["max_drawdown"] = abs(drawdowns.min())

        # Trade statistics
        if self.trades:
            pnls = [t.pnl for t in self.trades]
            metrics["win_rate"] = sum(1 for p in pnls if p > 0) / len(pnls)
            metrics["avg_trade_pnl"] = np.mean(pnls)
            metrics["total_commission"] = sum(t.commission for t in self.trades)

            winning = [p for p in pnls if p > 0]
            losing = [p for p in pnls if p < 0]

            metrics["avg_win"] = np.mean(winning) if winning else 0
            metrics["avg_loss"] = np.mean(losing) if losing else 0

            if losing and metrics["avg_loss"] != 0:
                metrics["profit_factor"] = (
                    abs(sum(winning) / sum(losing))
                )
            else:
                metrics["profit_factor"] = float("inf") if winning else 0
        else:
            metrics["win_rate"] = 0
            metrics["avg_trade_pnl"] = 0
            metrics["total_commission"] = 0
            metrics["profit_factor"] = 0

        metrics["num_trades"] = len(self.trades)

        return metrics
