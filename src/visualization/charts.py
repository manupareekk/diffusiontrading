"""
Visualization module for trading system analysis.

Provides charting functions for:
- Equity curves and drawdowns
- Signal analysis
- Diffusion model predictions
- Sensitivity analysis heatmaps
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for this visualization")


def _check_plotly():
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for this visualization")


# ============================================================================
# Equity Curve and Performance Charts
# ============================================================================


def plot_equity_curve(
    equity: pd.Series,
    benchmark: pd.Series = None,
    title: str = "Equity Curve",
    figsize: Tuple[int, int] = (12, 6),
    show_drawdown: bool = True,
) -> Figure:
    """
    Plot equity curve with optional benchmark and drawdown.

    Args:
        equity: Series of portfolio values indexed by datetime
        benchmark: Optional benchmark equity curve
        title: Chart title
        figsize: Figure size
        show_drawdown: Whether to show drawdown subplot

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    if show_drawdown:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    # Plot equity curve
    ax1.plot(equity.index, equity.values, label="Strategy", linewidth=1.5, color="blue")

    if benchmark is not None:
        # Normalize benchmark to same starting value
        benchmark_normalized = benchmark / benchmark.iloc[0] * equity.iloc[0]
        ax1.plot(benchmark_normalized.index, benchmark_normalized.values,
                label="Benchmark", linewidth=1.5, color="gray", alpha=0.7)

    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Format y-axis with comma separator
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    if show_drawdown:
        # Calculate drawdown
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max * 100

        ax2.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values, color="red", linewidth=1)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    return fig


def plot_returns_distribution(
    returns: pd.Series,
    title: str = "Returns Distribution",
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 50,
) -> Figure:
    """
    Plot histogram of returns with statistics.

    Args:
        returns: Series of returns
        title: Chart title
        figsize: Figure size
        bins: Number of histogram bins

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot histogram
    ax.hist(returns.values, bins=bins, density=True, alpha=0.7, color="blue", edgecolor="black")

    # Add statistics
    mean = returns.mean()
    std = returns.std()
    skew = returns.skew()
    kurt = returns.kurtosis()

    # Plot normal distribution overlay
    x = np.linspace(returns.min(), returns.max(), 100)
    normal_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    ax.plot(x, normal_pdf, "r-", linewidth=2, label="Normal Distribution")

    # Add vertical lines for mean and std
    ax.axvline(mean, color="green", linestyle="--", linewidth=2, label=f"Mean: {mean:.4f}")
    ax.axvline(mean + std, color="orange", linestyle=":", linewidth=1.5)
    ax.axvline(mean - std, color="orange", linestyle=":", linewidth=1.5, label=f"±1σ: {std:.4f}")

    # Add statistics text box
    stats_text = f"Mean: {mean:.4f}\nStd: {std:.4f}\nSkew: {skew:.2f}\nKurtosis: {kurt:.2f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment="top",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_monthly_returns_heatmap(
    returns: pd.Series,
    title: str = "Monthly Returns (%)",
    figsize: Tuple[int, int] = (12, 8),
) -> Figure:
    """
    Plot heatmap of monthly returns by year.

    Args:
        returns: Series of daily returns indexed by datetime
        title: Chart title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    # Resample to monthly returns
    monthly_returns = (1 + returns).resample("ME").prod() - 1

    # Create pivot table
    monthly_df = pd.DataFrame({
        "Year": monthly_returns.index.year,
        "Month": monthly_returns.index.month,
        "Return": monthly_returns.values * 100
    })
    pivot = monthly_df.pivot(index="Year", columns="Month", values="Return")

    # Month names
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = month_names[:len(pivot.columns)]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Create heatmap
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Return (%)")

    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)

    # Add values to cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if not np.isnan(value):
                color = "white" if abs(value) > 5 else "black"
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=color, fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    plt.tight_layout()
    return fig


# ============================================================================
# Trade Analysis Charts
# ============================================================================


def plot_trade_analysis(
    trades: List[Dict],
    title: str = "Trade Analysis",
    figsize: Tuple[int, int] = (14, 10),
) -> Figure:
    """
    Plot comprehensive trade analysis.

    Args:
        trades: List of trade dictionaries with pnl, return_pct, etc.
        title: Chart title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    if not trades:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(0.5, 0.5, "No trades to display", ha="center", va="center", fontsize=14)
        return fig

    # Extract trade data
    pnls = [t.get("pnl", 0) for t in trades]
    returns = [t.get("return_pct", 0) for t in trades]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Cumulative P&L
    ax1 = axes[0, 0]
    cum_pnl = np.cumsum(pnls)
    ax1.plot(range(len(cum_pnl)), cum_pnl, linewidth=1.5, color="blue")
    ax1.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                     where=np.array(cum_pnl) >= 0, color="green", alpha=0.3)
    ax1.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                     where=np.array(cum_pnl) < 0, color="red", alpha=0.3)
    ax1.set_xlabel("Trade Number")
    ax1.set_ylabel("Cumulative P&L ($)")
    ax1.set_title("Cumulative P&L")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="black", linestyle="-", linewidth=0.5)

    # 2. P&L Distribution
    ax2 = axes[0, 1]
    colors = ["green" if p > 0 else "red" for p in pnls]
    ax2.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
    ax2.set_xlabel("Trade Number")
    ax2.set_ylabel("P&L ($)")
    ax2.set_title("Individual Trade P&L")
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    # 3. Return Distribution Histogram
    ax3 = axes[1, 0]
    returns_pct = [r * 100 for r in returns]
    ax3.hist(returns_pct, bins=30, color="blue", alpha=0.7, edgecolor="black")
    ax3.axvline(np.mean(returns_pct), color="red", linestyle="--",
                label=f"Mean: {np.mean(returns_pct):.2f}%")
    ax3.set_xlabel("Return (%)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Return Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Win/Loss Statistics
    ax4 = axes[1, 1]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    stats = [
        f"Total Trades: {len(trades)}",
        f"Winners: {len(wins)} ({len(wins)/len(trades)*100:.1f}%)",
        f"Losers: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)",
        f"",
        f"Avg Win: ${np.mean(wins):,.2f}" if wins else "Avg Win: N/A",
        f"Avg Loss: ${np.mean(losses):,.2f}" if losses else "Avg Loss: N/A",
        f"",
        f"Largest Win: ${max(pnls):,.2f}",
        f"Largest Loss: ${min(pnls):,.2f}",
        f"",
        f"Profit Factor: {abs(sum(wins)/sum(losses)):.2f}" if losses and sum(losses) != 0 else "Profit Factor: N/A",
    ]

    ax4.axis("off")
    ax4.text(0.1, 0.9, "\n".join(stats), transform=ax4.transAxes,
             verticalalignment="top", fontsize=12, family="monospace")
    ax4.set_title("Trade Statistics")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ============================================================================
# Signal Visualization
# ============================================================================


def plot_signals_with_price(
    price: pd.Series,
    signals: pd.DataFrame,
    signal_names: List[str] = None,
    title: str = "Price and Signals",
    figsize: Tuple[int, int] = (14, 10),
) -> Figure:
    """
    Plot price chart with signal indicators.

    Args:
        price: Series of prices
        signals: DataFrame of signal values
        signal_names: List of signal column names to plot
        title: Chart title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    if signal_names is None:
        signal_names = signals.columns[:4].tolist()  # Plot first 4 signals

    n_signals = len(signal_names)
    fig, axes = plt.subplots(n_signals + 1, 1, figsize=figsize, sharex=True)

    # Plot price
    axes[0].plot(price.index, price.values, linewidth=1, color="blue")
    axes[0].set_ylabel("Price")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    # Plot each signal
    colors = plt.cm.tab10.colors
    for i, signal_name in enumerate(signal_names):
        if signal_name in signals.columns:
            ax = axes[i + 1]
            signal_data = signals[signal_name]
            ax.plot(signal_data.index, signal_data.values, linewidth=1, color=colors[i % len(colors)])
            ax.set_ylabel(signal_name)
            ax.grid(True, alpha=0.3)

            # Add reference lines for common signals
            if "rsi" in signal_name.lower():
                ax.axhline(70, color="red", linestyle="--", alpha=0.5)
                ax.axhline(30, color="green", linestyle="--", alpha=0.5)
            elif "macd" in signal_name.lower():
                ax.axhline(0, color="black", linestyle="-", alpha=0.5)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    return fig


# ============================================================================
# Sensitivity Analysis Visualization
# ============================================================================


def plot_parameter_sensitivity_heatmap(
    results: pd.DataFrame,
    param1: str,
    param2: str,
    metric: str = "sharpe_ratio",
    title: str = None,
    figsize: Tuple[int, int] = (10, 8),
) -> Figure:
    """
    Plot 2D heatmap of parameter sensitivity.

    Args:
        results: DataFrame with parameter values and metrics
        param1: First parameter name (x-axis)
        param2: Second parameter name (y-axis)
        metric: Metric to plot
        title: Chart title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    # Create pivot table
    pivot = results.pivot_table(values=metric, index=param2, columns=param1, aggfunc="mean")

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Create heatmap
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric)

    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([f"{v:.2f}" if isinstance(v, float) else str(v) for v in pivot.columns])
    ax.set_yticklabels([f"{v:.2f}" if isinstance(v, float) else str(v) for v in pivot.index])

    # Add values to cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8)

    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_title(title or f"Sensitivity: {metric}")

    plt.tight_layout()
    return fig


def plot_monte_carlo_distribution(
    samples: np.ndarray,
    metric_name: str = "Total Return",
    confidence_level: float = 0.95,
    title: str = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """
    Plot Monte Carlo simulation distribution with confidence intervals.

    Args:
        samples: Array of simulated values
        metric_name: Name of the metric
        confidence_level: Confidence level for interval
        title: Chart title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot histogram
    ax.hist(samples, bins=50, density=True, alpha=0.7, color="blue", edgecolor="black")

    # Calculate statistics
    mean = np.mean(samples)
    median = np.median(samples)
    std = np.std(samples)
    alpha = (1 - confidence_level) / 2
    ci_lower = np.percentile(samples, alpha * 100)
    ci_upper = np.percentile(samples, (1 - alpha) * 100)

    # Add vertical lines
    ax.axvline(mean, color="red", linestyle="-", linewidth=2, label=f"Mean: {mean:.4f}")
    ax.axvline(median, color="green", linestyle="--", linewidth=2, label=f"Median: {median:.4f}")
    ax.axvline(ci_lower, color="orange", linestyle=":", linewidth=2)
    ax.axvline(ci_upper, color="orange", linestyle=":", linewidth=2,
               label=f"{confidence_level*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Shade confidence interval
    ax.axvspan(ci_lower, ci_upper, alpha=0.2, color="orange")

    ax.set_xlabel(metric_name)
    ax.set_ylabel("Density")
    ax.set_title(title or f"Monte Carlo Distribution: {metric_name}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_stress_test_results(
    results: Dict[str, Dict],
    metric: str = "total_return",
    title: str = "Stress Test Results",
    figsize: Tuple[int, int] = (12, 6),
) -> Figure:
    """
    Plot stress test scenario comparison.

    Args:
        results: Dict of scenario name -> MonteCarloResult
        metric: Metric to compare
        title: Chart title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    scenarios = list(results.keys())
    means = [results[s].mean if hasattr(results[s], "mean") else results[s]["mean"] for s in scenarios]
    stds = [results[s].std if hasattr(results[s], "std") else results[s]["std"] for s in scenarios]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    x = np.arange(len(scenarios))
    colors = ["green" if m > 0 else "red" for m in means]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


# ============================================================================
# Interactive Plotly Charts
# ============================================================================


def create_interactive_equity_chart(
    equity: pd.Series,
    trades: List[Dict] = None,
    title: str = "Interactive Equity Curve",
) -> "go.Figure":
    """
    Create interactive Plotly equity chart with trade markers.

    Args:
        equity: Series of portfolio values
        trades: Optional list of trade dictionaries
        title: Chart title

    Returns:
        Plotly Figure object
    """
    _check_plotly()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve", "Drawdown")
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity",
                   line=dict(color="blue", width=1.5)),
        row=1, col=1
    )

    # Add trade markers if provided
    if trades:
        for trade in trades:
            entry_time = trade.get("entry_time")
            exit_time = trade.get("exit_time")
            pnl = trade.get("pnl", 0)

            if entry_time in equity.index:
                color = "green" if pnl > 0 else "red"
                fig.add_trace(
                    go.Scatter(
                        x=[entry_time],
                        y=[equity.loc[entry_time]],
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=10, color=color),
                        name=f"Entry: ${pnl:,.0f}",
                        showlegend=False
                    ),
                    row=1, col=1
                )

    # Drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100

    fig.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown.values, mode="lines", name="Drawdown",
                   fill="tozeroy", line=dict(color="red", width=1)),
        row=2, col=1
    )

    fig.update_layout(
        title=title,
        height=600,
        showlegend=True,
        xaxis2_title="Date",
        yaxis_title="Portfolio Value ($)",
        yaxis2_title="Drawdown (%)",
    )

    return fig


# ============================================================================
# Report Generation
# ============================================================================


def generate_performance_report(
    equity: pd.Series,
    trades: List[Dict],
    metrics: Dict,
    output_path: str = None,
) -> str:
    """
    Generate a text-based performance report.

    Args:
        equity: Series of portfolio values
        trades: List of trade dictionaries
        metrics: Dictionary of performance metrics
        output_path: Optional path to save report

    Returns:
        Report string
    """
    lines = [
        "=" * 60,
        "TRADING STRATEGY PERFORMANCE REPORT",
        "=" * 60,
        "",
        "SUMMARY",
        "-" * 40,
    ]

    # Performance metrics
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key:.<30} {value:>12.4f}")
        else:
            lines.append(f"  {key:.<30} {value:>12}")

    lines.extend(["", "TRADE STATISTICS", "-" * 40])

    if trades:
        pnls = [t.get("pnl", 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        lines.extend([
            f"  Total Trades:................ {len(trades):>12}",
            f"  Winning Trades:.............. {len(wins):>12} ({len(wins)/len(trades)*100:.1f}%)",
            f"  Losing Trades:............... {len(losses):>12} ({len(losses)/len(trades)*100:.1f}%)",
            f"  Average Win:................. ${np.mean(wins) if wins else 0:>10,.2f}",
            f"  Average Loss:................ ${np.mean(losses) if losses else 0:>10,.2f}",
            f"  Largest Win:................. ${max(pnls):>10,.2f}",
            f"  Largest Loss:................ ${min(pnls):>10,.2f}",
        ])

        if losses and sum(losses) != 0:
            lines.append(f"  Profit Factor:............... {abs(sum(wins)/sum(losses)):>12.2f}")
    else:
        lines.append("  No trades executed")

    lines.extend(["", "EQUITY CURVE", "-" * 40])
    lines.extend([
        f"  Starting Value:.............. ${equity.iloc[0]:>10,.2f}",
        f"  Ending Value:................ ${equity.iloc[-1]:>10,.2f}",
        f"  Peak Value:.................. ${equity.max():>10,.2f}",
        f"  Trough Value:................ ${equity.min():>10,.2f}",
    ])

    lines.extend(["", "=" * 60])

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)

    return report
