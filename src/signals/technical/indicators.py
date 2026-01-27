"""
Technical indicator implementations using pandas-ta.

All indicators are registered with the SignalRegistry for
unified access and computation.
"""

from typing import Optional

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None

from ..registry import SignalMetadata, SignalRegistry


def _ensure_pandas_ta():
    """Ensure pandas-ta is available."""
    if ta is None:
        raise ImportError(
            "pandas-ta is required for technical indicators. "
            "Install with: pip install pandas-ta"
        )


# =============================================================================
# Momentum Indicators
# =============================================================================


@SignalRegistry.register(
    SignalMetadata(
        name="rsi",
        category="momentum",
        lookback_period=14,
        output_columns=["rsi"],
        description="Relative Strength Index - momentum oscillator measuring speed and magnitude of price changes",
        default_params={"period": 14},
    )
)
def compute_rsi(data: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute Relative Strength Index (RSI).

    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.

    Args:
        data: DataFrame with 'close' column
        params: {'period': 14}

    Returns:
        DataFrame with 'rsi' column (0-100 range)
    """
    _ensure_pandas_ta()
    params = params or {}
    period = params.get("period", 14)

    rsi = ta.rsi(data["close"], length=period)
    return pd.DataFrame({"rsi": rsi}, index=data.index)


@SignalRegistry.register(
    SignalMetadata(
        name="macd",
        category="momentum",
        lookback_period=26,
        output_columns=["macd", "macd_signal", "macd_hist"],
        description="Moving Average Convergence Divergence - trend-following momentum indicator",
        default_params={"fast": 12, "slow": 26, "signal": 9},
    )
)
def compute_macd(data: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute MACD (Moving Average Convergence Divergence).

    MACD shows the relationship between two moving averages of a price.

    Args:
        data: DataFrame with 'close' column
        params: {'fast': 12, 'slow': 26, 'signal': 9}

    Returns:
        DataFrame with 'macd', 'macd_signal', 'macd_hist' columns
    """
    _ensure_pandas_ta()
    params = params or {}
    fast = params.get("fast", 12)
    slow = params.get("slow", 26)
    signal = params.get("signal", 9)

    macd_df = ta.macd(data["close"], fast=fast, slow=slow, signal=signal)

    # pandas-ta returns columns like MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    # Rename to standard names
    result = pd.DataFrame(index=data.index)
    result["macd"] = macd_df.iloc[:, 0]
    result["macd_hist"] = macd_df.iloc[:, 1]
    result["macd_signal"] = macd_df.iloc[:, 2]

    return result


@SignalRegistry.register(
    SignalMetadata(
        name="stochastic",
        category="momentum",
        lookback_period=14,
        output_columns=["stoch_k", "stoch_d"],
        description="Stochastic Oscillator - momentum indicator comparing closing price to price range",
        default_params={"k_period": 14, "d_period": 3, "smooth_k": 3},
    )
)
def compute_stochastic(data: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute Stochastic Oscillator.

    Compares a closing price to its price range over a period.

    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        params: {'k_period': 14, 'd_period': 3, 'smooth_k': 3}

    Returns:
        DataFrame with 'stoch_k', 'stoch_d' columns (0-100 range)
    """
    _ensure_pandas_ta()
    params = params or {}
    k_period = params.get("k_period", 14)
    d_period = params.get("d_period", 3)
    smooth_k = params.get("smooth_k", 3)

    stoch = ta.stoch(
        data["high"], data["low"], data["close"],
        k=k_period, d=d_period, smooth_k=smooth_k
    )

    result = pd.DataFrame(index=data.index)
    result["stoch_k"] = stoch.iloc[:, 0]
    result["stoch_d"] = stoch.iloc[:, 1]

    return result


@SignalRegistry.register(
    SignalMetadata(
        name="roc",
        category="momentum",
        lookback_period=10,
        output_columns=["roc"],
        description="Rate of Change - measures percentage change in price over a period",
        default_params={"period": 10},
    )
)
def compute_roc(data: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute Rate of Change (ROC).

    Measures the percentage change between current price and price n periods ago.

    Args:
        data: DataFrame with 'close' column
        params: {'period': 10}

    Returns:
        DataFrame with 'roc' column (percentage)
    """
    _ensure_pandas_ta()
    params = params or {}
    period = params.get("period", 10)

    roc = ta.roc(data["close"], length=period)
    return pd.DataFrame({"roc": roc}, index=data.index)


# =============================================================================
# Trend Indicators
# =============================================================================


@SignalRegistry.register(
    SignalMetadata(
        name="sma",
        category="trend",
        lookback_period=20,
        output_columns=["sma"],
        description="Simple Moving Average - average price over a period",
        default_params={"period": 20},
    )
)
def compute_sma(data: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute Simple Moving Average (SMA).

    Args:
        data: DataFrame with 'close' column
        params: {'period': 20}

    Returns:
        DataFrame with 'sma' column
    """
    _ensure_pandas_ta()
    params = params or {}
    period = params.get("period", 20)

    sma = ta.sma(data["close"], length=period)
    return pd.DataFrame({"sma": sma}, index=data.index)


@SignalRegistry.register(
    SignalMetadata(
        name="ema",
        category="trend",
        lookback_period=20,
        output_columns=["ema"],
        description="Exponential Moving Average - weighted average giving more weight to recent prices",
        default_params={"span": 20},
    )
)
def compute_ema(data: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute Exponential Moving Average (EMA).

    Args:
        data: DataFrame with 'close' column
        params: {'span': 20}

    Returns:
        DataFrame with 'ema' column
    """
    _ensure_pandas_ta()
    params = params or {}
    span = params.get("span", 20)

    ema = ta.ema(data["close"], length=span)
    return pd.DataFrame({"ema": ema}, index=data.index)


@SignalRegistry.register(
    SignalMetadata(
        name="adx",
        category="trend",
        lookback_period=14,
        output_columns=["adx", "dmp", "dmn"],
        description="Average Directional Index - measures trend strength",
        default_params={"period": 14},
    )
)
def compute_adx(data: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute Average Directional Index (ADX).

    Measures trend strength regardless of direction.

    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        params: {'period': 14}

    Returns:
        DataFrame with 'adx', 'dmp' (DI+), 'dmn' (DI-) columns
    """
    _ensure_pandas_ta()
    params = params or {}
    period = params.get("period", 14)

    adx_df = ta.adx(data["high"], data["low"], data["close"], length=period)

    result = pd.DataFrame(index=data.index)
    result["adx"] = adx_df.iloc[:, 0]
    result["dmp"] = adx_df.iloc[:, 1]
    result["dmn"] = adx_df.iloc[:, 2]

    return result


# =============================================================================
# Volatility Indicators
# =============================================================================


@SignalRegistry.register(
    SignalMetadata(
        name="bollinger",
        category="volatility",
        lookback_period=20,
        output_columns=["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct"],
        description="Bollinger Bands - volatility bands around a moving average",
        default_params={"period": 20, "std": 2.0},
    )
)
def compute_bollinger(data: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute Bollinger Bands.

    Consists of a middle band (SMA) and upper/lower bands based on standard deviation.

    Args:
        data: DataFrame with 'close' column
        params: {'period': 20, 'std': 2.0}

    Returns:
        DataFrame with band values and derived indicators
    """
    _ensure_pandas_ta()
    params = params or {}
    period = params.get("period", 20)
    std = params.get("std", 2.0)

    bb = ta.bbands(data["close"], length=period, std=std)

    result = pd.DataFrame(index=data.index)
    result["bb_lower"] = bb.iloc[:, 0]
    result["bb_middle"] = bb.iloc[:, 1]
    result["bb_upper"] = bb.iloc[:, 2]
    result["bb_width"] = bb.iloc[:, 3]
    result["bb_pct"] = bb.iloc[:, 4]

    return result


@SignalRegistry.register(
    SignalMetadata(
        name="atr",
        category="volatility",
        lookback_period=14,
        output_columns=["atr"],
        description="Average True Range - measures volatility",
        default_params={"period": 14},
    )
)
def compute_atr(data: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute Average True Range (ATR).

    Measures market volatility by decomposing the range of an asset price.

    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        params: {'period': 14}

    Returns:
        DataFrame with 'atr' column
    """
    _ensure_pandas_ta()
    params = params or {}
    period = params.get("period", 14)

    atr = ta.atr(data["high"], data["low"], data["close"], length=period)
    return pd.DataFrame({"atr": atr}, index=data.index)


@SignalRegistry.register(
    SignalMetadata(
        name="keltner",
        category="volatility",
        lookback_period=20,
        output_columns=["kc_upper", "kc_middle", "kc_lower"],
        description="Keltner Channels - volatility-based bands using ATR",
        default_params={"period": 20, "atr_period": 10, "multiplier": 2.0},
    )
)
def compute_keltner(data: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute Keltner Channels.

    Similar to Bollinger Bands but uses ATR instead of standard deviation.

    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        params: {'period': 20, 'atr_period': 10, 'multiplier': 2.0}

    Returns:
        DataFrame with channel values
    """
    _ensure_pandas_ta()
    params = params or {}
    period = params.get("period", 20)
    atr_period = params.get("atr_period", 10)
    multiplier = params.get("multiplier", 2.0)

    kc = ta.kc(
        data["high"], data["low"], data["close"],
        length=period, scalar=multiplier, mamode="ema"
    )

    result = pd.DataFrame(index=data.index)
    result["kc_lower"] = kc.iloc[:, 0]
    result["kc_middle"] = kc.iloc[:, 1]
    result["kc_upper"] = kc.iloc[:, 2]

    return result


# =============================================================================
# Volume Indicators
# =============================================================================


@SignalRegistry.register(
    SignalMetadata(
        name="vwap",
        category="volume",
        lookback_period=1,
        output_columns=["vwap"],
        description="Volume Weighted Average Price - key intraday benchmark",
        default_params={},
    )
)
def compute_vwap(data: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute Volume Weighted Average Price (VWAP).

    Critical benchmark for intraday trading.

    Args:
        data: DataFrame with 'high', 'low', 'close', 'volume' columns
        params: {}

    Returns:
        DataFrame with 'vwap' column
    """
    _ensure_pandas_ta()

    vwap = ta.vwap(data["high"], data["low"], data["close"], data["volume"])
    return pd.DataFrame({"vwap": vwap}, index=data.index)


@SignalRegistry.register(
    SignalMetadata(
        name="obv",
        category="volume",
        lookback_period=1,
        output_columns=["obv"],
        description="On Balance Volume - cumulative volume flow indicator",
        default_params={},
    )
)
def compute_obv(data: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute On Balance Volume (OBV).

    Relates volume to price change to measure buying/selling pressure.

    Args:
        data: DataFrame with 'close', 'volume' columns
        params: {}

    Returns:
        DataFrame with 'obv' column
    """
    _ensure_pandas_ta()

    obv = ta.obv(data["close"], data["volume"])
    return pd.DataFrame({"obv": obv}, index=data.index)


@SignalRegistry.register(
    SignalMetadata(
        name="mfi",
        category="volume",
        lookback_period=14,
        output_columns=["mfi"],
        description="Money Flow Index - volume-weighted RSI",
        default_params={"period": 14},
    )
)
def compute_mfi(data: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute Money Flow Index (MFI).

    A volume-weighted version of RSI, incorporating both price and volume.

    Args:
        data: DataFrame with 'high', 'low', 'close', 'volume' columns
        params: {'period': 14}

    Returns:
        DataFrame with 'mfi' column (0-100 range)
    """
    _ensure_pandas_ta()
    params = params or {}
    period = params.get("period", 14)

    mfi = ta.mfi(data["high"], data["low"], data["close"], data["volume"], length=period)
    return pd.DataFrame({"mfi": mfi}, index=data.index)


# =============================================================================
# Composite / Derived Indicators
# =============================================================================


@SignalRegistry.register(
    SignalMetadata(
        name="price_vs_vwap",
        category="volume",
        lookback_period=1,
        output_columns=["price_vs_vwap"],
        description="Price position relative to VWAP (percentage)",
        default_params={},
    )
)
def compute_price_vs_vwap(data: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute price position relative to VWAP.

    Useful for identifying if price is trading above/below average.

    Args:
        data: DataFrame with 'high', 'low', 'close', 'volume' columns
        params: {}

    Returns:
        DataFrame with percentage above/below VWAP
    """
    _ensure_pandas_ta()

    vwap = ta.vwap(data["high"], data["low"], data["close"], data["volume"])
    price_vs_vwap = (data["close"] - vwap) / vwap * 100

    return pd.DataFrame({"price_vs_vwap": price_vs_vwap}, index=data.index)


@SignalRegistry.register(
    SignalMetadata(
        name="returns",
        category="momentum",
        lookback_period=2,
        output_columns=["returns"],
        description="Simple returns (percentage change)",
        default_params={"period": 1},
    )
)
def compute_returns(data: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute simple returns.

    Args:
        data: DataFrame with 'close' column
        params: {'period': 1}

    Returns:
        DataFrame with percentage returns
    """
    params = params or {}
    period = params.get("period", 1)

    returns = data["close"].pct_change(periods=period)
    return pd.DataFrame({"returns": returns}, index=data.index)


@SignalRegistry.register(
    SignalMetadata(
        name="log_returns",
        category="momentum",
        lookback_period=2,
        output_columns=["log_returns"],
        description="Log returns (used for diffusion models)",
        default_params={"period": 1},
    )
)
def compute_log_returns(data: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute log returns.

    Log returns are preferred for financial modeling due to their
    additive property and better statistical properties.

    Args:
        data: DataFrame with 'close' column
        params: {'period': 1}

    Returns:
        DataFrame with log returns
    """
    params = params or {}
    period = params.get("period", 1)

    log_returns = np.log(data["close"] / data["close"].shift(period))
    return pd.DataFrame({"log_returns": log_returns}, index=data.index)


@SignalRegistry.register(
    SignalMetadata(
        name="realized_volatility",
        category="volatility",
        lookback_period=20,
        output_columns=["realized_vol"],
        description="Realized volatility (annualized standard deviation of returns)",
        default_params={"period": 20, "annualize": True},
    )
)
def compute_realized_volatility(
    data: pd.DataFrame, params: Optional[dict] = None
) -> pd.DataFrame:
    """
    Compute realized volatility.

    Rolling standard deviation of returns, optionally annualized.

    Args:
        data: DataFrame with 'close' column
        params: {'period': 20, 'annualize': True}

    Returns:
        DataFrame with volatility values
    """
    params = params or {}
    period = params.get("period", 20)
    annualize = params.get("annualize", True)

    returns = data["close"].pct_change()
    vol = returns.rolling(window=period).std()

    # Annualize (assuming 252 trading days, 78 5-min bars per day)
    if annualize:
        bars_per_year = 252 * 78  # For 5-min bars
        vol = vol * np.sqrt(bars_per_year)

    return pd.DataFrame({"realized_vol": vol}, index=data.index)
