"""
Configuration management using Pydantic settings.

Supports loading from environment variables, .env files, and YAML configs.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


class DataSettings(BaseSettings):
    """Data pipeline configuration."""

    model_config = SettingsConfigDict(env_prefix="DATA_")

    # Data source
    provider: Literal["yfinance", "alpaca", "polygon"] = "yfinance"
    default_symbols: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "AAPL"])

    # Intraday settings
    interval: Literal["1m", "2m", "5m", "15m", "30m", "1h", "1d"] = "5m"
    market_open: str = "09:30"  # ET
    market_close: str = "16:00"  # ET
    include_premarket: bool = False
    include_afterhours: bool = False

    # Cache settings
    cache_dir: Path = ARTIFACTS_DIR / "data" / "raw"
    use_cache: bool = True
    cache_expiry_hours: int = 24

    # Data quality
    min_data_points: int = 100
    max_missing_pct: float = 0.05  # Maximum 5% missing data allowed

    @field_validator("cache_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        return Path(v) if isinstance(v, str) else v


class ModelSettings(BaseSettings):
    """Diffusion model configuration."""

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    # Architecture
    hidden_dim: int = 128
    num_layers: int = 8
    kernel_size: int = 3
    dropout: float = 0.1

    # Diffusion process
    num_timesteps: int = 1000
    noise_schedule: Literal["linear", "cosine"] = "linear"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    prediction_type: Literal["epsilon", "x_start", "v"] = "epsilon"

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    early_stopping_patience: int = 10

    # Sequence parameters
    window_size: int = 64  # Historical context window
    prediction_horizon: int = 5  # Steps ahead to predict

    # Device
    device: str = "cuda"
    num_workers: int = 4

    # Checkpointing
    checkpoint_dir: Path = ARTIFACTS_DIR / "models" / "checkpoints"
    save_top_k: int = 3

    @field_validator("checkpoint_dir", mode="before")
    @classmethod
    def ensure_checkpoint_path(cls, v):
        return Path(v) if isinstance(v, str) else v


class SignalSettings(BaseSettings):
    """Technical signal configuration."""

    model_config = SettingsConfigDict(env_prefix="SIGNAL_")

    # Momentum indicators (shorter periods for intraday)
    rsi_period: int = 9
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    macd_fast: int = 8
    macd_slow: int = 17
    macd_signal: int = 9

    stochastic_k: int = 9
    stochastic_d: int = 3

    # Trend indicators
    sma_short: int = 9
    sma_long: int = 21
    ema_span: int = 12

    # Volatility indicators
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14

    # Volume indicators
    vwap_enabled: bool = True
    obv_enabled: bool = True


class StrategySettings(BaseSettings):
    """Trading strategy configuration."""

    model_config = SettingsConfigDict(env_prefix="STRATEGY_")

    # Position sizing
    position_sizing_method: Literal["fixed", "kelly", "volatility_scaled"] = "volatility_scaled"
    max_position_size: float = 1.0  # Maximum allocation per position
    kelly_fraction: float = 0.5  # Fractional Kelly

    # Risk management
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    max_drawdown_pct: float = 0.10  # 10% max drawdown limit
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = 0.015

    # Signal thresholds
    min_confidence: float = 0.6  # Minimum prediction confidence to trade
    min_expected_return: float = 0.001  # Minimum expected return threshold

    # Ensemble weights
    signal_weight: float = 0.4
    diffusion_weight: float = 0.6


class BacktestSettings(BaseSettings):
    """Backtesting configuration."""

    model_config = SettingsConfigDict(env_prefix="BACKTEST_")

    # Capital
    initial_capital: float = 100000.0

    # Transaction costs (important for intraday)
    commission_rate: float = 0.001  # 0.1% commission
    slippage_bps: float = 5.0  # 5 basis points slippage
    slippage_model: Literal["percentage", "fixed", "volume_based"] = "percentage"

    # Execution
    fill_assumption: Literal["open", "close", "vwap"] = "close"
    allow_shorting: bool = True
    margin_requirement: float = 0.5

    # Validation
    walk_forward_train_pct: float = 0.7
    purge_window: int = 10  # Samples to purge around test set
    embargo_window: int = 5  # Samples to embargo after test set
    n_splits: int = 5  # For k-fold CV

    # Output
    results_dir: Path = ARTIFACTS_DIR / "results" / "backtests"

    @field_validator("results_dir", mode="before")
    @classmethod
    def ensure_results_path(cls, v):
        return Path(v) if isinstance(v, str) else v


class SensitivitySettings(BaseSettings):
    """Sensitivity analysis configuration."""

    model_config = SettingsConfigDict(env_prefix="SENSITIVITY_")

    # Parameter sweep
    search_type: Literal["grid", "random", "sobol"] = "random"
    n_samples: int = 100
    n_jobs: int = -1  # Use all cores

    # Robustness testing
    noise_levels: list[float] = Field(default_factory=lambda: [0.001, 0.005, 0.01, 0.02])
    noise_trials: int = 10
    noise_type: Literal["gaussian", "uniform", "laplace"] = "gaussian"

    # Monte Carlo
    n_simulations: int = 1000
    bootstrap_block_size: int = 20  # For block bootstrap

    # Output
    results_dir: Path = ARTIFACTS_DIR / "results" / "sensitivity"

    @field_validator("results_dir", mode="before")
    @classmethod
    def ensure_sensitivity_path(cls, v):
        return Path(v) if isinstance(v, str) else v


class Settings(BaseSettings):
    """Main settings aggregating all sub-configurations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Sub-configurations
    data: DataSettings = Field(default_factory=DataSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    signal: SignalSettings = Field(default_factory=SignalSettings)
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    sensitivity: SensitivitySettings = Field(default_factory=SensitivitySettings)

    # Global settings
    seed: int = 42
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    def ensure_directories(self) -> None:
        """Create all necessary directories."""
        dirs = [
            self.data.cache_dir,
            self.model.checkpoint_dir,
            self.backtest.results_dir,
            self.sensitivity.results_dir,
            ARTIFACTS_DIR / "data" / "processed",
            ARTIFACTS_DIR / "models" / "final",
            ARTIFACTS_DIR / "reports",
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
