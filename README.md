# Diffusion Trading Model

A quantitative trading system using diffusion models to predict price movements, conditioned on technical indicators like RSI.

## Features

- **Diffusion Model**: DDPM-based price prediction with temporal convolution networks
- **Conditional Generation**: Prices conditioned on RSI and historical data
- **Band-Based Trading**: Generates buy/sell signals when price crosses predicted bands
- **Interactive Dashboard**: Streamlit UI for data loading, training, and backtesting
- **GPU Support**: MPS (Mac) and CUDA (Colab) acceleration
- **Backtesting Engine**: Rolling-window backtest with performance metrics

## Quick Start

### Local (CPU)

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
```

### Google Colab (GPU - Recommended)

1. Upload `colab_backtest.ipynb` to [Google Colab](https://colab.research.google.com)
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Follow notebook instructions

## Project Structure

```
diffusionmodel/
├── src/
│   ├── models/          # Diffusion model (DDPM, denoiser, schedulers)
│   ├── strategy/        # Trading strategies (band-based, probability)
│   ├── backtesting/     # Backtest engine and helpers
│   ├── data/            # Data loading, preprocessing, features
│   └── utils/           # Device detection, utilities
├── app.py               # Streamlit dashboard
├── colab_backtest.ipynb # Google Colab notebook
└── config/              # Model and strategy configurations
```

## How It Works

1. **Train Diffusion Model**: Learns to denoise random noise back to real price returns
2. **Condition on RSI**: Model is conditioned on RSI + historical prices
3. **Generate Predictions**: Run Monte Carlo sampling to get price distribution
4. **Calculate Bands**: Create upper/lower bands from percentiles or std
5. **Generate Signals**: 
   - BUY when price < lower band (oversold)
   - SELL when price > upper band (overbought)
6. **Backtest**: Test strategy on historical data

## Usage

### 1. Load Data

Go to **Data** tab → Upload CSV or use sample data

### 2. Train Model

Go to **Model** tab:
- Configure: Hidden dim (128), Layers (8), Prediction horizon (5)
- Click "🚀 Start Training"
- Wait for loss to converge (~5-10 epochs)

### 3. Backtest

Go to **Backtest** tab:
- Strategy: "Diffusion - Bands"
- Band Type: "percentile" 
- Band Width: "percentile_5_95"
- Monte Carlo Samples: 10-100
- Enable "Fast Backtest" for quick validation
- Click "🚀 Run Backtest"

### 4. Analyze Results

View:
- Total Return %
- Sharpe Ratio
- Max Drawdown
- Win Rate
- Equity curve and trade history

## Performance

**CPU (Mac without MPS)**:
- 50 bars, 100 samples: ~40 minutes ⏱️

**GPU (Google Colab T4)**:
- 50 bars, 100 samples: ~2-3 minutes ⚡
- 500 bars, 100 samples: ~15 minutes

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Streamlit
- pandas, numpy, plotly

## Files

- `COLAB_QUICKSTART.md` - Guide for using Google Colab
- `COLAB_GUIDE.md` - Detailed Colab setup
- `colab_backtest.ipynb` - Jupyter notebook for GPU backtesting

## License

MIT

## Author

Built for quantitative trading research and education.
