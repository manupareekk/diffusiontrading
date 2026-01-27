"""
Streamlit Dashboard for Diffusion Model Trading System.

Run with: streamlit run app.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Diffusion Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .positive { color: #00c853; }
    .negative { color: #ff1744; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Sidebar Navigation
# ============================================================================

st.sidebar.title("📈 Diffusion Trading")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Data", "🎯 Backtest", "🔬 Sensitivity", "🤖 Model"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Actions**")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()


# ============================================================================
# Home Page
# ============================================================================

if page == "🏠 Home":
    st.title("Diffusion Model Trading System")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Welcome!

        This dashboard provides an interface for:

        - **📊 Data**: Download and visualize market data
        - **🎯 Backtest**: Run trading strategy backtests
        - **🔬 Sensitivity**: Analyze parameter sensitivity
        - **🤖 Model**: Train and evaluate diffusion models

        ### Getting Started

        1. Go to **Data** tab to download market data
        2. Run a **Backtest** with your chosen strategy
        3. Analyze **Sensitivity** to optimize parameters
        4. Train the **Model** for prediction-based trading
        """)

    with col2:
        st.markdown("""
        ### System Components

        | Component | Status |
        |-----------|--------|
        | Data Pipeline | ✅ Ready |
        | Signal Generation | ✅ Ready |
        | Diffusion Model | ✅ Ready |
        | Backtesting Engine | ✅ Ready |
        | Sensitivity Analysis | ✅ Ready |

        ### Technical Indicators Available

        - RSI, MACD, Stochastic
        - Bollinger Bands, ATR, Keltner
        - VWAP, OBV, MFI
        - SMA, EMA, ADX
        """)

    st.markdown("---")
    st.info("💡 **Tip**: Start by downloading data in the Data tab, then run a backtest to see strategy performance.")


# ============================================================================
# Data Page
# ============================================================================

elif page == "📊 Data":
    st.title("📊 Market Data")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        symbol = st.text_input("Symbol", value="SPY")
    with col2:
        interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=1)
    with col3:
        days = st.number_input("Days of History", min_value=1, max_value=60, value=30)

    if st.button("📥 Download Data", type="primary"):
        with st.spinner(f"Downloading {symbol} data..."):
            try:
                import yfinance as yf

                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)

                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                )

                # Standardize column names
                data.columns = [c.lower() for c in data.columns]
                if 'stock splits' in data.columns:
                    data = data.drop(columns=['stock splits'])
                if 'dividends' in data.columns:
                    data = data.drop(columns=['dividends'])

                # Save to session state
                st.session_state['market_data'] = data
                st.session_state['symbol'] = symbol

                # Save to file
                output_dir = Path("artifacts/data/raw")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{symbol}_{interval}.parquet"
                data.to_parquet(output_file)

                st.success(f"✅ Downloaded {len(data)} bars and saved to {output_file}")

            except Exception as e:
                st.error(f"Error downloading data: {e}")

    st.markdown("---")

    # Display data if available
    if 'market_data' in st.session_state:
        data = st.session_state['market_data']
        symbol = st.session_state.get('symbol', 'Unknown')

        st.subheader(f"{symbol} Price Chart")

        # Price chart
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05, row_heights=[0.7, 0.3])

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='OHLC'
        ), row=1, col=1)

        # Volume
        colors = ['green' if c >= o else 'red'
                  for c, o in zip(data['close'], data['open'])]
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['volume'],
            marker_color=colors,
            name='Volume',
            opacity=0.7
        ), row=2, col=1)

        fig.update_layout(
            height=600,
            showlegend=False,
            xaxis_rangeslider_visible=False,
            yaxis_title="Price",
            yaxis2_title="Volume"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Data statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Bars", f"{len(data):,}")
        with col2:
            st.metric("Date Range", f"{data.index.min().date()} to {data.index.max().date()}")
        with col3:
            returns = data['close'].pct_change().dropna()
            st.metric("Avg Daily Return", f"{returns.mean()*100:.3f}%")
        with col4:
            st.metric("Volatility", f"{returns.std()*100:.3f}%")

        # Show raw data
        with st.expander("📋 View Raw Data"):
            st.dataframe(data.tail(100))

    else:
        st.info("👆 Download data using the form above to get started.")

    # Load existing data
    st.markdown("---")
    st.subheader("Load Existing Data")

    data_dir = Path("artifacts/data/raw")
    if data_dir.exists():
        files = list(data_dir.glob("*.parquet"))
        if files:
            selected_file = st.selectbox("Select file", [f.name for f in files])
            if st.button("📂 Load"):
                data = pd.read_parquet(data_dir / selected_file)
                st.session_state['market_data'] = data
                st.session_state['symbol'] = selected_file.split('_')[0]
                st.success(f"Loaded {len(data)} bars")
                st.rerun()


# ============================================================================
# Backtest Page
# ============================================================================

elif page == "🎯 Backtest":
    st.title("🎯 Strategy Backtest")
    st.markdown("---")

    if 'market_data' not in st.session_state:
        st.warning("⚠️ Please load market data first in the Data tab.")
        st.stop()

    data = st.session_state['market_data']

    # Strategy configuration
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Strategy Settings")

        strategy_type = st.selectbox(
            "Strategy Type",
            [
                "Signal-Based (RSI)",
                "Diffusion - Probability",
                "Diffusion - Bands"
            ],
            index=0
        )

        # Strategy-specific configuration
        if "Diffusion" in strategy_type:
            if 'trained_model' not in st.session_state:
                st.error("⚠️ No trained model found. Please train a model in the **Model** tab first.")
                st.stop()

            st.info(f"✅ Using trained model: {st.session_state.get('model_config', {}).get('hidden_dim', 'N/A')}D")

            # Diffusion-specific settings
            if "Bands" in strategy_type:
                band_type = st.selectbox("Band Type", ["percentile", "std"])
                band_width = st.selectbox(
                    "Band Width",
                    ["percentile_5_95", "percentile_10_90", "1std", "2std"],
                    index=0
                )

            num_samples = st.slider("Monte Carlo Samples", 10, 500, 100, step=10)
            fast_mode = st.checkbox("⚡ Fast Backtest (last 100 bars only)", value=True)

        elif "RSI" in strategy_type:
            rsi_period = st.slider("RSI Period", 5, 20, 9)
            rsi_oversold = st.slider("RSI Oversold", 20, 40, 30)
            rsi_overbought = st.slider("RSI Overbought", 60, 80, 70)

    with col2:
        st.subheader("Backtest Settings")

        initial_capital = st.number_input("Initial Capital ($)",
                                          min_value=10000, max_value=10000000,
                                          value=100000, step=10000)
        commission = st.number_input("Commission (%)",
                                     min_value=0.0, max_value=1.0,
                                     value=0.1, step=0.01) / 100
        slippage = st.number_input("Slippage (bps)",
                                   min_value=0.0, max_value=50.0,
                                   value=5.0, step=1.0)

    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                from dataclasses import dataclass
                
                # Check which strategy to use
                if "Diffusion" in strategy_type:
                    # Use diffusion backtest helper
                    from src.backtesting.diffusion_backtest import run_diffusion_backtest, BacktestConfig
                    
                    # Create progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def progress_callback(progress, message):
                        progress_bar.progress(progress)
                        status_text.text(message)
                    
                    # Configure backtest
                    bt_config = BacktestConfig(
                        initial_capital=initial_capital,
                        commission=commission,
                        slippage_bps=slippage,
                        fast_mode=fast_mode,
                        fast_mode_bars=100
                    )
                    
                    # Run diffusion backtest
                    result = run_diffusion_backtest(
                        data=data,
                        model=st.session_state['trained_model'],
                        model_config=st.session_state.get('model_config', {}),
                        strategy_type=strategy_type,
                        config=bt_config,
                        band_type=band_type if "Bands" in strategy_type else "percentile",
                        band_width=band_width if "Bands" in strategy_type else "percentile_5_95",
                        num_samples=num_samples,
                        progress_callback=progress_callback,
                    )
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Convert to format expected by display code
                    @dataclass
                    class Trade:
                        entry_time: any
                        exit_time: any
                        direction: int
                        entry_price: float
                        exit_price: float
                        pnl: float
                        return_pct: float
                    
                    trade_objects = [Trade(**t) for t in result.trades]
                    
                    @dataclass
                    class SimpleBacktestResult:
                        equity_curve: pd.Series
                        trades: list
                        metrics: dict
                    
                    backtest_result = SimpleBacktestResult(
                        equity_curve=result.equity_curve,
                        trades=trade_objects,
                        metrics=result.metrics
                    )
                    
                    st.session_state['backtest_result'] = backtest_result
                    st.success("✅ Backtest completed!")
                    
                else:  # RSI Strategy
                    from src.strategy.diffusion_strategy import SignalBasedStrategy

                    # Create strategy
                    strategy = SignalBasedStrategy(
                        rsi_period=rsi_period if "RSI" in strategy_type else 9,
                        rsi_oversold=rsi_oversold if "RSI" in strategy_type else 30,
                        rsi_overbought=rsi_overbought if "RSI" in strategy_type else 70,
                    )

                    # Generate signals
                    signals = strategy.generate_signals(data)

                    # Simple backtest
                    cash = initial_capital
                    position = 0  # 0 = flat, 1 = long
                    equity_history = []
                    trades = []
                    entry_price = 0
                    entry_time = None

                    for i in range(len(data)):
                        price = data['close'].iloc[i]
                        time = data.index[i]
                        signal = signals.iloc[i] if i < len(signals) else 0

                        # Execute signals
                        if signal == 1 and position == 0:  # Buy
                            position = 1
                            entry_price = price * (1 + slippage/10000)  # Slippage
                            shares = (cash * (1 - commission)) / entry_price
                            cash = 0
                            entry_time = time
                        elif signal == -1 and position == 1:  # Sell
                            exit_price = price * (1 - slippage/10000)
                            cash = shares * exit_price * (1 - commission)
                            pnl = cash - initial_capital
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

                        # Calculate equity
                        if position == 1:
                            equity = shares * price
                        else:
                            equity = cash if cash > 0 else initial_capital
                        equity_history.append((time, equity))

                    # Create result object
                    @dataclass
                    class SimpleBacktestResult:
                        equity_curve: pd.Series
                        trades: list
                        metrics: dict

                    equity_df = pd.DataFrame(equity_history, columns=['time', 'equity'])
                    equity_df.set_index('time', inplace=True)
                    equity_curve = equity_df['equity']

                    # Calculate metrics
                    returns = equity_curve.pct_change().dropna()
                    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

                    # Sharpe ratio
                    if len(returns) > 1 and returns.std() > 0:
                        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 78)  # 78 5-min bars per day
                    else:
                        sharpe = 0

                    # Max drawdown
                    rolling_max = equity_curve.expanding().max()
                    drawdown = (equity_curve - rolling_max) / rolling_max
                    max_dd = abs(drawdown.min())

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

                    # Create Trade objects for display
                    @dataclass
                    class Trade:
                        entry_time: any
                        exit_time: any
                        direction: int
                        entry_price: float
                        exit_price: float
                        pnl: float
                        return_pct: float

                    trade_objects = [Trade(**t) for t in trades]

                    result = SimpleBacktestResult(
                        equity_curve=equity_curve,
                        trades=trade_objects,
                        metrics=metrics
                    )

                    # Store results
                    st.session_state['backtest_result'] = result

                    st.success("✅ Backtest completed!")

            except Exception as e:
                st.error(f"Error running backtest: {e}")
                import traceback
                st.code(traceback.format_exc())

    st.markdown("---")

    # Display results
    if 'backtest_result' in st.session_state:
        result = st.session_state['backtest_result']

        st.subheader("Performance Summary")

        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        metrics = result.metrics

        with col1:
            total_return = metrics.get('total_return', 0) * 100
            st.metric("Total Return", f"{total_return:.2f}%",
                     delta=f"{'+'if total_return > 0 else ''}{total_return:.2f}%")

        with col2:
            sharpe = metrics.get('sharpe_ratio', 0)
            st.metric("Sharpe Ratio", f"{sharpe:.3f}")

        with col3:
            max_dd = metrics.get('max_drawdown', 0) * 100
            st.metric("Max Drawdown", f"{max_dd:.2f}%")

        with col4:
            win_rate = metrics.get('win_rate', 0) * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")

        with col5:
            n_trades = metrics.get('total_trades', len(result.trades))
            st.metric("Total Trades", f"{n_trades}")

        # Equity curve
        st.subheader("Equity Curve")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05, row_heights=[0.7, 0.3])

        equity = result.equity_curve

        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity.values,
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=1.5)
        ), row=1, col=1)

        # Drawdown
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max * 100

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red', width=1)
        ), row=2, col=1)

        fig.update_layout(
            height=500,
            showlegend=True,
            yaxis_title="Portfolio Value ($)",
            yaxis2_title="Drawdown (%)"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Trade list
        if result.trades:
            st.subheader("Trade History")

            trades_df = pd.DataFrame([
                {
                    "Entry Time": t.entry_time,
                    "Exit Time": t.exit_time,
                    "Direction": t.direction,
                    "Entry Price": f"${t.entry_price:.2f}",
                    "Exit Price": f"${t.exit_price:.2f}",
                    "P&L": f"${t.pnl:.2f}",
                    "Return": f"{t.return_pct*100:.2f}%"
                }
                for t in result.trades[-50:]  # Last 50 trades
            ])

            st.dataframe(trades_df, use_container_width=True)

        # All metrics
        with st.expander("📊 All Metrics"):
            metrics_df = pd.DataFrame([
                {"Metric": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)}
                for k, v in metrics.items()
            ])
            st.dataframe(metrics_df, use_container_width=True)


# ============================================================================
# Sensitivity Page
# ============================================================================

elif page == "🔬 Sensitivity":
    st.title("🔬 Sensitivity Analysis")
    st.markdown("---")

    if 'market_data' not in st.session_state:
        st.warning("⚠️ Please load market data first in the Data tab.")
        st.stop()

    data = st.session_state['market_data']

    analysis_type = st.selectbox(
        "Analysis Type",
        ["Parameter Sweep", "Monte Carlo Simulation", "Robustness Test"]
    )

    if analysis_type == "Parameter Sweep":
        st.subheader("Parameter Sweep Configuration")

        col1, col2 = st.columns(2)

        with col1:
            param1_name = st.selectbox("Parameter 1", ["rsi_period", "rsi_oversold", "rsi_overbought"])
            param1_min = st.number_input(f"{param1_name} Min", value=5)
            param1_max = st.number_input(f"{param1_name} Max", value=15)
            param1_step = st.number_input(f"{param1_name} Step", value=2)

        with col2:
            param2_name = st.selectbox("Parameter 2", ["rsi_oversold", "rsi_period", "rsi_overbought"], index=0)
            param2_min = st.number_input(f"{param2_name} Min", value=25)
            param2_max = st.number_input(f"{param2_name} Max", value=35)
            param2_step = st.number_input(f"{param2_name} Step", value=5)

        if st.button("🔍 Run Parameter Sweep", type="primary"):
            with st.spinner("Running parameter sweep..."):
                try:
                    from src.backtesting.engine import BacktestEngine, BacktestConfig
                    from src.strategy.diffusion_strategy import SignalBasedStrategy

                    param1_values = list(range(int(param1_min), int(param1_max)+1, int(param1_step)))
                    param2_values = list(range(int(param2_min), int(param2_max)+1, int(param2_step)))

                    results = []
                    total = len(param1_values) * len(param2_values)
                    progress = st.progress(0)

                    for i, p1 in enumerate(param1_values):
                        for j, p2 in enumerate(param2_values):
                            params = {"rsi_period": 9, "rsi_oversold": 30, "rsi_overbought": 70}
                            params[param1_name] = p1
                            params[param2_name] = p2

                            strategy = SignalBasedStrategy(**params)
                            config = BacktestConfig(initial_capital=100000, commission_rate=0.001)
                            engine = BacktestEngine(strategy, config)
                            result = engine.run(data)

                            results.append({
                                param1_name: p1,
                                param2_name: p2,
                                "sharpe_ratio": result.metrics.get("sharpe_ratio", 0),
                                "total_return": result.metrics.get("total_return", 0),
                            })

                            progress.progress((i * len(param2_values) + j + 1) / total)

                    results_df = pd.DataFrame(results)
                    st.session_state['sweep_results'] = results_df
                    st.success("✅ Parameter sweep complete!")

                except Exception as e:
                    st.error(f"Error: {e}")

        if 'sweep_results' in st.session_state:
            results_df = st.session_state['sweep_results']

            st.subheader("Sharpe Ratio Heatmap")

            pivot = results_df.pivot_table(
                values='sharpe_ratio',
                index=param2_name,
                columns=param1_name
            )

            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='RdYlGn',
                text=np.round(pivot.values, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
            ))

            fig.update_layout(
                xaxis_title=param1_name,
                yaxis_title=param2_name,
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Best parameters
            best_idx = results_df['sharpe_ratio'].idxmax()
            best = results_df.loc[best_idx]

            st.success(f"**Best Configuration:** {param1_name}={best[param1_name]}, {param2_name}={best[param2_name]} → Sharpe={best['sharpe_ratio']:.4f}")

    elif analysis_type == "Monte Carlo Simulation":
        st.subheader("Monte Carlo Configuration")

        n_simulations = st.slider("Number of Simulations", 100, 5000, 1000)

        if st.button("🎲 Run Monte Carlo", type="primary"):
            with st.spinner("Running Monte Carlo simulation..."):
                try:
                    from src.sensitivity.monte_carlo import MonteCarloSimulator
                    from src.backtesting.engine import BacktestEngine, BacktestConfig
                    from src.strategy.diffusion_strategy import SignalBasedStrategy

                    # Run base backtest
                    strategy = SignalBasedStrategy()
                    config = BacktestConfig(initial_capital=100000, commission_rate=0.001)
                    engine = BacktestEngine(strategy, config)
                    result = engine.run(data)

                    returns = result.equity_curve.pct_change().dropna().values

                    # Monte Carlo
                    simulator = MonteCarloSimulator(n_simulations=n_simulations, seed=42)
                    mc_result = simulator.run_full_simulation(returns)

                    st.session_state['mc_result'] = mc_result
                    st.success("✅ Monte Carlo complete!")

                except Exception as e:
                    st.error(f"Error: {e}")

        if 'mc_result' in st.session_state:
            mc_result = st.session_state['mc_result']

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Return Distribution")

                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=mc_result.return_distribution * 100,
                    nbinsx=50,
                    name='Returns'
                ))

                mean_ret = np.mean(mc_result.return_distribution) * 100
                fig.add_vline(x=mean_ret, line_dash="dash", line_color="red",
                             annotation_text=f"Mean: {mean_ret:.2f}%")

                fig.update_layout(xaxis_title="Total Return (%)", yaxis_title="Frequency")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Risk Metrics")

                st.metric("VaR (95%)", f"{mc_result.var_95*100:.2f}%")
                st.metric("CVaR (95%)", f"{mc_result.cvar_95*100:.2f}%")
                st.metric("Prob of Loss", f"{(mc_result.return_distribution < 0).mean()*100:.1f}%")

                tr = mc_result.metric_results["total_return"]
                st.metric("Mean Return", f"{tr.mean*100:.2f}%")
                st.metric("95% CI", f"[{tr.percentile_5*100:.1f}%, {tr.percentile_95*100:.1f}%]")


# ============================================================================
# Model Page
# ============================================================================

elif page == "🤖 Model":
    st.title("🤖 Diffusion Model")
    st.markdown("---")

    st.info("🚧 **Model Training Interface** - Configure and train the diffusion model for return prediction.")

    if 'market_data' not in st.session_state:
        st.warning("⚠️ Please load market data first in the Data tab.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Configuration")

        hidden_dim = st.selectbox("Hidden Dimension", [64, 128, 256], index=1)
        n_layers = st.slider("Number of Layers", 4, 12, 8)
        num_timesteps = st.selectbox("Diffusion Timesteps", [100, 500, 1000], index=1)
        noise_schedule = st.selectbox("Noise Schedule", ["cosine", "linear", "sigmoid"])

    with col2:
        st.subheader("Training Configuration")

        epochs = st.slider("Epochs", 10, 200, 50)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        learning_rate = st.select_slider("Learning Rate",
                                         options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                                         value=1e-4)
        window_size = st.slider("Input Window Size", 20, 100, 50)
        prediction_horizon = st.slider("Prediction Horizon (steps)", 1, 20, 5)

    st.info(f"💡 Prediction Horizon: {prediction_horizon} steps = {prediction_horizon * 5} minutes (assuming 5-min bars)")

    st.markdown("---")

    if st.button("🚀 Start Training", type="primary"):
        st.warning("⚠️ Training requires GPU for reasonable speed. This is a demonstration of the interface.")

        with st.spinner("Preparing model..."):
            try:
                from src.models.schedulers.noise_schedule import create_noise_schedule
                from src.models.networks.temporal_conv import TemporalConvDenoiser
                from src.models.diffusion.ddpm import DDPM

                # Create model components
                schedule = create_noise_schedule(
                    schedule_type=noise_schedule,
                    num_timesteps=num_timesteps,
                    as_module=False
                )

                denoiser = TemporalConvDenoiser(
                    input_dim=1,
                    hidden_dim=hidden_dim,
                    num_layers=n_layers,
                    kernel_size=3,
                    condition_dim=2,
                )

                model = DDPM(
                    denoiser=denoiser,
                    noise_schedule=schedule,
                    prediction_type="epsilon"
                )

                # Count parameters
                n_params = sum(p.numel() for p in model.parameters())

                st.success(f"✅ Model created with {n_params:,} parameters")

                st.json({
                    "hidden_dim": hidden_dim,
                    "n_layers": n_layers,
                    "num_timesteps": num_timesteps,
                    "noise_schedule": noise_schedule,
                    "total_parameters": f"{n_params:,}"
                })
                
                # --- Training Loop ---
                from src.data.dataset import FinancialTimeSeriesDataset, DatasetConfig
                from src.models.trainer import SimpleTrainer
                from src.data.features import ModelConditioner
                from torch.utils.data import DataLoader
                from src.utils.device import get_device
                import torch

                # 1. Prepare Data
                st.info("Preparing training data...")
                
                # Use current market data from session
                df = st.session_state['market_data'].copy()
                
                # Generate RSI features
                conditioner = ModelConditioner()
                df = conditioner.add_features(df)
                
                # Normalize conditioning features (RSI) manually for now to match strategy expectation
                # Strategy expects normalized RSI in range [-1, 1] approximately
                # ModelConditioner.get_condition_tensor does normalization, but here we need it in the DF
                # for the Dataset to handle.
                # Actually, Dataset handles its own normalization for targets/history.
                # But conditioning features usually need custom handling.
                # Let's use the same logic as ModelConditioner.get_condition_tensor but keep it in DF
                df['rsi_norm'] = (df['rsi'] - 50.0) / 50.0
                
                # We need to construct a dataframe that has 'close' and 'rsi_norm' columns clearly
                # The Dataset takes feature indices. 
                # Let's create a clean DF with [close, rsi_norm]
                train_df = df[['close', 'rsi_norm']].copy()
                
                # Config: Predict 'close' (idx 0), Condition on 'rsi_norm' (idx 1)
                # AND condition on 'close' history too!
                # Wait, diffusion model usually diffuses 'close' (target). 
                # Conditioning is 'history of close' + 'history of RSI'.
                # The dataset class separates 'target' (what we diffuse) from 'condition'.
                # By default Dataset uses 'history' (sliding window of input) as condition if not specified?
                # Let's check Dataset logic.
                # Dataset: 
                # - history: (window_size, num_features)
                # - target: (prediction_horizon, target_features)
                # - condition: history[:, condition_indices]
                
                # We want:
                # Target: Future Close
                # History Input to Denoiser: Noisy Future Close (x_t)
                # Conditioning Input to Denoiser: Past Close + Past RSI
                
                # So input data to Dataset should have 2 columns: [close, rsi_norm]
                # features_to_predict = [0] (Close)
                # conditional_features = [0, 1] (Close, RSI)
                
                ds_config = DatasetConfig(
                    window_size=window_size,
                    prediction_horizon=prediction_horizon,
                    features_to_predict=[0], # Predict Close
                    conditional_features=[0, 1], # Condition on Close + RSI
                    normalize=True
                )
                
                dataset = FinancialTimeSeriesDataset(
                    data=train_df,
                    config=ds_config,
                    is_train=True
                )
                
                # Create DataLoader
                train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                # 2. Setup Trainer
                device = get_device("auto")
                trainer = SimpleTrainer(model, device=device)
                
                # 3. Train
                progress_bar = st.progress(0)
                status_text = st.empty()
                chart_place = st.empty()
                
                loss_history = []
                
                st.info(f"Training on {device.upper()} for {epochs} epochs...")
                
                for epoch in range(epochs):
                    train_loss = trainer.train_epoch(train_loader)
                    loss_history.append(train_loss)
                    
                    # Update UI
                    progress_bar.progress((epoch + 1) / epochs)
                    status_text.text(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.6f}")
                    
                    # Update Chart
                    if len(loss_history) > 1:
                        chart_data = pd.DataFrame(loss_history, columns=["Train Loss"])
                        chart_place.line_chart(chart_data)
                
                st.success("🎉 Training Complete!")
                
                # Save model to session (for Strategy use)
                st.session_state['trained_model'] = model
                st.session_state['normalizer'] = dataset.normalizer
                st.session_state['model_config'] = {
                    "window_size": window_size,
                    "hidden_dim": hidden_dim,
                    "prediction_horizon": prediction_horizon
                }
                
                st.info("You can now go to the **Backtest** tab and select 'Diffusion Model' strategy (coming soon)!")



            except Exception as e:
                st.error(f"Error creating model: {e}")

    st.markdown("---")
    st.subheader("Model Architecture")

    st.markdown("""
    ```
    DDPM (Denoising Diffusion Probabilistic Model)
    ├── Noise Schedule: Cosine/Linear/Sigmoid
    │   └── Precomputed: β, α, α_bar, σ
    │
    ├── Denoiser: Temporal Convolutional Network
    │   ├── Input Projection
    │   ├── Time Embedding (Sinusoidal → MLP)
    │   ├── Condition Embedding
    │   └── Dilated Causal Conv Blocks × N
    │       ├── Causal Conv1d (no future leakage)
    │       ├── Gated Activation (tanh × sigmoid)
    │       └── Residual Connection
    │
    └── Output: Predicted noise ε or x₀
    ```

    **Key Features:**
    - Causal architecture prevents lookahead bias
    - Dilated convolutions for large receptive field
    - Conditional generation based on market signals
    """)


# ============================================================================
# Footer
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("**Diffusion Trading System**")
st.sidebar.markdown("v1.0.0 | Built with Streamlit")
