"""
Model Validation Framework

Tests different configurations on properly split data to find optimal parameters.
This ensures we don't overfit and can predict on truly unseen data.

Usage:
    Run cells in order in Google Colab or Jupyter
"""

# Cell 1: Download & Split Data
def download_and_split_data():
    import yfinance as yf
    import pandas as pd
    
    print("📥 Downloading SPY data...")
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="60d", interval="5m")
    data.columns = [col.lower() for col in data.columns]
    data = data[['close']]
    
    # Split: 60% train, 20% validation, 20% test
    total_bars = len(data)
    train_end = int(total_bars * 0.6)
    val_end = int(total_bars * 0.8)
    
    train_data = data.iloc[:train_end].copy()
    val_data = data.iloc[train_end:val_end].copy()
    test_data = data.iloc[val_end:].copy()
    
    print(f"\n📊 Data Split:")
    print(f"   Train:      {len(train_data)} bars")
    print(f"   Validation: {len(val_data)} bars")
    print(f"   Test:       {len(test_data)} bars")
    
    return train_data, val_data, test_data


# Cell 2: Prepare Returns
def prepare_returns(price_data):
    import pandas as pd
    import numpy as np
    
    # Convert to returns
    returns = price_data[['close']].pct_change().dropna()
    returns.columns = ['close_return']
    
    # Add RSI
    close = price_data['close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    returns['rsi'] = rsi.loc[returns.index]
    returns = returns.dropna()
    
    return returns


# Configurations to test
CONFIGS = [
    {'window': 10, 'horizon': 1, 'name': 'Short-10bars-1step'},
    {'window': 20, 'horizon': 3, 'name': 'Medium-20bars-3step'},
    {'window': 50, 'horizon': 5, 'name': 'Long-50bars-5step'},
    {'window': 30, 'horizon': 1, 'name': 'Medium-30bars-1step'},
]


def validate_config(config, train_returns, val_returns, device='cuda'):
    """Train and validate a single configuration"""
    import torch
    from torch.utils.data import DataLoader
    from src.data.dataset import FinancialTimeSeriesDataset, DatasetConfig
    from src.models.diffusion.ddpm import DDPM
    from src.models.networks.temporal_conv import TemporalConvDenoiser
    import numpy as np
    
    print(f"\n{'='*60}")
    print(f"Testing: {config['name']}")
    print(f"{'='*60}")
    
    # Create dataset
    dataset = FinancialTimeSeriesDataset(
        data=train_returns,
        config=DatasetConfig(
            window_size=config['window'],
            prediction_horizon=config['horizon'],
            normalize=False
        ),
        is_train=True
    )
    
    # Create model
    model = DDPM(
        denoiser=TemporalConvDenoiser(input_dim=2, hidden_dim=128, num_layers=8),
        num_timesteps=500
    ).to(device)
    
    # Train
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch in train_loader:
            history = batch['history'].to(device)
            target = batch['target'].to(device)
            condition = batch.get('condition')
            if condition is not None:
                condition = condition.to(device)
            
            optimizer.zero_grad()
            loss = model(target, condition=condition)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/5 - Loss: {avg_loss:.6f}")
    
    # Validate
    model.eval()
    predictions = []
    actuals = []
    
    val_indices = np.random.choice(
        range(config['window'], len(val_returns) - config['horizon']),
        size=min(50, len(val_returns) - config['window'] - config['horizon']),
        replace=False
    )
    
    with torch.no_grad():
        for idx in val_indices:
            window = val_returns.iloc[idx-config['window']:idx].values.astype('float32')
            actual = val_returns.iloc[idx]['close_return']
            
            x = torch.from_numpy(window).unsqueeze(0).to(device)
            pred = model.sample(shape=(1, config['horizon'], 2), condition=x)
            
            predictions.append(pred[0, 0, 0].cpu().item())
            actuals.append(actual)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Metrics
    mae = np.mean(np.abs(predictions - actuals))
    direction_acc = np.sum((predictions > 0) == (actuals > 0)) / len(predictions)
    correlation = np.corrcoef(predictions, actuals)[0, 1]
    
    print(f"\nValidation Results:")
    print(f"  MAE:         {mae:.6f}")
    print(f"  Direction:   {direction_acc:.1%}")
    print(f"  Correlation: {correlation:.4f}")
    
    return {
        'config': config['name'],
        'window': config['window'],
        'horizon': config['horizon'],
        'mae': mae,
        'direction_acc': direction_acc,
        'correlation': correlation,
        'model': model
    }
