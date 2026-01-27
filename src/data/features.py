"""
Feature generation for diffusion models.
"""

import pandas as pd
import numpy as np

class ModelConditioner:
    """
    Generates conditioning features for the diffusion model.
    """
    
    def __init__(self, rsi_period: int = 14):
        self.rsi_period = rsi_period
        
    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to dataframe.
        """
        df = data.copy()
        
        # Calculate RSI manually to avoid pandas-ta dependency
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
        # Fill NaN values (common at start of series)
        df['rsi'] = df['rsi'].fillna(50.0)
        
        return df
    
    def get_condition_tensor(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract and normalize conditioning features for the model.
        Returns: numpy array of shape (N, condition_dim)
        """
        if 'rsi' not in data.columns:
            data = self.add_features(data)
            
        # Normalize RSI to [-1, 1] range for better model stability
        # RSI is [0, 100], so (rsi - 50) / 50
        rsi_norm = (data['rsi'].values - 50.0) / 50.0
        
        # Handle potential NaNs from division by zero in RS
        rsi_norm = np.nan_to_num(rsi_norm, nan=0.0)
        
        # Reshape to (N, 1)
        return rsi_norm.reshape(-1, 1)
