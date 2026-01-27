"""
PyTorch Dataset implementation for financial time series.

Provides efficient data loading for diffusion model training
with proper handling of sliding windows, conditioning, and
temporal alignment to prevent lookahead bias.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessors.normalizers import BaseNormalizer, ZScoreNormalizer


@dataclass
class DatasetConfig:
    """Configuration for the financial time series dataset."""

    window_size: int = 64  # Historical context window
    prediction_horizon: int = 5  # Steps ahead to predict
    features_to_predict: list[int] = None  # Feature indices to predict (default: all)
    conditional_features: list[int] = None  # Feature indices for conditioning
    normalize: bool = True
    return_timestamps: bool = False  # Whether to return timestamps (for debugging)


class FinancialTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for financial time series with diffusion model training.

    Handles:
    - Sliding window creation with proper temporal alignment
    - Separation of features to predict vs conditioning features
    - Normalization with fit/transform pattern (no lookahead)
    - Support for both training and inference modes

    Data layout:
    - Each sample contains:
      - history: (window_size, num_features) - historical context
      - target: (prediction_horizon, num_target_features) - what to predict
      - condition: conditioning features if specified
    """

    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        config: DatasetConfig = None,
        normalizer: BaseNormalizer = None,
        normalizer_state: dict = None,
        is_train: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            data: Time series data, shape (T, num_features)
            config: Dataset configuration
            normalizer: Fitted normalizer (for test/val sets)
            normalizer_state: Normalizer state dict (alternative to normalizer)
            is_train: Whether this is training data (affects normalization)
        """
        self.config = config or DatasetConfig()
        self.is_train = is_train

        # Convert to numpy
        if isinstance(data, pd.DataFrame):
            self.timestamps = data.index.values if self.config.return_timestamps else None
            self.feature_names = data.columns.tolist()
            data = data.values
        else:
            self.timestamps = None
            self.feature_names = None

        self.raw_data = data.astype(np.float32)

        # Validate data shape
        if self.raw_data.ndim == 1:
            self.raw_data = self.raw_data.reshape(-1, 1)

        self.num_timesteps, self.num_features = self.raw_data.shape

        # Setup feature indices
        self._setup_feature_indices()

        # Setup normalization
        self._setup_normalization(normalizer, normalizer_state)

        # Normalize data
        if self.config.normalize and self.normalizer is not None:
            self.data = self.normalizer.transform(self.raw_data)
            if isinstance(self.data, pd.DataFrame):
                self.data = self.data.values.astype(np.float32)
        else:
            self.data = self.raw_data

        # Calculate valid indices
        self._calculate_valid_indices()

    def _setup_feature_indices(self):
        """Setup which features to predict and which to condition on."""
        # Features to predict (targets)
        if self.config.features_to_predict is None:
            # Default: predict all features
            self.target_indices = list(range(self.num_features))
        else:
            self.target_indices = self.config.features_to_predict

        # Conditional features
        self.condition_indices = self.config.conditional_features

        self.num_target_features = len(self.target_indices)

    def _setup_normalization(
        self,
        normalizer: BaseNormalizer,
        normalizer_state: dict,
    ):
        """Setup normalization (fit on training data only)."""
        if not self.config.normalize:
            self.normalizer = None
            return

        if normalizer is not None:
            # Use provided normalizer (for val/test sets)
            self.normalizer = normalizer
        elif normalizer_state is not None:
            # Reconstruct from state
            self.normalizer = ZScoreNormalizer()
            self.normalizer.mean_ = np.array(normalizer_state["mean"])
            self.normalizer.std_ = np.array(normalizer_state["std"])
            self.normalizer.state.is_fitted = True
        elif self.is_train:
            # Fit new normalizer on training data
            self.normalizer = ZScoreNormalizer()
            self.normalizer.fit(self.raw_data)
        else:
            raise ValueError(
                "Normalizer or normalizer_state must be provided for non-training data"
            )

    def _calculate_valid_indices(self):
        """Calculate valid start indices for samples."""
        # We need at least window_size + prediction_horizon consecutive points
        min_length = self.config.window_size + self.config.prediction_horizon

        if self.num_timesteps < min_length:
            raise ValueError(
                f"Data length ({self.num_timesteps}) is less than "
                f"minimum required ({min_length})"
            )

        # Valid indices are those where we can extract a full sample
        self.valid_indices = np.arange(
            self.config.window_size - 1,  # Start after enough history
            self.num_timesteps - self.config.prediction_horizon,  # End before running out of future
        )

    def __len__(self) -> int:
        """Return number of valid samples."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            dict with keys:
            - 'history': (window_size, num_features) - historical context
            - 'target': (prediction_horizon, num_target_features) - prediction target
            - 'condition': conditioning features (if specified)
            - 'timestamp': original timestamp (if return_timestamps=True)
        """
        # Map to actual data index
        data_idx = self.valid_indices[idx]

        # Extract history window (ends at data_idx, inclusive)
        history_start = data_idx - self.config.window_size + 1
        history_end = data_idx + 1
        history = self.data[history_start:history_end]

        # Extract target (prediction_horizon steps after history)
        target_start = data_idx + 1
        target_end = target_start + self.config.prediction_horizon
        target = self.data[target_start:target_end, self.target_indices]

        sample = {
            "history": torch.from_numpy(history.copy()),
            "target": torch.from_numpy(target.copy()),
        }

        # Add conditioning features if specified
        if self.condition_indices is not None:
            condition = history[:, self.condition_indices]
            sample["condition"] = torch.from_numpy(condition.copy())

        # Add timestamp if requested
        if self.config.return_timestamps and self.timestamps is not None:
            sample["timestamp"] = self.timestamps[data_idx]

        return sample

    def get_normalizer_state(self) -> Optional[dict]:
        """Get normalizer state for saving/loading."""
        if self.normalizer is None:
            return None
        return self.normalizer.state.params

    def get_full_sequence(self, normalized: bool = True) -> np.ndarray:
        """Return the full sequence (for inference)."""
        return self.data if normalized else self.raw_data

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        config: DatasetConfig = None,
        **kwargs,
    ) -> "FinancialTimeSeriesDataset":
        """
        Create dataset from pandas DataFrame.

        Args:
            df: DataFrame with datetime index and feature columns
            config: Dataset configuration
            **kwargs: Additional arguments to __init__

        Returns:
            FinancialTimeSeriesDataset instance
        """
        return cls(data=df, config=config, **kwargs)


class InferenceDataset(Dataset):
    """
    Dataset for inference/prediction.

    Provides only the history needed for generating predictions,
    without targets (since we're predicting the future).
    """

    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        window_size: int = 64,
        normalizer: BaseNormalizer = None,
        condition_indices: list[int] = None,
    ):
        """
        Initialize inference dataset.

        Args:
            data: Time series data
            window_size: Historical context window
            normalizer: Fitted normalizer from training
            condition_indices: Indices of conditioning features
        """
        self.window_size = window_size
        self.normalizer = normalizer
        self.condition_indices = condition_indices

        # Convert to numpy
        if isinstance(data, pd.DataFrame):
            data = data.values

        self.raw_data = data.astype(np.float32)

        if self.raw_data.ndim == 1:
            self.raw_data = self.raw_data.reshape(-1, 1)

        # Normalize
        if self.normalizer is not None:
            self.data = self.normalizer.transform(self.raw_data)
            if isinstance(self.data, pd.DataFrame):
                self.data = self.data.values.astype(np.float32)
        else:
            self.data = self.raw_data

    def get_latest_window(self) -> dict:
        """
        Get the most recent window for prediction.

        Returns:
            dict with 'history' and optionally 'condition'
        """
        history = self.data[-self.window_size:]

        sample = {
            "history": torch.from_numpy(history.copy()).unsqueeze(0),  # Add batch dim
        }

        if self.condition_indices is not None:
            condition = history[:, self.condition_indices]
            sample["condition"] = torch.from_numpy(condition.copy()).unsqueeze(0)

        return sample

    def __len__(self) -> int:
        """Return number of valid prediction points."""
        return max(0, len(self.data) - self.window_size + 1)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample for prediction at index idx."""
        history = self.data[idx : idx + self.window_size]

        sample = {
            "history": torch.from_numpy(history.copy()),
        }

        if self.condition_indices is not None:
            condition = history[:, self.condition_indices]
            sample["condition"] = torch.from_numpy(condition.copy())

        return sample


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate function for DataLoader.

    Stacks all tensors in the batch properly.
    """
    result = {}

    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([sample[key] for sample in batch])
        else:
            result[key] = [sample[key] for sample in batch]

    return result
