"""
Train/validation/test splitting for financial time series.

Implements purging and embargo to prevent lookahead bias,
following methodologies from Lopez de Prado's
"Advances in Financial Machine Learning".
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class TemporalSplit:
    """
    Represents a single train/test split with purge and embargo.

    Attributes:
        train_indices: Indices for training data
        test_indices: Indices for test data
        purge_indices: Indices removed due to purging (overlap with test)
        embargo_indices: Indices in embargo period (after test, before next train)
    """

    train_indices: np.ndarray
    test_indices: np.ndarray
    purge_indices: np.ndarray = None
    embargo_indices: np.ndarray = None

    def __post_init__(self):
        if self.purge_indices is None:
            self.purge_indices = np.array([], dtype=np.int64)
        if self.embargo_indices is None:
            self.embargo_indices = np.array([], dtype=np.int64)

    @property
    def train_size(self) -> int:
        return len(self.train_indices)

    @property
    def test_size(self) -> int:
        return len(self.test_indices)


class TimeSeriesSplit:
    """
    Simple expanding window time series split.

    Divides data chronologically into train/val/test sets
    with optional purge and embargo windows.
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        purge_window: int = 0,
        embargo_window: int = 0,
    ):
        """
        Initialize the splitter.

        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            purge_window: Number of samples to purge around splits
            embargo_window: Number of samples to embargo after test set
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.purge_window = purge_window
        self.embargo_window = embargo_window

    def split(
        self, n_samples: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data indices into train/val/test.

        Args:
            n_samples: Total number of samples

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        indices = np.arange(n_samples)

        # Calculate split points
        train_end = int(n_samples * self.train_ratio)
        val_end = train_end + int(n_samples * self.val_ratio)

        # Apply purge and embargo
        train_end_purged = max(0, train_end - self.purge_window)
        val_start_embargoed = train_end + self.embargo_window
        val_end_purged = max(val_start_embargoed, val_end - self.purge_window)
        test_start_embargoed = val_end + self.embargo_window

        train_indices = indices[:train_end_purged]
        val_indices = indices[val_start_embargoed:val_end_purged]
        test_indices = indices[test_start_embargoed:]

        return train_indices, val_indices, test_indices


class PurgedKFoldCV:
    """
    Purged K-Fold Cross-Validation for financial time series.

    Implements the method from Lopez de Prado to prevent lookahead bias:
    1. Purging: Remove training samples that overlap with test samples
    2. Embargo: Add a gap after the test set before resuming training

    This is critical for financial data where samples may have
    overlapping information (e.g., returns computed over overlapping windows).
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_window: int = 10,
        embargo_window: int = 5,
    ):
        """
        Initialize the cross-validator.

        Args:
            n_splits: Number of folds
            purge_window: Number of samples to purge around test set
            embargo_window: Number of samples to embargo after test set
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_window = embargo_window

    def split(
        self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray = None
    ) -> Iterator[TemporalSplit]:
        """
        Generate purged k-fold splits.

        Args:
            X: Feature matrix (only used for length)
            y: Target array (ignored, for sklearn compatibility)

        Yields:
            TemporalSplit objects for each fold
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate fold boundaries
        fold_size = n_samples // self.n_splits
        fold_starts = [i * fold_size for i in range(self.n_splits)]
        fold_ends = [min((i + 1) * fold_size, n_samples) for i in range(self.n_splits)]

        for fold_idx in range(self.n_splits):
            # Test set is the current fold
            test_start = fold_starts[fold_idx]
            test_end = fold_ends[fold_idx]
            test_indices = indices[test_start:test_end]

            # Calculate purge boundaries
            purge_start = max(0, test_start - self.purge_window)
            embargo_end = min(n_samples, test_end + self.embargo_window)

            # Track purged and embargoed indices
            purge_indices = indices[purge_start:test_start]
            embargo_indices = indices[test_end:embargo_end]

            # Train set is everything except test, purge, and embargo
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[purge_start:embargo_end] = False
            train_indices = indices[train_mask]

            yield TemporalSplit(
                train_indices=train_indices,
                test_indices=test_indices,
                purge_indices=purge_indices,
                embargo_indices=embargo_indices,
            )

    def get_n_splits(self) -> int:
        """Return the number of splits."""
        return self.n_splits


class WalkForwardValidator:
    """
    Walk-forward validation with expanding or rolling windows.

    Simulates realistic trading scenarios where models are
    trained on past data and tested on future data.

    Supports:
    - Expanding window: Training set grows over time
    - Rolling window: Fixed-size training window slides forward
    """

    def __init__(
        self,
        initial_train_size: int,
        test_size: int,
        step_size: int = None,
        expanding: bool = True,
        purge_window: int = 0,
        embargo_window: int = 0,
        max_splits: int = None,
    ):
        """
        Initialize the walk-forward validator.

        Args:
            initial_train_size: Initial training set size
            test_size: Size of each test set
            step_size: How much to advance each iteration (default: test_size)
            expanding: If True, training window expands; if False, it rolls
            purge_window: Samples to purge before test set
            embargo_window: Samples to embargo after test set
            max_splits: Maximum number of splits to generate
        """
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.expanding = expanding
        self.purge_window = purge_window
        self.embargo_window = embargo_window
        self.max_splits = max_splits

    def split(
        self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray = None
    ) -> Iterator[TemporalSplit]:
        """
        Generate walk-forward splits.

        Args:
            X: Feature matrix (only used for length)
            y: Target array (ignored)

        Yields:
            TemporalSplit objects for each walk-forward step
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate number of possible splits
        remaining = n_samples - self.initial_train_size
        n_splits = remaining // self.step_size

        if self.max_splits is not None:
            n_splits = min(n_splits, self.max_splits)

        for i in range(n_splits):
            # Calculate current positions
            if self.expanding:
                train_start = 0
            else:
                train_start = i * self.step_size

            train_end = self.initial_train_size + i * self.step_size
            test_start = train_end + self.embargo_window
            test_end = min(test_start + self.test_size, n_samples)

            # Skip if test set would be empty
            if test_start >= n_samples:
                break

            # Apply purging
            purge_start = max(train_start, train_end - self.purge_window)

            train_indices = indices[train_start:purge_start]
            test_indices = indices[test_start:test_end]
            purge_indices = indices[purge_start:train_end]
            embargo_indices = indices[train_end:test_start]

            yield TemporalSplit(
                train_indices=train_indices,
                test_indices=test_indices,
                purge_indices=purge_indices,
                embargo_indices=embargo_indices,
            )

    def get_n_splits(self, n_samples: int) -> int:
        """Estimate the number of splits for given data size."""
        remaining = n_samples - self.initial_train_size
        n_splits = remaining // self.step_size
        if self.max_splits is not None:
            n_splits = min(n_splits, self.max_splits)
        return n_splits


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).

    Generates multiple backtest paths to enable:
    1. Distribution of performance estimates
    2. Probability of Backtest Overfitting (PBO) calculation
    3. Deflated Sharpe Ratio computation

    The idea is to generate C(n_splits, n_test_splits) different
    backtest paths, each testing on a different combination of folds.

    Reference: Lopez de Prado, "Advances in Financial Machine Learning"
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        purge_window: int = 10,
        embargo_window: int = 5,
    ):
        """
        Initialize CPCV.

        Args:
            n_splits: Total number of folds to divide data into
            n_test_splits: Number of folds to use for testing in each path
            purge_window: Samples to purge around test sets
            embargo_window: Samples to embargo after test sets
        """
        if n_test_splits >= n_splits:
            raise ValueError("n_test_splits must be less than n_splits")

        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_window = purge_window
        self.embargo_window = embargo_window

    def get_n_paths(self) -> int:
        """
        Return the number of unique backtest paths.

        This is C(n_splits, n_test_splits).
        """
        from math import comb
        return comb(self.n_splits, self.n_test_splits)

    def split(
        self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray = None
    ) -> Iterator[list[TemporalSplit]]:
        """
        Generate all combinatorial backtest paths.

        Each path is a list of TemporalSplit objects representing
        the sequential test sets in that path.

        Args:
            X: Feature matrix (only used for length)
            y: Target array (ignored)

        Yields:
            List of TemporalSplit objects for each backtest path
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Divide data into n_splits groups
        fold_size = n_samples // self.n_splits
        folds = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            folds.append(indices[start:end])

        # Generate all combinations for test sets
        for test_combo in combinations(range(self.n_splits), self.n_test_splits):
            path_splits = []

            for test_fold_idx in sorted(test_combo):
                test_indices = folds[test_fold_idx]
                test_start = test_indices[0]
                test_end = test_indices[-1] + 1

                # Calculate purge and embargo
                purge_start = max(0, test_start - self.purge_window)
                embargo_end = min(n_samples, test_end + self.embargo_window)

                # Training is all folds not in test_combo, with purge/embargo
                train_mask = np.ones(n_samples, dtype=bool)

                # Exclude all test folds and their purge/embargo regions
                for other_test_idx in test_combo:
                    other_test = folds[other_test_idx]
                    other_start = max(0, other_test[0] - self.purge_window)
                    other_end = min(n_samples, other_test[-1] + 1 + self.embargo_window)
                    train_mask[other_start:other_end] = False

                train_indices = indices[train_mask]

                path_splits.append(
                    TemporalSplit(
                        train_indices=train_indices,
                        test_indices=test_indices,
                        purge_indices=indices[purge_start:test_start],
                        embargo_indices=indices[test_end:embargo_end],
                    )
                )

            yield path_splits

    def compute_pbo(
        self,
        in_sample_metrics: np.ndarray,
        out_sample_metrics: np.ndarray,
    ) -> float:
        """
        Compute Probability of Backtest Overfitting (PBO).

        PBO = probability that the best in-sample strategy
              performs below median out-of-sample.

        Args:
            in_sample_metrics: Array of in-sample metrics for each path
            out_sample_metrics: Array of out-of-sample metrics for each path

        Returns:
            PBO value between 0 and 1
        """
        if len(in_sample_metrics) != len(out_sample_metrics):
            raise ValueError("Metric arrays must have same length")

        n_paths = len(in_sample_metrics)

        # For each path, check if best in-sample is below median out-of-sample
        count_overfit = 0

        for i in range(n_paths):
            # Find the strategy with best in-sample performance
            best_in_sample_idx = np.argmax(in_sample_metrics[i])

            # Check if its out-of-sample performance is below median
            oos_performance = out_sample_metrics[i][best_in_sample_idx]
            oos_median = np.median(out_sample_metrics[i])

            if oos_performance < oos_median:
                count_overfit += 1

        return count_overfit / n_paths


def create_train_val_test_datasets(
    data: Union[np.ndarray, pd.DataFrame],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    purge_window: int = 10,
    embargo_window: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to create train/val/test splits.

    Args:
        data: Full dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        purge_window: Samples to purge
        embargo_window: Samples to embargo

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    splitter = TimeSeriesSplit(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=1 - train_ratio - val_ratio,
        purge_window=purge_window,
        embargo_window=embargo_window,
    )

    if isinstance(data, pd.DataFrame):
        data_arr = data.values
    else:
        data_arr = data

    train_idx, val_idx, test_idx = splitter.split(len(data_arr))

    return data_arr[train_idx], data_arr[val_idx], data_arr[test_idx]
