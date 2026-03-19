"""
Preprocessing utilities for Linear Regression.
Includes feature scaling, train/test splitting, and polynomial features.
"""

import numpy as np
from typing import Tuple, Optional


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    z = (x - μ) / σ
    """

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1  # Avoid division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        return X * self.std_ + self.mean_


class MinMaxScaler:
    """
    Scale features to a given range [feature_range[0], feature_range[1]].
    x_scaled = (x - x_min) / (x_max - x_min) * (max - min) + min
    """

    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        scale = self.max_ - self.min_
        scale[scale == 0] = 1
        X_std = (X - self.min_) / scale
        a, b = self.feature_range
        return X_std * (b - a) + a

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        a, b = self.feature_range
        scale = self.max_ - self.min_
        return (X - a) / (b - a) * scale + self.min_


class PolynomialFeatures:
    """
    Generate polynomial features up to a given degree.

    Example: degree=2, input [x1, x2] → [1, x1, x2, x1², x1x2, x2²]
    """

    def __init__(self, degree: int = 2, include_bias: bool = True):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree
        self.include_bias = include_bias
        self.n_input_features_: Optional[int] = None
        self.n_output_features_: Optional[int] = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_input_features_ = X.shape[1]
        return self._generate_poly_features(X)

    def _generate_poly_features(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        features = []

        if self.include_bias:
            features.append(np.ones((n_samples, 1)))

        for d in range(1, self.degree + 1):
            combos = self._combinations_with_replacement(range(X.shape[1]), d)
            for combo in combos:
                feature = np.prod(X[:, combo], axis=1, keepdims=True)
                features.append(feature)

        result = np.hstack(features)
        self.n_output_features_ = result.shape[1]
        return result

    @staticmethod
    def _combinations_with_replacement(iterable, r):
        """Generate combinations with replacement."""
        pool = tuple(iterable)
        n = len(pool)
        if not n and r:
            return
        indices = [0] * r
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != n - 1:
                    break
            else:
                return
            new_val = indices[i] + 1
            indices[i:] = [new_val] * (r - i)
            yield tuple(pool[i] for i in indices)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split arrays into random train and test subsets.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    test_size : float, optional
        Proportion of data for test set. Default is 0.2 (20%).
    random_state : int or None, optional
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    rng = np.random.default_rng(random_state)
    n_samples = len(y)
    n_test = int(n_samples * test_size)

    indices = rng.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones to X for the bias/intercept term."""
    X = np.array(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.column_stack([np.ones(X.shape[0]), X])


def normalize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features using mean normalization.
    Returns normalized X, mean, and range (max - min).
    """
    X = np.array(X, dtype=float)
    mean = np.mean(X, axis=0)
    rng = np.max(X, axis=0) - np.min(X, axis=0)
    rng[rng == 0] = 1
    return (X - mean) / rng, mean, rng
