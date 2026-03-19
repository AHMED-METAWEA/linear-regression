"""
linear_regression
=================
Linear Regression from scratch — no scikit-learn.

Modules
-------
model          : LinearRegression class + metric functions
preprocessing  : StandardScaler, MinMaxScaler, PolynomialFeatures, train_test_split
visualization  : Plotting utilities for regression diagnostics

Quick Start
-----------
>>> import numpy as np
>>> from linear_regression import LinearRegression, train_test_split, StandardScaler

>>> X = np.random.rand(100, 3)
>>> y = 3 * X[:, 0] + 1.5 * X[:, 1] - 2 * X[:, 2] + np.random.randn(100) * 0.1

>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>> scaler = StandardScaler()
>>> X_train = scaler.fit_transform(X_train)
>>> X_test = scaler.transform(X_test)

>>> model = LinearRegression(solver='gradient_descent', learning_rate=0.1, n_iterations=1000)
>>> model.fit(X_train, y_train)
>>> print(model.score(X_test, y_test))
"""

from .model import (
    LinearRegression,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    adjusted_r2_score,
    mean_absolute_percentage_error,
)
from .preprocessing import (
    StandardScaler,
    MinMaxScaler,
    PolynomialFeatures,
    train_test_split,
    add_bias_column,
    normalize,
)
from .visualization import (
    plot_regression_line,
    plot_residuals,
    plot_cost_history,
    plot_feature_importance,
    plot_actual_vs_predicted,
    plot_full_report,
)

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    # Model
    "LinearRegression",
    # Metrics
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error",
    "r2_score",
    "adjusted_r2_score",
    "mean_absolute_percentage_error",
    # Preprocessing
    "StandardScaler",
    "MinMaxScaler",
    "PolynomialFeatures",
    "train_test_split",
    "add_bias_column",
    "normalize",
    # Visualization
    "plot_regression_line",
    "plot_residuals",
    "plot_cost_history",
    "plot_feature_importance",
    "plot_actual_vs_predicted",
    "plot_full_report",
]
