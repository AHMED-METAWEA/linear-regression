"""
Linear Regression implementation from scratch using NumPy.
Supports Simple and Multiple Linear Regression with Gradient Descent
and Normal Equation solvers.
"""

import numpy as np
from typing import Optional, Literal


class LinearRegression:
    """
    Linear Regression from scratch.

    Supports two solvers:
        - 'gradient_descent': Iterative optimization using GD
        - 'normal_equation': Closed-form analytical solution

    Parameters
    ----------
    solver : str, optional
        Optimization method: 'gradient_descent' or 'normal_equation'.
        Default is 'gradient_descent'.
    learning_rate : float, optional
        Step size for gradient descent. Default is 0.01.
    n_iterations : int, optional
        Number of gradient descent iterations. Default is 1000.
    fit_intercept : bool, optional
        Whether to add a bias term. Default is True.
    regularization : str or None, optional
        Regularization type: 'l1', 'l2', or None. Default is None.
    lambda_ : float, optional
        Regularization strength. Default is 0.01.
    tolerance : float, optional
        Convergence tolerance for gradient descent. Default is 1e-6.
    verbose : bool, optional
        Print training progress. Default is False.
    """

    def __init__(
        self,
        solver: Literal["gradient_descent", "normal_equation"] = "gradient_descent",
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        fit_intercept: bool = True,
        regularization: Optional[Literal["l1", "l2"]] = None,
        lambda_: float = 0.01,
        tolerance: float = 1e-6,
        verbose: bool = False,
    ):
        self.solver = solver
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.lambda_ = lambda_
        self.tolerance = tolerance
        self.verbose = verbose

        # Learned parameters
        self.weights_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.cost_history_: list = []
        self.n_features_: int = 0
        self.n_samples_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Fit the model to training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = self._validate_inputs(X, y)
        self.n_samples_, self.n_features_ = X.shape

        # Initialize weights
        self.weights_ = np.zeros(self.n_features_)
        self.bias_ = 0.0
        self.cost_history_ = []

        if self.solver == "gradient_descent":
            self._gradient_descent(X, y)
        elif self.solver == "normal_equation":
            self._normal_equation(X, y)
        else:
            raise ValueError(f"Unknown solver: '{self.solver}'. Choose 'gradient_descent' or 'normal_equation'.")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        self._check_is_fitted()
        X = self._validate_X(X)
        return self._linear_model(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score (coefficient of determination).

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray

        Returns
        -------
        float
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def get_params(self) -> dict:
        """Return model parameters as a dictionary."""
        self._check_is_fitted()
        return {
            "weights": self.weights_.copy(),
            "bias": self.bias_,
            "n_features": self.n_features_,
            "n_samples_trained": self.n_samples_,
        }

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------

    def _gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """Iterative gradient descent optimizer."""
        prev_cost = np.inf

        for iteration in range(self.n_iterations):
            y_pred = self._linear_model(X)
            cost = self._compute_cost(y, y_pred)
            self.cost_history_.append(cost)

            # Compute gradients
            error = y_pred - y
            dw = (1 / self.n_samples_) * X.T @ error
            db = (1 / self.n_samples_) * np.sum(error)

            # Apply regularization to weight gradient (not bias)
            if self.regularization == "l2":
                dw += (self.lambda_ / self.n_samples_) * self.weights_
            elif self.regularization == "l1":
                dw += (self.lambda_ / self.n_samples_) * np.sign(self.weights_)

            # Update parameters
            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

            # Convergence check
            if abs(prev_cost - cost) < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}, cost={cost:.6f}")
                break

            prev_cost = cost

            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations} — Cost: {cost:.6f}")

    def _normal_equation(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Closed-form Normal Equation: θ = (XᵀX)⁻¹ Xᵀy

        For L2 regularization: θ = (XᵀX + λI)⁻¹ Xᵀy
        """
        if self.fit_intercept:
            X_b = np.column_stack([np.ones(self.n_samples_), X])
        else:
            X_b = X

        n_params = X_b.shape[1]

        if self.regularization == "l2":
            # Ridge regression closed form
            I = np.eye(n_params)
            if self.fit_intercept:
                I[0, 0] = 0  # Don't regularize bias
            matrix = X_b.T @ X_b + self.lambda_ * I
        elif self.regularization == "l1":
            # L1 has no closed form — fall back to pseudo-inverse
            matrix = X_b.T @ X_b
        else:
            matrix = X_b.T @ X_b

        try:
            theta = np.linalg.solve(matrix, X_b.T @ y)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse for singular matrices
            theta = np.linalg.pinv(matrix) @ X_b.T @ y

        if self.fit_intercept:
            self.bias_ = theta[0]
            self.weights_ = theta[1:]
        else:
            self.weights_ = theta

        # Record final cost
        y_pred = self._linear_model(X)
        self.cost_history_.append(self._compute_cost(y, y_pred))

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _linear_model(self, X: np.ndarray) -> np.ndarray:
        """Compute ŷ = Xw + b."""
        return X @ self.weights_ + self.bias_

    def _compute_cost(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error cost with optional regularization.

        J(w, b) = (1/2m) Σ(ŷᵢ - yᵢ)² + regularization_term
        """
        m = len(y)
        mse = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)

        if self.regularization == "l2":
            reg_term = (self.lambda_ / (2 * m)) * np.sum(self.weights_ ** 2)
            return mse + reg_term
        elif self.regularization == "l1":
            reg_term = (self.lambda_ / m) * np.sum(np.abs(self.weights_))
            return mse + reg_term

        return mse

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray):
        """Validate and reshape inputs."""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}"
            )

        return X, y

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        """Validate prediction input."""
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}"
            )
        return X

    def _check_is_fitted(self) -> None:
        """Raise error if model is not fitted."""
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

    def __repr__(self) -> str:
        return (
            f"LinearRegression("
            f"solver='{self.solver}', "
            f"learning_rate={self.learning_rate}, "
            f"n_iterations={self.n_iterations}, "
            f"regularization={self.regularization!r})"
        )


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error: MSE = (1/n) Σ(ŷ - y)²"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean((y_pred - y_true) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error: MAE = (1/n) Σ|ŷ - y|"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs(y_pred - y_true)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error: RMSE = √MSE"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² Score (Coefficient of Determination).

    R² = 1 - SS_res / SS_tot
    Perfect predictions → R² = 1.0
    Baseline (mean) model → R² = 0.0
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


def adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """
    Adjusted R² penalizes adding non-informative features.

    Adj R² = 1 - (1 - R²)(n - 1) / (n - p - 1)
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    if n - n_features - 1 <= 0:
        return float("nan")
    return float(1 - (1 - r2) * (n - 1) / (n - n_features - 1))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE = (100/n) Σ|y - ŷ| / |y|  (excludes zero y values)"""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
