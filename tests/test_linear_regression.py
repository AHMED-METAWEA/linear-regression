"""
Unit tests for the LinearRegression implementation.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from linear_regression import (
    LinearRegression,
    StandardScaler,
    MinMaxScaler,
    PolynomialFeatures,
    train_test_split,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    adjusted_r2_score,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_data():
    """Simple 1D linear data: y = 2x + 1"""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(100, 1))
    y = 2 * X.flatten() + 1 + rng.normal(0, 0.5, 100)
    return X, y


@pytest.fixture
def multi_data():
    """Multi-feature linear data: y = 3x1 + 1.5x2 - 2x3"""
    rng = np.random.default_rng(42)
    X = rng.uniform(-3, 3, size=(200, 3))
    y = 3 * X[:, 0] + 1.5 * X[:, 1] - 2 * X[:, 2] + rng.normal(0, 0.2, 200)
    return X, y


@pytest.fixture
def perfect_data():
    """Noise-free data for exact fit tests."""
    X = np.arange(1, 11).reshape(-1, 1).astype(float)
    y = (5 * X.flatten() - 3).astype(float)
    return X, y


# ─── Model Tests ─────────────────────────────────────────────────────────────

class TestLinearRegressionGradientDescent:

    def test_fit_returns_self(self, simple_data):
        X, y = simple_data
        model = LinearRegression(solver="gradient_descent", n_iterations=500, learning_rate=0.05)
        result = model.fit(X, y)
        assert result is model

    def test_simple_regression_weights(self, simple_data):
        X, y = simple_data
        model = LinearRegression(solver="gradient_descent", n_iterations=2000, learning_rate=0.01)
        model.fit(X, y)
        # slope ≈ 2, intercept ≈ 1
        assert abs(model.weights_[0] - 2.0) < 0.3
        assert abs(model.bias_ - 1.0) < 0.5

    def test_r2_high_on_clean_data(self, simple_data):
        X, y = simple_data
        model = LinearRegression(solver="gradient_descent", n_iterations=2000, learning_rate=0.01)
        model.fit(X, y)
        r2 = model.score(X, y)
        assert r2 > 0.95

    def test_multiple_regression(self, multi_data):
        X, y = multi_data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression(solver="gradient_descent", n_iterations=3000, learning_rate=0.1)
        model.fit(X_scaled, y)
        r2 = model.score(X_scaled, y)
        assert r2 > 0.98

    def test_cost_decreases(self, simple_data):
        X, y = simple_data
        model = LinearRegression(solver="gradient_descent", n_iterations=500, learning_rate=0.01)
        model.fit(X, y)
        costs = model.cost_history_
        assert costs[0] > costs[-1], "Cost should decrease over training"

    def test_predict_shape(self, simple_data):
        X, y = simple_data
        model = LinearRegression(solver="gradient_descent", n_iterations=100, learning_rate=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_1d_input_accepted(self):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        model = LinearRegression(solver="gradient_descent", n_iterations=5000, learning_rate=0.05)
        model.fit(X, y)
        assert abs(model.weights_[0] - 2.0) < 0.2

    def test_l2_regularization(self, multi_data):
        X, y = multi_data
        model = LinearRegression(
            solver="gradient_descent", n_iterations=1000, learning_rate=0.1,
            regularization="l2", lambda_=0.1
        )
        model.fit(X, y)
        r2 = model.score(X, y)
        assert r2 > 0.90

    def test_l1_regularization(self, multi_data):
        X, y = multi_data
        model = LinearRegression(
            solver="gradient_descent", n_iterations=1000, learning_rate=0.05,
            regularization="l1", lambda_=0.01
        )
        model.fit(X, y)
        r2 = model.score(X, y)
        assert r2 > 0.85

    def test_not_fitted_raises(self):
        model = LinearRegression()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(np.array([[1.0]]))

    def test_wrong_feature_count_raises(self, simple_data):
        X, y = simple_data
        model = LinearRegression(n_iterations=100)
        model.fit(X, y)
        with pytest.raises(ValueError):
            model.predict(np.array([[1.0, 2.0]]))  # 2 features, trained on 1

    def test_invalid_solver_raises(self, simple_data):
        X, y = simple_data
        model = LinearRegression(solver="magic")
        with pytest.raises(ValueError, match="Unknown solver"):
            model.fit(X, y)


class TestLinearRegressionNormalEquation:

    def test_exact_fit_no_noise(self, perfect_data):
        X, y = perfect_data
        model = LinearRegression(solver="normal_equation")
        model.fit(X, y)
        assert abs(model.weights_[0] - 5.0) < 1e-6
        assert abs(model.bias_ - (-3.0)) < 1e-6

    def test_r2_near_perfect(self, simple_data):
        X, y = simple_data
        model = LinearRegression(solver="normal_equation")
        model.fit(X, y)
        assert model.score(X, y) > 0.95

    def test_multiple_features(self, multi_data):
        X, y = multi_data
        model = LinearRegression(solver="normal_equation")
        model.fit(X, y)
        assert model.score(X, y) > 0.98

    def test_ridge_normal_equation(self, multi_data):
        X, y = multi_data
        model = LinearRegression(solver="normal_equation", regularization="l2", lambda_=1.0)
        model.fit(X, y)
        assert model.score(X, y) > 0.90

    def test_get_params(self, simple_data):
        X, y = simple_data
        model = LinearRegression(solver="normal_equation")
        model.fit(X, y)
        params = model.get_params()
        assert "weights" in params
        assert "bias" in params
        assert params["n_features"] == 1


# ─── Preprocessing Tests ─────────────────────────────────────────────────────

class TestStandardScaler:

    def test_zero_mean_unit_std(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)

    def test_inverse_transform(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)
        assert np.allclose(X_recovered, X, atol=1e-10)

    def test_not_fitted_raises(self):
        scaler = StandardScaler()
        with pytest.raises(RuntimeError, match="not fitted"):
            scaler.transform(np.array([[1.0]]))


class TestMinMaxScaler:

    def test_range_0_1(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        assert np.isclose(X_scaled.min(), 0.0)
        assert np.isclose(X_scaled.max(), 1.0)

    def test_custom_range(self):
        X = np.array([[0.0], [5.0], [10.0]])
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        assert np.isclose(X_scaled.min(), -1.0)
        assert np.isclose(X_scaled.max(), 1.0)


class TestPolynomialFeatures:

    def test_degree1_identity(self):
        X = np.array([[2.0, 3.0]])
        pf = PolynomialFeatures(degree=1, include_bias=False)
        X_poly = pf.fit_transform(X)
        assert np.allclose(X_poly, X)

    def test_degree2_shape(self):
        # 1 feature, degree 2, with bias → [1, x, x²] = 3 columns
        X = np.array([[1.0], [2.0], [3.0]])
        pf = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = pf.fit_transform(X)
        assert X_poly.shape == (3, 3)

    def test_degree2_values(self):
        X = np.array([[3.0]])
        pf = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = pf.fit_transform(X)
        expected = np.array([[1.0, 3.0, 9.0]])
        assert np.allclose(X_poly, expected)


class TestTrainTestSplit:

    def test_sizes(self):
        X = np.arange(100).reshape(-1, 1).astype(float)
        y = np.arange(100, dtype=float)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_reproducibility(self):
        X = np.arange(50).reshape(-1, 1).astype(float)
        y = np.arange(50, dtype=float)
        split1 = train_test_split(X, y, test_size=0.2, random_state=7)
        split2 = train_test_split(X, y, test_size=0.2, random_state=7)
        assert np.array_equal(split1[0], split2[0])

    def test_invalid_test_size(self):
        X, y = np.zeros((10, 1)), np.zeros(10)
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size=1.5)


# ─── Metrics Tests ───────────────────────────────────────────────────────────

class TestMetrics:

    def test_mse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_squared_error(y, y) == pytest.approx(0.0)

    def test_mae_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_absolute_error(y, y) == pytest.approx(0.0)

    def test_rmse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])  # error of 1 on last
        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        assert rmse == pytest.approx(np.sqrt(mse))

    def test_r2_perfect(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert r2_score(y, y) == pytest.approx(1.0)

    def test_r2_baseline(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y_true, np.mean(y_true))  # predicting mean always
        assert r2_score(y_true, y_pred) == pytest.approx(0.0)

    def test_adjusted_r2(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert adjusted_r2_score(y, y, n_features=1) == pytest.approx(1.0)
