# 📈 Linear Regression from Scratch

[![CI](https://github.com/yourusername/linear-regression/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/linear-regression/actions)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete, production-quality **Linear Regression** implementation built from scratch using **only NumPy** — no scikit-learn. Designed to be clear, well-tested, and mathematically rigorous.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Solvers** | Gradient Descent & Normal Equation |
| **Regularization** | L1 (Lasso), L2 (Ridge) |
| **Metrics** | MSE, MAE, RMSE, R², Adjusted R², MAPE |
| **Preprocessing** | StandardScaler, MinMaxScaler, PolynomialFeatures, train_test_split |
| **Visualization** | Regression fit, residuals, cost history, feature importance, full diagnostic report |
| **Tests** | Full pytest suite with 30+ unit tests |
| **CI/CD** | GitHub Actions across Python 3.8 – 3.11 |

---

## 🗂️ Project Structure

```
linear-regression/
├── linear_regression/
│   ├── __init__.py          # Public API exports
│   ├── model.py             # LinearRegression + all metrics
│   ├── preprocessing.py     # Scalers, PolynomialFeatures, train_test_split
│   └── visualization.py     # Matplotlib plotting utilities
├── tests/
│   └── test_linear_regression.py   # pytest unit tests (30+ tests)
├── examples/
│   ├── 01_simple_regression.py     # Simple 1D regression
│   ├── 02_multiple_regression.py   # Multiple features + regularization
│   └── 03_polynomial_regression.py # Non-linear curve fitting
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI pipeline
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/linear-regression.git
cd linear-regression
pip install -r requirements.txt
```

### Basic Usage

```python
import numpy as np
from linear_regression import LinearRegression, StandardScaler, train_test_split

# Generate synthetic data
rng = np.random.default_rng(42)
X = rng.uniform(-5, 5, (200, 3))
y = 3 * X[:, 0] - 1.5 * X[:, 1] + 2 * X[:, 2] + rng.normal(0, 0.5, 200)

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Train
model = LinearRegression(
    solver='gradient_descent',
    learning_rate=0.1,
    n_iterations=1000,
)
model.fit(X_train, y_train)

# Evaluate
print(f"R²   : {model.score(X_test, y_test):.4f}")
print(f"Weights: {model.weights_}")
print(f"Bias   : {model.bias_:.4f}")
```

---

## 📐 Math Behind It

### Model
```
ŷ = Xw + b
```

### Cost Function (MSE)
```
J(w, b) = (1 / 2m) Σ (ŷᵢ - yᵢ)²
```

### Gradient Descent Update
```
w := w - α · (1/m) Xᵀ(ŷ - y)
b := b - α · (1/m) Σ(ŷ - y)
```

### Normal Equation (closed-form)
```
θ = (XᵀX)⁻¹ Xᵀy
```

### L2 Regularization (Ridge)
```
J(w, b) = MSE + (λ / 2m) Σ wⱼ²
```

### L1 Regularization (Lasso)
```
J(w, b) = MSE + (λ / m) Σ |wⱼ|
```

---

## ⚙️ API Reference

### `LinearRegression`

```python
LinearRegression(
    solver='gradient_descent',   # 'gradient_descent' | 'normal_equation'
    learning_rate=0.01,          # Step size for GD
    n_iterations=1000,           # Max GD iterations
    fit_intercept=True,          # Learn bias term?
    regularization=None,         # None | 'l1' | 'l2'
    lambda_=0.01,                # Regularization strength
    tolerance=1e-6,              # GD convergence threshold
    verbose=False,               # Print training progress
)
```

| Method | Description |
|---|---|
| `.fit(X, y)` | Train the model |
| `.predict(X)` | Predict target values |
| `.score(X, y)` | Compute R² score |
| `.get_params()` | Return learned weights and bias |

**Fitted attributes:** `weights_`, `bias_`, `cost_history_`, `n_features_`, `n_samples_`

---

### Metrics

```python
from linear_regression import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    adjusted_r2_score,
    mean_absolute_percentage_error,
)
```

### Preprocessing

```python
from linear_regression import (
    StandardScaler,        # z = (x - μ) / σ
    MinMaxScaler,          # x_scaled ∈ [a, b]
    PolynomialFeatures,    # x → [1, x, x², ...]
    train_test_split,      # random train/test split
)
```

### Visualization

```python
from linear_regression import (
    plot_regression_line,      # Fit line on 1D data
    plot_residuals,            # Residuals vs Predicted
    plot_cost_history,         # GD loss curve
    plot_feature_importance,   # Coefficient bar chart
    plot_actual_vs_predicted,  # Actual vs Predicted scatter
    plot_full_report,          # 2×2 diagnostic figure
)
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=linear_regression --cov-report=term-missing

# Run a specific test class
pytest tests/test_linear_regression.py::TestLinearRegressionGradientDescent -v
```

---

## 📊 Examples

```bash
# Simple 1D linear regression
python examples/01_simple_regression.py

# Multiple features + L1/L2 regularization
python examples/02_multiple_regression.py

# Polynomial curve fitting (degree comparison)
python examples/03_polynomial_regression.py
```

---

## 🔑 Key Design Decisions

- **Pure NumPy** — no hidden ML libraries; every operation is explicit.
- **Two solvers** — Gradient Descent for large datasets; Normal Equation for exact closed-form solutions on small ones.
- **Regularization on weights only** — bias is never penalized (standard practice).
- **Convergence check** — GD stops early when cost change < `tolerance`.
- **Stable Normal Equation** — uses `np.linalg.solve` with a `pinv` fallback for singular matrices.
- **Vectorized operations** — all forward/backward passes use matrix math for speed.

---

## 🤝 Contributing

1. Fork the repo
2. Create your branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please make sure all tests pass before submitting a PR.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
