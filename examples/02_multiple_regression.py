"""
Example 2: Multiple Linear Regression + Regularization
========================================================
Multi-feature dataset with L1 / L2 regularization comparison.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from linear_regression import (
    LinearRegression,
    StandardScaler,
    train_test_split,
    r2_score,
    mean_squared_error,
    root_mean_squared_error,
    plot_full_report,
)


def make_dataset(n_samples: int = 300, n_features: int = 5, noise: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    true_weights = np.array([4.0, -2.5, 1.8, 0.0, 3.2])   # one irrelevant feature (w=0)
    y = X @ true_weights + 7.0 + rng.normal(0, noise, n_samples)
    return X, y, true_weights


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n  [{name}]")
    print(f"    Weights : {np.round(model.weights_, 3)}")
    print(f"    Bias    : {model.bias_:.4f}")
    print(f"    MSE     : {mean_squared_error(y_test, y_pred):.4f}")
    print(f"    RMSE    : {root_mean_squared_error(y_test, y_pred):.4f}")
    print(f"    R²      : {r2_score(y_test, y_pred):.4f}")


def main():
    print("=" * 60)
    print("   Example 2: Multiple Regression + Regularization")
    print("=" * 60)

    X, y, true_w = make_dataset(n_samples=400, n_features=5, noise=1.5)
    print(f"\nTrue weights : {true_w}")
    print(f"Dataset      : {X.shape[0]} samples, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    feature_names = [f"x{i+1}" for i in range(X.shape[1])]

    # No regularization
    m_none = LinearRegression(solver="gradient_descent", n_iterations=2000, learning_rate=0.1)
    m_none.fit(X_train_s, y_train)
    evaluate("No Regularization", m_none, X_test_s, y_test)

    # L2 (Ridge)
    m_l2 = LinearRegression(solver="gradient_descent", n_iterations=2000,
                            learning_rate=0.1, regularization="l2", lambda_=0.5)
    m_l2.fit(X_train_s, y_train)
    evaluate("L2 Ridge  λ=0.5", m_l2, X_test_s, y_test)

    # L1 (Lasso)
    m_l1 = LinearRegression(solver="gradient_descent", n_iterations=2000,
                            learning_rate=0.05, regularization="l1", lambda_=0.1)
    m_l1.fit(X_train_s, y_train)
    evaluate("L1 Lasso  λ=0.1", m_l1, X_test_s, y_test)

    # Normal Equation (no regularization)
    m_ne = LinearRegression(solver="normal_equation")
    m_ne.fit(X_train_s, y_train)
    evaluate("Normal Equation", m_ne, X_test_s, y_test)

    # ── Diagnostic report ──────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        out = os.path.join(os.path.dirname(__file__), "multiple_regression_report.png")
        fig = plot_full_report(m_none, X_train_s, y_train, X_test_s, y_test,
                               feature_names=feature_names, save_path=out)
        print(f"\n  Report saved → {out}")
    except Exception as e:
        print(f"\n  (Skipped report: {e})")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
