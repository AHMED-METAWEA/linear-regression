"""
Example 1: Simple Linear Regression
=====================================
Fit y = 3x + 5 with gradient descent and normal equation.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from linear_regression import (
    LinearRegression,
    StandardScaler,
    train_test_split,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
    plot_regression_line,
    plot_cost_history,
)


def main():
    print("=" * 55)
    print("   Example 1: Simple Linear Regression")
    print("=" * 55)

    # ── 1. Generate synthetic data ──────────────────────────────
    rng = np.random.default_rng(42)
    X = rng.uniform(-5, 5, size=(150, 1))
    y = 3 * X.flatten() + 5 + rng.normal(0, 1.5, 150)

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} feature")

    # ── 2. Train / test split ───────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 3. Scale features ───────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── 4a. Gradient Descent ────────────────────────────────────
    print("\n[Gradient Descent]")
    gd_model = LinearRegression(
        solver="gradient_descent",
        learning_rate=0.1,
        n_iterations=1000,
        verbose=False,
    )
    gd_model.fit(X_train_s, y_train)

    y_pred_gd = gd_model.predict(X_test_s)
    print(f"  Weight  : {gd_model.weights_[0]:.4f}")
    print(f"  Bias    : {gd_model.bias_:.4f}")
    print(f"  MSE     : {mean_squared_error(y_test, y_pred_gd):.4f}")
    print(f"  RMSE    : {root_mean_squared_error(y_test, y_pred_gd):.4f}")
    print(f"  R²      : {r2_score(y_test, y_pred_gd):.4f}")

    # ── 4b. Normal Equation ─────────────────────────────────────
    print("\n[Normal Equation]")
    ne_model = LinearRegression(solver="normal_equation")
    ne_model.fit(X_train_s, y_train)

    y_pred_ne = ne_model.predict(X_test_s)
    print(f"  Weight  : {ne_model.weights_[0]:.4f}")
    print(f"  Bias    : {ne_model.bias_:.4f}")
    print(f"  MSE     : {mean_squared_error(y_test, y_pred_ne):.4f}")
    print(f"  RMSE    : {root_mean_squared_error(y_test, y_pred_ne):.4f}")
    print(f"  R²      : {r2_score(y_test, y_pred_ne):.4f}")

    # ── 5. Plots ────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_regression_line(X_test_s, y_test, y_pred_gd, ax=axes[0], title="GD: Regression Fit")
        plot_cost_history(gd_model.cost_history_, ax=axes[1])
        fig.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "simple_regression.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"\n  Plot saved → {out}")
    except Exception as e:
        print(f"  (Skipped plot: {e})")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
