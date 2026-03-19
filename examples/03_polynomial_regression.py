"""
Example 3: Polynomial Regression
==================================
Fit a non-linear curve using polynomial feature expansion.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from linear_regression import (
    LinearRegression,
    PolynomialFeatures,
    StandardScaler,
    train_test_split,
    r2_score,
    mean_squared_error,
)


def main():
    print("=" * 55)
    print("   Example 3: Polynomial Regression")
    print("=" * 55)

    # True function: y = 0.5x³ - x² + 2x + 3
    rng = np.random.default_rng(7)
    X = rng.uniform(-4, 4, size=(200, 1))
    y = 0.5 * X.flatten() ** 3 - X.flatten() ** 2 + 2 * X.flatten() + 3 + rng.normal(0, 2, 200)

    print(f"\nTrue function: y = 0.5x³ - x² + 2x + 3 + noise")
    print(f"Dataset      : {len(y)} samples\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    for degree in [1, 2, 3, 4]:
        pf = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = pf.fit_transform(X_train)
        X_test_poly = pf.fit_transform(X_test)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_poly)
        X_test_s = scaler.transform(X_test_poly)

        model = LinearRegression(solver="normal_equation", regularization="l2", lambda_=0.001)
        model.fit(X_train_s, y_train)

        train_r2 = r2_score(y_train, model.predict(X_train_s))
        test_r2 = r2_score(y_test, model.predict(X_test_s))
        test_mse = mean_squared_error(y_test, model.predict(X_test_s))

        print(f"  Degree {degree}: Train R²={train_r2:.4f}  Test R²={test_r2:.4f}  MSE={test_mse:.4f}")

    # ── Plot best fit (degree 3) ────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pf = PolynomialFeatures(degree=3, include_bias=False)
        X_train_poly = pf.fit_transform(X_train)
        X_test_poly = pf.fit_transform(X_test)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_poly)
        X_test_s = scaler.transform(X_test_poly)
        model = LinearRegression(solver="normal_equation", regularization="l2", lambda_=0.001)
        model.fit(X_train_s, y_train)

        X_line = np.linspace(-4, 4, 300).reshape(-1, 1)
        X_line_poly = pf.fit_transform(X_line)
        X_line_s = scaler.transform(X_line_poly)
        y_line = model.predict(X_line_s)

        bg = "#0F172A"
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=bg)
        ax.set_facecolor("#1E293B")
        ax.scatter(X_train.flatten(), y_train, s=20, alpha=0.5, color="#2563EB", label="Train")
        ax.scatter(X_test.flatten(), y_test, s=20, alpha=0.7, color="#10B981", label="Test")
        ax.plot(X_line.flatten(), y_line, color="#F59E0B", lw=2.5, label="Degree-3 fit")
        ax.set_xlabel("x", color="white")
        ax.set_ylabel("y", color="white")
        ax.set_title("Polynomial Regression — Degree 3", fontweight="bold", color="white")
        ax.legend(facecolor="#1E293B", edgecolor="#475569")
        ax.tick_params(colors="#94A3B8")
        ax.grid(True, linestyle="--", alpha=0.4, color="#334155")
        fig.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "polynomial_regression.png")
        fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=bg)
        print(f"\n  Plot saved → {out}")
    except Exception as e:
        print(f"\n  (Skipped plot: {e})")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
