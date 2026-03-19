"""
Visualization utilities for Linear Regression.
Includes plots for regression line, residuals, cost history, and feature importance.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, List


# ─── Style Defaults ────────────────────────────────────────────────────────────
PALETTE = {
    "primary": "#2563EB",
    "secondary": "#10B981",
    "accent": "#F59E0B",
    "danger": "#EF4444",
    "muted": "#94A3B8",
    "bg": "#0F172A",
    "surface": "#1E293B",
    "text": "#F8FAFC",
}

plt.rcParams.update(
    {
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": PALETTE["surface"],
        "axes.edgecolor": PALETTE["muted"],
        "axes.labelcolor": PALETTE["text"],
        "axes.titlecolor": PALETTE["text"],
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "grid.color": "#334155",
        "grid.alpha": 0.5,
        "text.color": PALETTE["text"],
        "legend.facecolor": PALETTE["surface"],
        "legend.edgecolor": PALETTE["muted"],
        "font.family": "monospace",
    }
)


# ─── Plot Functions ─────────────────────────────────────────────────────────────

def plot_regression_line(
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    feature_name: str = "X",
    target_name: str = "y",
    title: str = "Linear Regression Fit",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot actual data points and regression line (simple regression only)."""
    X = np.array(X).flatten()

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    ax.scatter(X, y, color=PALETTE["primary"], alpha=0.7, s=40, label="Actual", zorder=3)
    sort_idx = np.argsort(X)
    ax.plot(X[sort_idx], y_pred[sort_idx], color=PALETTE["accent"], lw=2.5, label="Predicted", zorder=4)

    ax.set_xlabel(feature_name)
    ax.set_ylabel(target_name)
    ax.set_title(title, pad=12, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--")

    if standalone:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot residuals vs predicted values."""
    residuals = y_true - y_pred

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    ax.scatter(y_pred, residuals, color=PALETTE["secondary"], alpha=0.7, s=40, zorder=3)
    ax.axhline(0, color=PALETTE["danger"], lw=1.5, linestyle="--", label="Zero residual")
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Predicted", pad=12, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--")

    if standalone:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    return fig


def plot_cost_history(
    cost_history: List[float],
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot gradient descent cost (loss) over iterations."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    iterations = range(1, len(cost_history) + 1)
    ax.plot(iterations, cost_history, color=PALETTE["accent"], lw=2)
    ax.fill_between(iterations, cost_history, alpha=0.15, color=PALETTE["accent"])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost (MSE)")
    ax.set_title("Gradient Descent — Cost History", pad=12, fontweight="bold")
    ax.grid(True, linestyle="--")

    if standalone:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    return fig


def plot_feature_importance(
    weights: np.ndarray,
    feature_names: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of feature coefficients (importance)."""
    n = len(weights)
    if feature_names is None:
        feature_names = [f"Feature {i + 1}" for i in range(n)]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(max(7, n * 0.8), 5))
    else:
        fig = ax.figure

    colors = [PALETTE["primary"] if w >= 0 else PALETTE["danger"] for w in weights]
    bars = ax.barh(feature_names, weights, color=colors, edgecolor="none")

    for bar, val in zip(bars, weights):
        x = bar.get_width()
        ax.text(
            x + (0.01 * max(abs(weights))),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
            color=PALETTE["text"],
        )

    ax.axvline(0, color=PALETTE["muted"], lw=1)
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Feature Coefficients", pad=12, fontweight="bold")
    ax.grid(True, linestyle="--", axis="x")

    if standalone:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot of actual vs. predicted values with perfect-prediction line."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    ax.scatter(y_true, y_pred, color=PALETTE["primary"], alpha=0.65, s=40, label="Predictions", zorder=3)

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, color=PALETTE["danger"], lw=1.5, linestyle="--", label="Perfect Prediction")

    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted", pad=12, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--")

    if standalone:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    return fig


def plot_full_report(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate a comprehensive 2×2 diagnostic report figure:
      [Cost History] [Actual vs Predicted]
      [Residuals   ] [Feature Importance ]
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    fig = plt.figure(figsize=(16, 10), facecolor=PALETTE["bg"])
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Row 0
    if model.cost_history_:
        plot_cost_history(model.cost_history_, ax=ax1)
    else:
        ax1.text(0.5, 0.5, "No cost history\n(Normal Equation)", ha="center", va="center",
                 transform=ax1.transAxes, color=PALETTE["muted"], fontsize=12)
        ax1.set_title("Cost History", fontweight="bold")

    plot_actual_vs_predicted(y_test, y_pred_test, ax=ax2)
    ax2.set_title("Actual vs Predicted (Test Set)", fontweight="bold")

    # Row 1
    plot_residuals(y_test, y_pred_test, ax=ax3)

    if feature_names is None and hasattr(model, "n_features_"):
        feature_names = [f"Feature {i + 1}" for i in range(model.n_features_)]

    plot_feature_importance(model.weights_, feature_names=feature_names, ax=ax4)

    fig.suptitle("Linear Regression — Diagnostic Report", fontsize=16, fontweight="bold", y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])

    return fig
