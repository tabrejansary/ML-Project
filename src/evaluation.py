"""
evaluation.py — Comprehensive metrics, confusion matrix, and plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    matthews_corrcoef, classification_report,
    RocCurveDisplay,
)

from src.utils import set_plot_style, save_figure, get_logger
from src.config import FIGURES_DIR, CLASS_NAMES

logger = get_logger(__name__)


def compute_all_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_prob: Optional[np.ndarray] = None,
                        model_name: str = "Model") -> dict:
    """
    Compute the full set of evaluation metrics.

    Returns
    -------
    dict with keys: model, accuracy, precision, recall, specificity,
                    f1, roc_auc, mcc
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan

    return {
        "model":       model_name,
        "accuracy":    round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision":   round(precision_score(y_true, y_pred) * 100, 2),
        "recall":      round(recall_score(y_true, y_pred) * 100, 2),
        "specificity": round(specificity * 100, 2),
        "f1":          round(f1_score(y_true, y_pred) * 100, 2),
        "roc_auc":     round(auc * 100, 2) if not np.isnan(auc) else np.nan,
        "mcc":         round(matthews_corrcoef(y_true, y_pred), 4),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot a styled confusion matrix."""
    set_plot_style()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0f1117")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, linecolor="#3d3f4e",
        ax=ax, cbar=False,
        annot_kws={"size": 16, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_roc_curves(
    models_results: List[dict],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot ROC curves for multiple models on one axes.

    models_results : list of dicts, each with keys:
        name, y_true, y_prob
    """
    set_plot_style()
    colors = ["#7c3aed", "#2563eb", "#059669", "#d97706",
              "#dc2626", "#db2777"]
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f1117")

    for i, res in enumerate(models_results):
        RocCurveDisplay.from_predictions(
            res["y_true"], res["y_prob"],
            name=res["name"],
            ax=ax, color=colors[i % len(colors)],
        )
    ax.plot([0, 1], [0, 1], "w--", alpha=0.4, label="Random Classifier")
    ax.set_title("ROC Curves — All Classifiers", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def print_results_table(results: List[dict]) -> pd.DataFrame:
    """Print a formatted results table and return as DataFrame."""
    df = pd.DataFrame(results)
    cols = ["model", "accuracy", "precision", "recall",
            "specificity", "f1", "roc_auc", "mcc"]
    df = df[[c for c in cols if c in df.columns]]
    print("\n" + "="*80)
    print("📊  RESULTS TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    return df
