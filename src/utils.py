"""
utils.py — Helper functions used across the project.
"""

import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ─── Logging Setup ────────────────────────────────────────────────────────────
def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Return a nicely formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─── Timing Decorator ─────────────────────────────────────────────────────────
def timeit(func):
    """Decorator: prints how long a function took."""
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"✅  {func.__name__} finished in {elapsed:.1f}s")
        return result
    return wrapper


# ─── Plot Helpers ─────────────────────────────────────────────────────────────
def save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save a matplotlib figure with consistent settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"📊  Saved figure → {path}")


def set_plot_style() -> None:
    """Apply a clean, publication-quality plot style."""
    plt.rcParams.update({
        "figure.facecolor": "#0f1117",
        "axes.facecolor":   "#1a1d27",
        "axes.edgecolor":   "#3d3f4e",
        "axes.labelcolor":  "#e0e0e0",
        "xtick.color":      "#e0e0e0",
        "ytick.color":      "#e0e0e0",
        "text.color":       "#e0e0e0",
        "grid.color":       "#2a2d3e",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.titleweight": "bold",
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
    })


# ─── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    print(f"🌱  Random seed set to {seed}")
