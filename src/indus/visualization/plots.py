"""Visualization functions for training curves, metrics, and diagnostics.

All functions save figures to disk and return the figure path for artifact logging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from indus.logging_utils import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    }
)


def _ema(values: list[float], alpha: float = 0.05) -> list[float]:
    """Compute exponential moving average."""
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


def plot_training_loss(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    output_path: str | Path = "reports/figures/training_loss.png",
    title: str = "Training Loss Curve",
) -> Path:
    """Plot training and optionally validation loss over steps."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    steps = list(range(1, len(train_losses) + 1))
    ax.plot(steps, train_losses, label="Train Loss", alpha=0.7, linewidth=1.0)

    if len(train_losses) > 20:
        smoothed = _ema(train_losses, alpha=0.05)
        ax.plot(steps, smoothed, label="Train Loss (smoothed)", linewidth=2.0)

    if val_losses:
        val_steps = np.linspace(1, len(train_losses), len(val_losses)).astype(int)
        ax.plot(val_steps, val_losses, "o-", label="Val Loss", markersize=5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Training loss plot saved to %s", output_path)
    return output_path


def plot_gradient_norms(
    gradient_norms: list[float],
    output_path: str | Path = "reports/figures/gradient_norms.png",
    title: str = "Gradient Norm Trend",
) -> Path:
    """Plot L2 gradient norm over training steps."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    steps = list(range(1, len(gradient_norms) + 1))
    ax.plot(steps, gradient_norms, alpha=0.5, linewidth=0.8, color="tab:orange")

    if len(gradient_norms) > 20:
        smoothed = _ema(gradient_norms, alpha=0.05)
        ax.plot(steps, smoothed, linewidth=2.0, color="tab:red", label="EMA")
        ax.legend()

    ax.set_xlabel("Training Step")
    ax.set_ylabel("L2 Gradient Norm")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Gradient norm plot saved to %s", output_path)
    return output_path


def plot_eval_comparison(
    baseline_metrics: dict[str, float],
    finetuned_metrics: dict[str, float],
    output_path: str | Path = "reports/figures/eval_comparison.png",
    title: str = "Baseline vs Fine-tuned Metrics",
) -> Path:
    """Plot side-by-side comparison of baseline and fine-tuned metrics."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display_keys = ["rougeL_fmeasure", "bleu", "bertscore_f1"]
    display_labels = ["ROUGE-L F1", "BLEU", "BERTScore F1"]

    baseline_vals = [baseline_metrics.get(k, 0.0) for k in display_keys]
    finetuned_vals = [finetuned_metrics.get(k, 0.0) for k in display_keys]

    x = np.arange(len(display_labels))
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color="tab:gray")
    bars2 = ax.bar(x + width / 2, finetuned_vals, width, label="Fine-tuned", color="tab:blue")

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.legend()
    ax.set_ylim(0, 1.0)

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Evaluation comparison plot saved to %s", output_path)
    return output_path


def plot_adapter_convergence(
    train_losses: list[float],
    gradient_norms: list[float],
    output_path: str | Path = "reports/figures/adapter_convergence.png",
    title: str = "Adapter Convergence Analysis",
) -> Path:
    """Plot loss and gradient norm on dual y-axes as a convergence proxy."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots()
    steps = list(range(1, len(train_losses) + 1))

    color1 = "tab:blue"
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss", color=color1)
    if len(train_losses) > 20:
        smoothed_loss = _ema(train_losses, alpha=0.05)
        ax1.plot(steps, smoothed_loss, color=color1, linewidth=2.0, label="Loss (EMA)")
    else:
        ax1.plot(steps, train_losses, color=color1, linewidth=1.5, label="Loss")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("Gradient Norm", color=color2)
    gn_steps = steps[: len(gradient_norms)]
    if len(gradient_norms) > 20:
        smoothed_gn = _ema(gradient_norms, alpha=0.05)
        ax2.plot(gn_steps, smoothed_gn, color=color2, linewidth=2.0, label="Grad Norm (EMA)")
    else:
        ax2.plot(gn_steps, gradient_norms, color=color2, linewidth=1.5, label="Grad Norm")
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(title)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Adapter convergence plot saved to %s", output_path)
    return output_path


def generate_all_plots(
    train_state: Any,
    baseline_metrics: dict[str, float] | None = None,
    finetuned_metrics: dict[str, float] | None = None,
    output_dir: str | Path = "reports/figures",
) -> list[Path]:
    """Generate all standard analysis plots from a training run."""
    output_dir = Path(output_dir)
    paths: list[Path] = []

    if train_state.train_losses:
        paths.append(
            plot_training_loss(
                train_state.train_losses,
                train_state.val_losses,
                output_path=output_dir / "training_loss.png",
            )
        )
    if train_state.gradient_norms:
        paths.append(
            plot_gradient_norms(
                train_state.gradient_norms,
                output_path=output_dir / "gradient_norms.png",
            )
        )
    if train_state.train_losses and train_state.gradient_norms:
        paths.append(
            plot_adapter_convergence(
                train_state.train_losses,
                train_state.gradient_norms,
                output_path=output_dir / "adapter_convergence.png",
            )
        )
    if baseline_metrics and finetuned_metrics:
        paths.append(
            plot_eval_comparison(
                baseline_metrics,
                finetuned_metrics,
                output_path=output_dir / "eval_comparison.png",
            )
        )

    logger.info("Generated %d analysis plots in %s", len(paths), output_dir)
    return paths
