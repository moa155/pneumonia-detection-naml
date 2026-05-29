"""Visualization and plotting utilities for pneumonia detection results.

Generates all figures needed for the LaTeX report and presentation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import torch

# Consistent style
sns.set_theme(style="whitegrid", font_scale=1.2)
MODEL_COLORS = {
    "fcos": "#2196F3",
    "fcos_paper": "#0D47A1",
    "retinanet": "#FF9800",
    "faster_rcnn": "#4CAF50",
    "ensemble": "#9C27B0",
}
MODEL_LABELS = {
    "fcos": "FCOS",
    "fcos_paper": "FCOS (paper SGD)",
    "retinanet": "RetinaNet",
    "faster_rcnn": "Faster R-CNN",
    "ensemble": "Ensemble (FCOS+RetinaNet, WBF)",
}


def _save(fig, path: Path, dpi: int = 150, close: bool = True):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    print(f"  Saved: {path}")


# -----------------------------------------------------------------------
# 1. Training loss curves
# -----------------------------------------------------------------------

def plot_training_losses(histories: Dict[str, Dict], output_dir: Path):
    """Plot training loss vs. epoch for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, hist in histories.items():
        losses = [ep["total_loss"] for ep in hist["train_losses"]]
        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, label=MODEL_LABELS.get(name, name),
                color=MODEL_COLORS.get(name, None), linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Training Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir / "training_loss.png", close=False)
    _save(fig, output_dir / "training_loss.pdf")


# -----------------------------------------------------------------------
# 2. Learning rate schedule
# -----------------------------------------------------------------------

def plot_learning_rates(histories: Dict[str, Dict], output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, hist in histories.items():
        lrs = hist.get("learning_rates", [])
        if lrs:
            ax.plot(range(1, len(lrs) + 1), lrs, label=MODEL_LABELS.get(name, name),
                    color=MODEL_COLORS.get(name, None), linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    ax.set_yscale("log")
    _save(fig, output_dir / "learning_rate.png")


# -----------------------------------------------------------------------
# 3. AP comparison bar chart (paper Table 2/3 style)
# -----------------------------------------------------------------------

def plot_ap_comparison(all_metrics: Dict[str, Dict], output_dir: Path):
    """Grouped bar chart of AP metrics across models."""
    metrics_keys = ["AP@0.5", "AP@0.5:0.95", "AP_M", "AP_L"]
    labels = ["AP@0.5", "AP@[.5:.95]", r"$AP_M$", r"$AP_L$"]
    model_names = list(all_metrics.keys())
    n_metrics = len(metrics_keys)
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    for i, name in enumerate(model_names):
        vals = [all_metrics[name].get(k, 0) * 100 for k in metrics_keys]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=MODEL_LABELS.get(name, name),
                      color=MODEL_COLORS.get(name, None), edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_title("Average Precision Comparison")
    ax.legend()
    # Auto-scale y-axis based on data with padding
    all_vals = []
    for name in model_names:
        all_vals.extend([all_metrics[name].get(k, 0) * 100 for k in metrics_keys])
    max_val = max(all_vals) if all_vals else 1
    ax.set_ylim(0, max(max_val * 1.4, 1))
    _save(fig, output_dir / "ap_comparison.png", close=False)
    _save(fig, output_dir / "ap_comparison.pdf")


# -----------------------------------------------------------------------
# 4. AR comparison bar chart
# -----------------------------------------------------------------------

def plot_ar_comparison(all_metrics: Dict[str, Dict], output_dir: Path):
    metrics_keys = ["AR@10", "AR_M", "AR_L"]
    labels = [r"$AR_{10}$", r"$AR_M$", r"$AR_L$"]
    model_names = list(all_metrics.keys())
    n_metrics = len(metrics_keys)
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    for i, name in enumerate(model_names):
        vals = [all_metrics[name].get(k, 0) * 100 for k in metrics_keys]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=MODEL_LABELS.get(name, name),
                      color=MODEL_COLORS.get(name, None), edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_title("Average Recall Comparison")
    ax.legend()
    # Auto-scale y-axis based on data with padding
    all_vals = []
    for name in model_names:
        all_vals.extend([all_metrics[name].get(k, 0) * 100 for k in metrics_keys])
    max_val = max(all_vals) if all_vals else 1
    ax.set_ylim(0, max(max_val * 1.4, 1))
    _save(fig, output_dir / "ar_comparison.png", close=False)
    _save(fig, output_dir / "ar_comparison.pdf")


# -----------------------------------------------------------------------
# 5. Precision–Recall curves
# -----------------------------------------------------------------------

def plot_pr_curves(all_metrics: Dict[str, Dict], output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 8))

    for name, metrics in all_metrics.items():
        prec = metrics.get("precisions", [])
        rec = metrics.get("recalls", [])
        if prec and rec:
            ap = metrics.get("AP@0.5", 0)
            ax.plot(rec, prec,
                    label=f"{MODEL_LABELS.get(name, name)} (AP={ap*100:.1f})",
                    color=MODEL_COLORS.get(name, None), linewidth=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve (IoU=0.5)")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, output_dir / "pr_curve.png", close=False)
    _save(fig, output_dir / "pr_curve.pdf")


# -----------------------------------------------------------------------
# 6. AP across IoU thresholds
# -----------------------------------------------------------------------

def plot_ap_vs_iou(
    predictions_by_model: Dict[str, List[Dict]],
    targets: List[Dict],
    output_dir: Path,
):
    """AP as a function of IoU threshold for each model."""
    from src.evaluate import compute_ap_at_iou

    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, preds in predictions_by_model.items():
        aps = []
        for iou_t in iou_thresholds:
            result = compute_ap_at_iou(preds, targets, iou_threshold=iou_t)
            aps.append(result["AP"] * 100)
        ax.plot(iou_thresholds, aps, "o-",
                label=MODEL_LABELS.get(name, name),
                color=MODEL_COLORS.get(name, None), linewidth=2)

    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("AP (%)")
    ax.set_title("AP vs. IoU Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir / "ap_vs_iou.png", close=False)
    _save(fig, output_dir / "ap_vs_iou.pdf")


# -----------------------------------------------------------------------
# 7. Validation AP over epochs
# -----------------------------------------------------------------------

def plot_val_ap_over_epochs(histories: Dict[str, Dict], output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, hist in histories.items():
        val_metrics = hist.get("val_metrics", [])
        # Filter out None entries (skipped validation epochs)
        epochs_with_val = []
        aps = []
        for i, m in enumerate(val_metrics):
            if m is not None:
                epochs_with_val.append(i + 1)
                aps.append(m.get("AP@0.5", 0) * 100)
        if aps:
            ax.plot(epochs_with_val, aps,
                    label=MODEL_LABELS.get(name, name),
                    color=MODEL_COLORS.get(name, None), linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP@0.5 (%)")
    ax.set_title("Validation AP@0.5 Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir / "val_ap_over_epochs.png", close=False)
    _save(fig, output_dir / "val_ap_over_epochs.pdf")


# -----------------------------------------------------------------------
# 8. Patient-level classification metrics
# -----------------------------------------------------------------------

def plot_classification_metrics(all_metrics: Dict[str, Dict], output_dir: Path):
    keys = ["patient_accuracy", "patient_precision", "patient_recall", "patient_f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names = list(all_metrics.keys())
    n = len(keys)
    n_m = len(model_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n)
    width = 0.8 / n_m

    for i, name in enumerate(model_names):
        vals = [all_metrics[name].get(k, 0) * 100 for k in keys]
        offset = (i - n_m / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=MODEL_LABELS.get(name, name),
                      color=MODEL_COLORS.get(name, None), edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_title("Patient-Level Classification Metrics", pad=14)
    # Place legend below the axes (tall Recall bars collide with top-anchored legends)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=min(len(model_names), 4), frameon=False)
    ax.set_ylim(0, 110)
    fig.subplots_adjust(bottom=0.18)
    _save(fig, output_dir / "classification_metrics.png", close=False)
    _save(fig, output_dir / "classification_metrics.pdf")


# -----------------------------------------------------------------------
# 9. Detection visualization on sample images
# -----------------------------------------------------------------------

def plot_detection_samples(
    images: List[torch.Tensor],
    targets: List[Dict],
    predictions_by_model: Dict[str, List[Dict]],
    output_dir: Path,
    num_samples: int = 4,
    per_model_threshold: Optional[Dict[str, float]] = None,
    default_threshold: float = 0.3,
):
    """Show ground truth vs. predictions side by side.

    Args:
        per_model_threshold: Optional per-model confidence threshold for display.
            Models output scores on different scales (e.g. FCOS optimal ≈ 0.49,
            RetinaNet ≈ 0.14, Faster R-CNN ≈ 0.84), so a single global threshold
            hides boxes from the more-conservatively-calibrated models. Passing
            per-model (typically Youden-optimal) thresholds produces a visually
            fair comparison.
    """
    n_models = len(predictions_by_model)
    model_names = list(predictions_by_model.keys())
    n_cols = 1 + n_models  # GT + each model
    n_rows = min(num_samples, len(images))

    if n_rows == 0:
        return

    if per_model_threshold is None:
        per_model_threshold = {}

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows))
    axes = np.atleast_2d(axes)

    for row in range(n_rows):
        img = images[row]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
        gt = targets[row]
        has_gt = len(gt["boxes"]) > 0

        # Ground truth column
        ax = axes[row, 0]
        ax.imshow(img, cmap="gray")
        for box in gt["boxes"]:
            rect = patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2.5, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)
        title = "Ground Truth" if row == 0 else ""
        if row == 0:
            ax.set_title(title, fontsize=11)
        # Tag the row with pos/neg
        ax.set_ylabel(f"case {row+1}\n{'(positive)' if has_gt else '(negative)'}",
                      fontsize=9, rotation=0, labelpad=34, va="center")
        ax.set_xticks([]); ax.set_yticks([])

        # Each model's predictions
        for col, name in enumerate(model_names, 1):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray")
            preds = predictions_by_model[name][row]
            thr = per_model_threshold.get(name, default_threshold)
            model_color = MODEL_COLORS.get(name, "red")
            for j in range(len(preds["boxes"])):
                score = float(preds["scores"][j])
                if score > thr:
                    box = preds["boxes"][j]
                    rect = patches.Rectangle(
                        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                        linewidth=2, edgecolor=model_color, facecolor="none"
                    )
                    ax.add_patch(rect)
                    ax.text(box[0], box[1] - 5, f"{score:.2f}",
                            color=model_color, fontsize=9, weight="bold")
            if row == 0:
                label = MODEL_LABELS.get(name, name)
                ax.set_title(f"{label}\n(thr={thr:.2f})", fontsize=11)
            ax.axis("off")

    fig.suptitle("Detection Results: Ground Truth vs. Model Predictions"
                 "  —  per-model visualisation thresholds "
                 "($\\max(0.5\\,\\tau^{\\star}, 0.10)$)",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    _save(fig, output_dir / "detection_samples.png", close=False)
    _save(fig, output_dir / "detection_samples.pdf")


# -----------------------------------------------------------------------
# 10. Epoch time comparison
# -----------------------------------------------------------------------

def plot_epoch_times(histories: Dict[str, Dict], output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    names = []
    avg_times = []
    colors = []
    for name, hist in histories.items():
        times = hist.get("epoch_times", [])
        if times:
            names.append(MODEL_LABELS.get(name, name))
            avg_times.append(np.mean(times))
            colors.append(MODEL_COLORS.get(name, "#888"))

    bars = ax.bar(names, avg_times, color=colors, edgecolor="white")
    for bar, t in zip(bars, avg_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{t:.1f}s", ha="center", va="bottom")

    ax.set_ylabel("Average Epoch Time (s)")
    ax.set_title("Training Speed Comparison")
    _save(fig, output_dir / "epoch_times.png", close=False)
    _save(fig, output_dir / "epoch_times.pdf")


# -----------------------------------------------------------------------
# 11. Summary table (LaTeX-ready)
# -----------------------------------------------------------------------

def generate_latex_table(all_metrics: Dict[str, Dict], output_dir: Path):
    """Generate a LaTeX-formatted comparison table (paper Table 3 style)."""
    keys = ["AP@0.5", "AP_M", "AP_L", "AR@10", "AR_M", "AR_L"]
    headers = ["Method", "AP", r"$AP_M$", r"$AP_L$", r"$AR_{10}$", r"$AR_M$", r"$AR_L$"]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Detection performance comparison on the RSNA dataset.}",
        r"\label{tab:comparison}",
        r"\begin{tabular}{l" + "c" * len(keys) + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]

    for name, metrics in all_metrics.items():
        row = [MODEL_LABELS.get(name, name)]
        for k in keys:
            row.append(f"{metrics.get(k, 0) * 100:.1f}")
        lines.append(" & ".join(row) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    table_str = "\n".join(lines)
    with open(output_dir / "comparison_table.tex", "w") as f:
        f.write(table_str)
    print(f"  Saved: {output_dir / 'comparison_table.tex'}")

    return table_str


# -----------------------------------------------------------------------
# Master function
# -----------------------------------------------------------------------

def generate_all_plots(
    histories: Dict[str, Dict],
    all_metrics: Dict[str, Dict],
    output_dir: str = "results",
    predictions_by_model: Optional[Dict[str, List[Dict]]] = None,
    targets: Optional[List[Dict]] = None,
    images: Optional[List[torch.Tensor]] = None,
):
    """Generate all plots and save to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    plot_training_losses(histories, out)
    plot_learning_rates(histories, out)
    plot_ap_comparison(all_metrics, out)
    plot_ar_comparison(all_metrics, out)
    plot_pr_curves(all_metrics, out)
    plot_val_ap_over_epochs(histories, out)
    plot_classification_metrics(all_metrics, out)
    plot_epoch_times(histories, out)
    generate_latex_table(all_metrics, out)

    if predictions_by_model is not None and targets is not None:
        plot_ap_vs_iou(predictions_by_model, targets, out)

    if images is not None and targets is not None and predictions_by_model is not None:
        plot_detection_samples(images, targets, predictions_by_model, out)

    # Save metrics as JSON (include precisions/recalls for PR curves)
    serializable = {}
    for name, m in all_metrics.items():
        serializable[name] = {k: v for k, v in m.items()}
    with open(out / "all_metrics.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved: {out / 'all_metrics.json'}")

    print("All plots generated.")
