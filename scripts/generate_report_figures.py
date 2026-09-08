"""Generate compact, reproducible figures used by the project README."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets"


def _read_json(path: str) -> dict:
    return json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))


def plot_model_comparison() -> Path:
    baselines = _read_json("artifacts/baselines/metrics.json")
    model = _read_json("artifacts/group_model/metrics.json")
    names = ["Majority", "Size-only", "ResNet-18\n+ geometry"]
    reports = [
        baselines["models"]["majority_combination"]["test"],
        baselines["models"]["size_only"]["test"],
        model["test"],
    ]
    metric_names = ["macro_f1", "micro_f1", "exact_match_accuracy"]
    metric_labels = ["Macro-F1", "Micro-F1", "Exact match"]
    colors = ["#0072B2", "#E69F00", "#009E73"]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    positions = np.arange(len(names))
    width = 0.23
    for index, (metric, label, color) in enumerate(
        zip(metric_names, metric_labels, colors)
    ):
        values = [report[metric] for report in reports]
        bars = axes[0].bar(
            positions + (index - 1) * width, values, width, label=label, color=color
        )
        axes[0].bar_label(
            bars, labels=[f"{value:.2f}" for value in values], padding=2, fontsize=8
        )
    axes[0].set_title("Held-out test performance")
    axes[0].set_ylabel("Score")
    axes[0].set_xticks(positions, names)
    axes[0].set_ylim(0, 1.03)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")

    labels = ["A", "B", "C", "D"]
    baseline_f1 = [reports[1]["per_label"][f"form_{label}"]["f1"] for label in labels]
    model_f1 = [model["test"]["per_label"][f"form_{label}"]["f1"] for label in labels]
    class_positions = np.arange(len(labels))
    bars_a = axes[1].bar(
        class_positions - 0.18, baseline_f1, 0.36, label="Size-only", color="#999999"
    )
    bars_b = axes[1].bar(
        class_positions + 0.18,
        model_f1,
        0.36,
        label="ResNet-18 + geometry",
        color="#0072B2",
    )
    axes[1].bar_label(
        bars_a, labels=[f"{value:.2f}" for value in baseline_f1], padding=2, fontsize=8
    )
    axes[1].bar_label(
        bars_b, labels=[f"{value:.2f}" for value in model_f1], padding=2, fontsize=8
    )
    axes[1].set_title("Per-class F1")
    axes[1].set_ylabel("F1 score")
    axes[1].set_xlabel("Group form")
    axes[1].set_xticks(class_positions, labels)
    axes[1].set_ylim(0, 1.08)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper right")
    figure.suptitle("Full-data provisional evaluation", fontsize=14)

    output = OUTPUT_DIR / "model-comparison.png"
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_training_history() -> Path:
    history = pd.read_csv(PROJECT_ROOT / "artifacts/group_model/history.csv")
    model = _read_json("artifacts/group_model/metrics.json")
    best_epoch = int(model["best_epoch"])
    figure, left_axis = plt.subplots(figsize=(8, 4.3), constrained_layout=True)
    right_axis = left_axis.twinx()

    left_axis.plot(
        history["epoch"],
        history["train_loss"],
        label="Train loss",
        color="#0072B2",
        linewidth=2,
    )
    left_axis.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation loss",
        color="#D55E00",
        linewidth=2,
    )
    right_axis.plot(
        history["epoch"],
        history["val_macro_f1"],
        label="Validation Macro-F1",
        color="#009E73",
        linewidth=2,
    )
    left_axis.axvline(best_epoch, color="#555555", linestyle="--", linewidth=1)
    selected_loss = history.loc[history["epoch"] == best_epoch, "val_loss"].iloc[0]
    left_axis.annotate(
        f"Selected epoch {best_epoch}",
        xy=(best_epoch, selected_loss),
        xytext=(best_epoch - 7, 0.88),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=9,
    )
    left_axis.set_title("Frozen ResNet-18 + geometry training history")
    left_axis.set_xlabel("Epoch")
    left_axis.set_ylabel("Weighted BCE loss")
    right_axis.set_ylabel("Validation Macro-F1")
    left_axis.set_ylim(bottom=0)
    right_axis.set_ylim(0, 1)
    left_axis.grid(alpha=0.25)
    lines = left_axis.get_lines()[:2] + right_axis.get_lines()
    left_axis.legend(
        lines, [line.get_label() for line in lines], frameon=False, loc="center right"
    )

    output = OUTPUT_DIR / "training-history.png"
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (plot_model_comparison(), plot_training_history()):
        print(f"Wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
