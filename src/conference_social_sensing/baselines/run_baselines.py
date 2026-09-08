"""Fit and evaluate simple group-form baselines on the normalized dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from conference_social_sensing.baselines.metrics import multilabel_metrics
from conference_social_sensing.baselines.models import (
    MajorityCombinationBaseline,
    SizeOnlyBaseline,
    decode_labels,
)


def _serializable_vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def run(config_path: Path) -> tuple[Path, Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.resolve().parent.parent
    label_columns = list(config["label_columns"])
    groups = pd.read_csv(project_root / config["groups_file"], dtype={"photo_id": str})
    splits = pd.read_csv(project_root / config["splits_file"], dtype={"photo_id": str})
    data = groups.merge(
        splits[["photo_id", "split", "is_provisional"]], on="photo_id", how="left"
    )
    if data["split"].isna().any():
        missing = sorted(data.loc[data["split"].isna(), "photo_id"].unique())
        raise ValueError(f"Groups reference photos absent from split: {missing}")

    train = data.loc[data["split"] == "train"]
    if train.empty:
        raise ValueError("Training split is empty")
    y_train = train[label_columns].to_numpy(dtype=int)

    models = {
        "majority_combination": MajorityCombinationBaseline().fit(y_train),
        "size_only": SizeOnlyBaseline(
            threshold=float(config.get("threshold", 0.5))
        ).fit(train["group_size"].to_numpy(dtype=int), y_train),
    }

    metrics: dict[str, object] = {
        "status": "provisional_smoke_test"
        if bool(data["is_provisional"].all())
        else "final",
        "is_provisional": bool(data["is_provisional"].all()),
        "label_columns": label_columns,
        "models": {},
        "note": "Fit on train only. Validation/test scores are smoke-test results until the full dataset and final scene split are frozen.",
    }
    prediction_frames = []
    for model_name, model in models.items():
        model_metrics: dict[str, object] = {}
        for split_name in ("val", "test"):
            part = data.loc[data["split"] == split_name].copy()
            if model_name == "majority_combination":
                predictions, probabilities = model.predict(len(part))
            else:
                predictions, probabilities = model.predict(
                    part["group_size"].to_numpy(dtype=int)
                )

            truth = part[label_columns].to_numpy(dtype=int)
            model_metrics[split_name] = multilabel_metrics(
                truth, predictions, label_columns
            )
            output = part[["group_id", "photo_id", "group_size"]].copy()
            output.insert(0, "model", model_name)
            output.insert(1, "split", split_name)
            output["true_labels"] = [decode_labels(row, label_columns) for row in truth]
            output["predicted_labels"] = [
                decode_labels(row, label_columns) for row in predictions
            ]
            for index, label in enumerate(label_columns):
                output[f"prob_{label.removeprefix('form_')}"] = probabilities[:, index]
            prediction_frames.append(output)
        metrics["models"][model_name] = model_metrics

    majority = models["majority_combination"]
    size_only = models["size_only"]
    metrics["fitted_parameters"] = {
        "majority_combination": {
            "predicted_labels": decode_labels(majority.combination_, label_columns),
            "train_prevalence": dict(
                zip(label_columns, _serializable_vector(majority.prevalence_))
            ),
        },
        "size_only": {
            "threshold": size_only.threshold,
            "global_prevalence": dict(
                zip(label_columns, _serializable_vector(size_only.global_prevalence_))
            ),
            "prevalence_by_group_size": {
                str(size): dict(zip(label_columns, _serializable_vector(values)))
                for size, values in sorted(size_only.prevalence_by_size_.items())
            },
        },
    }

    output_dir = project_root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "predictions.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        predictions_path, index=False
    )
    return metrics_path, predictions_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baselines.json"))
    args = parser.parse_args()
    metrics_path, predictions_path = run(args.config)
    print(f"Wrote baseline metrics: {metrics_path}")
    print(f"Wrote baseline predictions: {predictions_path}")


if __name__ == "__main__":
    main()
