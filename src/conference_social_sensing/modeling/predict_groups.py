"""Run a trained group-form checkpoint over rows in a group manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from conference_social_sensing.modeling.predictor import GroupFormPredictor


def run(
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    output_path: Path,
    project_root: Path,
    split: str | None,
    batch_size: int,
    device: str,
    threshold: float | None,
) -> pd.DataFrame:
    predictor = GroupFormPredictor(checkpoint_path, device=device)
    manifest = pd.read_csv(manifest_path, dtype={"photo_id": str})
    if split is not None:
        manifest = manifest.loc[manifest["split"] == split].copy()
    manifest = manifest.reset_index(drop=True)
    if manifest.empty:
        raise ValueError("No manifest rows selected for inference")

    output_rows: list[dict[str, object]] = []
    for start in range(0, len(manifest), batch_size):
        batch = manifest.iloc[start : start + batch_size]
        images = [project_root / path for path in batch["crop_path"].astype(str)]
        geometries = batch[predictor.geometry_columns].to_dict("records")
        predictions = predictor.predict_batch(images, geometries, threshold=threshold)
        for row, prediction in zip(batch.itertuples(index=False), predictions):
            output = {
                "group_id": row.group_id,
                "photo_id": row.photo_id,
                "split": row.split,
                "predicted_labels": ";".join(prediction["labels"]),
                "threshold": prediction["threshold"],
            }
            for label, probability in prediction["probabilities"].items():
                output[f"prob_{label}"] = probability
            output_rows.append(output)

    result = pd.DataFrame(output_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/group_model/best_model.pt")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/group_samples.csv")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/group_model/inference_predictions.csv"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    result = run(
        checkpoint_path=args.checkpoint.resolve(),
        manifest_path=args.manifest.resolve(),
        output_path=args.output.resolve(),
        project_root=args.project_root.resolve(),
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        threshold=args.threshold,
    )
    print(f"Predicted {len(result)} groups and wrote {args.output}")


if __name__ == "__main__":
    main()
