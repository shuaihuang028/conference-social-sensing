"""Train and evaluate the frozen-ResNet visual + geometry group classifier."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from conference_social_sensing.baselines.metrics import multilabel_metrics
from conference_social_sensing.baselines.models import decode_labels
from conference_social_sensing.modeling.group_dataset import (
    FeatureStandardizer,
    GroupCropDataset,
)
from conference_social_sensing.modeling.group_model import VisualGeometryGroupClassifier


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            normalization,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, evaluation_transform


def positive_weights(labels: np.ndarray, cap: float) -> torch.Tensor:
    positives = labels.sum(axis=0)
    negatives = len(labels) - positives
    weights = np.divide(
        negatives,
        positives,
        out=np.ones_like(negatives, dtype=np.float32),
        where=positives > 0,
    )
    return torch.tensor(np.clip(weights, 1.0, cap), dtype=torch.float32)


def _make_loaders(
    manifest: pd.DataFrame,
    *,
    project_root: Path,
    geometry_columns: Sequence[str],
    label_columns: Sequence[str],
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> tuple[dict[str, DataLoader], FeatureStandardizer]:
    train_transform, evaluation_transform = build_transforms(image_size)
    train_values = manifest.loc[
        manifest["split"] == "train", geometry_columns
    ].to_numpy(dtype=np.float32)
    standardizer = FeatureStandardizer().fit(train_values)
    loaders: dict[str, DataLoader] = {}
    for split_name in ("train", "val", "test"):
        frame = (
            manifest.loc[manifest["split"] == split_name].copy().reset_index(drop=True)
        )
        values = standardizer.transform(
            frame[list(geometry_columns)].to_numpy(dtype=np.float32)
        )
        dataset = GroupCropDataset(
            frame,
            project_root=project_root,
            geometry_columns=geometry_columns,
            label_columns=label_columns,
            geometry_values=values,
            transform=train_transform
            if split_name == "train"
            else evaluation_transform,
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=num_workers,
            pin_memory=False,
        )
    return loaders, standardizer


def train_one_epoch(
    model: VisualGeometryGroupClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    if model.freeze_backbone:
        model.backbone.eval()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        images = batch["image"].to(device)
        geometry = batch["geometry"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images, geometry)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        total_samples += len(labels)
    return total_loss / max(total_samples, 1)


@torch.inference_mode()
def evaluate(
    model: VisualGeometryGroupClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_columns: Sequence[str],
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    losses: list[float] = []
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    group_ids: list[str] = []
    photo_ids: list[str] = []
    for batch in loader:
        images = batch["image"].to(device)
        geometry = batch["geometry"].to(device)
        labels = batch["labels"].to(device)
        logits = model(images, geometry)
        losses.append(float(criterion(logits, labels).item()) * len(labels))
        all_labels.append(labels.cpu().numpy())
        all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
        group_ids.extend(batch["group_id"])
        photo_ids.extend(batch["photo_id"])

    truth = np.concatenate(all_labels).astype(int)
    probabilities = np.concatenate(all_probabilities)
    predictions = (probabilities >= threshold).astype(int)
    empty = np.where(predictions.sum(axis=1) == 0)[0]
    if len(empty):
        predictions[empty, probabilities[empty].argmax(axis=1)] = 1
    metrics = multilabel_metrics(truth, predictions, label_columns)
    metrics["loss"] = float(sum(losses) / max(len(truth), 1))
    output = pd.DataFrame({"group_id": group_ids, "photo_id": photo_ids})
    output["true_labels"] = [decode_labels(row, label_columns) for row in truth]
    output["predicted_labels"] = [
        decode_labels(row, label_columns) for row in predictions
    ]
    for index, label in enumerate(label_columns):
        output[f"prob_{label.removeprefix('form_')}"] = probabilities[:, index]
    return metrics, output


def run(config_path: Path) -> tuple[Path, Path]:
    started = time.perf_counter()
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parent.parent
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = choose_device(str(config.get("device", "auto")))
    label_columns = list(config["label_columns"])
    geometry_columns = list(config["geometry_columns"])
    manifest = pd.read_csv(
        project_root / config["manifest_file"], dtype={"photo_id": str}
    )
    if set(manifest["split"].unique()) != {"train", "val", "test"}:
        raise ValueError("Manifest must contain train, val, and test rows")

    loaders, standardizer = _make_loaders(
        manifest,
        project_root=project_root,
        geometry_columns=geometry_columns,
        label_columns=label_columns,
        batch_size=int(config["batch_size"]),
        num_workers=int(config.get("num_workers", 0)),
        image_size=int(config["image_size"]),
    )
    model = VisualGeometryGroupClassifier(
        num_geometry_features=len(geometry_columns),
        num_labels=len(label_columns),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        pretrained=bool(config.get("pretrained", True)),
        freeze_backbone=bool(config.get("freeze_backbone", True)),
    ).to(device)
    train_labels = manifest.loc[manifest["split"] == "train", label_columns].to_numpy(
        dtype=np.float32
    )
    pos_weight = positive_weights(
        train_labels, float(config.get("positive_weight_cap", 10.0))
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    patience = int(config["patience"])
    threshold = float(config["threshold"])
    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device
        )
        val_metrics, _ = evaluate(
            model, loaders["val"], criterion, device, label_columns, threshold
        )
        score = float(val_metrics["macro_f1"])
        val_loss = float(val_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": score,
                "val_micro_f1": float(val_metrics["micro_f1"]),
                "val_exact_match": float(val_metrics["exact_match_accuracy"]),
            }
        )
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_f1={score:.4f}"
        )
        improved = score > best_score + 1e-8 or (
            abs(score - best_score) <= 1e-8 and val_loss < best_loss
        )
        if improved:
            best_score, best_loss, best_epoch = score, val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(
                    f"Early stopping after epoch {epoch}; best epoch was {best_epoch}."
                )
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    val_metrics, val_predictions = evaluate(
        model, loaders["val"], criterion, device, label_columns, threshold
    )
    test_metrics, test_predictions = evaluate(
        model, loaders["test"], criterion, device, label_columns, threshold
    )

    output_dir = project_root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "VisualGeometryGroupClassifier",
            "label_columns": label_columns,
            "geometry_columns": geometry_columns,
            "geometry_mean": standardizer.mean_.tolist(),
            "geometry_scale": standardizer.scale_.tolist(),
            "config": config,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    val_predictions.insert(0, "split", "val")
    test_predictions.insert(0, "split", "test")
    predictions_path = output_dir / "predictions.csv"
    pd.concat([val_predictions, test_predictions], ignore_index=True).to_csv(
        predictions_path, index=False
    )

    is_provisional = bool(manifest.get("is_provisional", pd.Series([True])).all())
    report: dict[str, Any] = {
        "status": "provisional_full_data" if is_provisional else "final",
        "is_provisional": is_provisional,
        "model": "frozen_resnet18_plus_geometry",
        "device": str(device),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "trainable_parameters": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "positive_weights": dict(
            zip(label_columns, [float(value) for value in pos_weight.cpu()])
        ),
        "validation": val_metrics,
        "test": test_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "note": (
            "Full labeled two-source dataset. Split remains provisional because contiguous "
            "photo blocks are a proxy for true event/session scenes."
            if is_provisional
            else "Full labeled dataset evaluated on the frozen final split."
        ),
    }
    baseline_path = project_root / config.get("baseline_metrics_file", "")
    if baseline_path.is_file():
        baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
        report["baseline_test_metrics"] = {
            name: values["test"]
            for name, values in baseline_report.get("models", {}).items()
        }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return metrics_path, checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/group_model.json"))
    args = parser.parse_args()
    metrics_path, checkpoint_path = run(args.config)
    print(f"Wrote model metrics: {metrics_path}")
    print(f"Wrote best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
