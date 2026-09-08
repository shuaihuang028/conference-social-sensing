"""Build one context-aware image crop and geometry row per annotated group."""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

LABEL_COLUMNS = ("form_A", "form_B", "form_C", "form_D")


def expanded_union_bounds(
    boxes: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    """Return a clipped integer crop around the union of member boxes."""
    values = np.asarray(boxes, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or len(values) == 0:
        raise ValueError("boxes must be a non-empty Nx4 array")
    if image_width <= 0 or image_height <= 0 or margin_ratio < 0:
        raise ValueError("invalid image dimensions or margin ratio")

    x1, y1 = values[:, :2].min(axis=0)
    x2, y2 = values[:, 2:].max(axis=0)
    width, height = x2 - x1, y2 - y1
    left = max(0, math.floor(x1 - width * margin_ratio))
    top = max(0, math.floor(y1 - height * margin_ratio))
    right = min(image_width, math.ceil(x2 + width * margin_ratio))
    bottom = min(image_height, math.ceil(y2 + height * margin_ratio))
    if right <= left or bottom <= top:
        raise ValueError("computed crop has zero area")
    return left, top, right, bottom


def geometry_features(
    boxes: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    """Compute fixed-length, image-normalized group geometry features."""
    values = np.asarray(boxes, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or len(values) == 0:
        raise ValueError("boxes must be a non-empty Nx4 array")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    x1, y1 = values[:, :2].min(axis=0)
    x2, y2 = values[:, 2:].max(axis=0)
    widths = values[:, 2] - values[:, 0]
    heights = values[:, 3] - values[:, 1]
    centers = np.column_stack(
        ((values[:, 0] + values[:, 2]) / 2, (values[:, 1] + values[:, 3]) / 2)
    )
    diagonal = math.hypot(image_width, image_height)
    pair_distances = [
        float(np.linalg.norm(a - b) / diagonal) for a, b in combinations(centers, 2)
    ]

    return {
        "geom_center_x": float(((x1 + x2) / 2) / image_width),
        "geom_center_y": float(((y1 + y2) / 2) / image_height),
        "geom_union_width": float((x2 - x1) / image_width),
        "geom_union_height": float((y2 - y1) / image_height),
        "geom_union_area": float(
            ((x2 - x1) * (y2 - y1)) / (image_width * image_height)
        ),
        "geom_mean_member_width": float(widths.mean() / image_width),
        "geom_mean_member_height": float(heights.mean() / image_height),
        "geom_center_std_x": float(centers[:, 0].std() / image_width),
        "geom_center_std_y": float(centers[:, 1].std() / image_height),
        "geom_pair_distance_mean": float(np.mean(pair_distances))
        if pair_distances
        else 0.0,
        "geom_pair_distance_max": float(np.max(pair_distances))
        if pair_distances
        else 0.0,
    }


def build_group_crops(config_path: Path) -> pd.DataFrame:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parent.parent
    processed_dir = project_root / config["processed_dir"]
    output_dir = project_root / config["output_dir"]
    manifest_path = project_root / config["manifest_file"]

    groups = pd.read_csv(processed_dir / "groups.csv", dtype={"photo_id": str})
    memberships = pd.read_csv(
        processed_dir / "memberships.csv", dtype={"photo_id": str}
    )
    detections = pd.read_csv(processed_dir / "detections.csv", dtype={"photo_id": str})
    photos = pd.read_csv(processed_dir / "photos.csv", dtype={"photo_id": str})
    splits = pd.read_csv(processed_dir / "splits.csv", dtype={"photo_id": str})

    member_boxes = memberships.merge(
        detections[["photo_id", "person_id", "detection_id", "x1", "y1", "x2", "y2"]],
        on=["photo_id", "person_id", "detection_id"],
        how="left",
        validate="one_to_one",
    )
    if member_boxes[["x1", "y1", "x2", "y2"]].isna().any().any():
        raise ValueError(
            "At least one group membership could not be matched to a detection box"
        )

    photo_lookup = photos.set_index("photo_id")
    split_lookup = splits.set_index("photo_id")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_size = int(config.get("output_size", 256))
    margin_ratio = float(config.get("margin_ratio", 0.20))
    jpeg_quality = int(config.get("jpeg_quality", 90))
    rows: list[dict[str, object]] = []

    for group in groups.itertuples(index=False):
        members = member_boxes.loc[
            member_boxes["group_id"] == group.group_id
        ].sort_values("member_order")
        if len(members) != int(group.member_count):
            raise ValueError(f"Member count mismatch while cropping {group.group_id}")
        photo = photo_lookup.loc[group.photo_id]
        boxes = members[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        bounds = expanded_union_bounds(
            boxes,
            image_width=int(photo["width"]),
            image_height=int(photo["height"]),
            margin_ratio=margin_ratio,
        )

        source_path = project_root / str(photo["image_path"])
        crop_path = output_dir / f"{group.group_id}.jpg"
        with Image.open(source_path) as source:
            crop = source.convert("RGB").crop(bounds)
            crop = ImageOps.pad(
                crop,
                (output_size, output_size),
                method=Image.Resampling.LANCZOS,
                color=(0, 0, 0),
                centering=(0.5, 0.5),
            )
            crop.save(crop_path, format="JPEG", quality=jpeg_quality, optimize=True)

        left, top, right, bottom = bounds
        row: dict[str, object] = {
            "group_id": group.group_id,
            "photo_id": group.photo_id,
            "source_batch": group.source_batch,
            "split": str(split_lookup.loc[group.photo_id, "split"]),
            "is_provisional": bool(split_lookup.loc[group.photo_id, "is_provisional"]),
            "crop_path": str(crop_path.relative_to(project_root)),
            "group_size": int(group.group_size),
            "member_count": int(group.member_count),
            "crop_x1": left,
            "crop_y1": top,
            "crop_x2": right,
            "crop_y2": bottom,
            "crop_width_norm": (right - left) / int(photo["width"]),
            "crop_height_norm": (bottom - top) / int(photo["height"]),
            "is_multilabel": int(group.is_multilabel),
        }
        row.update(
            geometry_features(
                boxes,
                image_width=int(photo["width"]),
                image_height=int(photo["height"]),
            )
        )
        for label in LABEL_COLUMNS:
            row[label] = int(getattr(group, label))
        rows.append(row)

    manifest = (
        pd.DataFrame(rows).sort_values(["photo_id", "group_id"]).reset_index(drop=True)
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/group_crops.json"))
    args = parser.parse_args()
    manifest = build_group_crops(args.config)
    print(
        f"Built {len(manifest)} group crops and wrote {args.config.parent.parent / json.loads(args.config.read_text())['manifest_file']}"
    )


if __name__ == "__main__":
    main()
