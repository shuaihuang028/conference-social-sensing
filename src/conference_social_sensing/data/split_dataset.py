"""Create a provisional, leakage-aware photo split.

Nearby photo IDs are treated as a scene block until true scene/session metadata is
available. A whole block is always assigned to one split.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

SPLIT_ORDER = ("train", "val", "test")


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    )


def allocate_block_counts(
    num_blocks: int, ratios: Mapping[str, float]
) -> dict[str, int]:
    """Allocate all blocks by largest remainder, deterministically."""
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if set(ratios) != set(SPLIT_ORDER):
        raise ValueError(f"ratios must contain exactly {SPLIT_ORDER}")
    total = float(sum(ratios.values()))
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")

    quotas = {name: num_blocks * float(ratios[name]) / total for name in SPLIT_ORDER}
    counts = {name: int(quotas[name]) for name in SPLIT_ORDER}
    remaining = num_blocks - sum(counts.values())
    ranked = sorted(
        SPLIT_ORDER,
        key=lambda name: (-(quotas[name] - counts[name]), SPLIT_ORDER.index(name)),
    )
    for name in ranked[:remaining]:
        counts[name] += 1
    return counts


def create_contiguous_block_split(
    photo_ids: Sequence[str],
    *,
    block_size: int = 10,
    ratios: Mapping[str, float] | None = None,
    provisional: bool = True,
    scene_prefix: str = "auto",
) -> pd.DataFrame:
    """Assign naturally ordered photo IDs to unsplittable contiguous blocks."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    ordered_ids = sorted({str(photo_id) for photo_id in photo_ids}, key=_natural_key)
    if not ordered_ids:
        raise ValueError("photo_ids cannot be empty")

    ratios = ratios or {"train": 0.70, "val": 0.15, "test": 0.15}
    num_blocks = (len(ordered_ids) + block_size - 1) // block_size
    block_counts = allocate_block_counts(num_blocks, ratios)

    block_to_split: dict[int, str] = {}
    cursor = 0
    for split_name in SPLIT_ORDER:
        for block_index in range(cursor, cursor + block_counts[split_name]):
            block_to_split[block_index] = split_name
        cursor += block_counts[split_name]

    rows = []
    for sequence_index, photo_id in enumerate(ordered_ids):
        block_index = sequence_index // block_size
        rows.append(
            {
                "photo_id": photo_id,
                "split": block_to_split[block_index],
                "sequence_index": sequence_index,
                "scene_id": f"{scene_prefix}_block_{block_index:03d}",
                "split_strategy": "contiguous_blocks",
                "is_provisional": bool(provisional),
            }
        )
    return pd.DataFrame(rows)


def create_source_aware_block_split(
    photos: pd.DataFrame,
    *,
    block_size: int = 10,
    ratios: Mapping[str, float] | None = None,
    provisional: bool = True,
) -> pd.DataFrame:
    """Create independent contiguous splits within each conference/source batch."""
    if "source_batch" not in photos.columns:
        return create_contiguous_block_split(
            photos["photo_id"].astype(str).tolist(),
            block_size=block_size,
            ratios=ratios,
            provisional=provisional,
        )

    frames: list[pd.DataFrame] = []
    for source_name, source_photos in photos.groupby("source_batch", sort=True):
        source_split = create_contiguous_block_split(
            source_photos["photo_id"].astype(str).tolist(),
            block_size=block_size,
            ratios=ratios,
            provisional=provisional,
            scene_prefix=str(source_name),
        )
        source_split.insert(1, "source_batch", str(source_name))
        source_split["split_strategy"] = "source_contiguous_blocks"
        frames.append(source_split)
    return pd.concat(frames, ignore_index=True)


def build_split_report(
    split_df: pd.DataFrame, groups_df: pd.DataFrame
) -> dict[str, object]:
    merged = groups_df.merge(split_df[["photo_id", "split"]], on="photo_id", how="left")
    if merged["split"].isna().any():
        missing = sorted(merged.loc[merged["split"].isna(), "photo_id"].unique())
        raise ValueError(f"Groups reference photos absent from split: {missing}")

    photo_counts = split_df["split"].value_counts().reindex(SPLIT_ORDER, fill_value=0)
    group_counts = merged["split"].value_counts().reindex(SPLIT_ORDER, fill_value=0)
    label_counts: dict[str, dict[str, int]] = {}
    for split_name in SPLIT_ORDER:
        part = merged.loc[merged["split"] == split_name]
        label_counts[split_name] = {
            label: int(part[label].sum())
            for label in ("form_A", "form_B", "form_C", "form_D")
        }

    leakage = split_df.groupby("scene_id")["split"].nunique()
    report: dict[str, object] = {
        "status": "provisional_smoke_test"
        if bool(split_df["is_provisional"].all())
        else "final",
        "strategy": str(split_df["split_strategy"].iloc[0]),
        "is_provisional": bool(split_df["is_provisional"].all()),
        "num_photos": len(split_df),
        "num_scene_blocks": int(split_df["scene_id"].nunique()),
        "photo_counts": {name: int(photo_counts[name]) for name in SPLIT_ORDER},
        "group_counts": {name: int(group_counts[name]) for name in SPLIT_ORDER},
        "label_positive_counts": label_counts,
        "scene_blocks_spanning_multiple_splits": int((leakage > 1).sum()),
        "note": (
            "Photo-order blocks are a conservative proxy for scenes and are allocated "
            "independently within each source batch. Replace them with true event/session "
            "or burst metadata before reporting final performance."
        ),
    }
    if "source_batch" in split_df.columns:
        source_summary: dict[str, object] = {}
        group_source = (
            groups_df[["photo_id", "source_batch"]]
            if "source_batch" in groups_df
            else None
        )
        for source_name, source_photos in split_df.groupby("source_batch", sort=True):
            if group_source is not None:
                source_groups = merged.loc[merged["source_batch"] == source_name]
            else:
                photo_ids = set(source_photos["photo_id"].astype(str))
                source_groups = merged.loc[
                    merged["photo_id"].astype(str).isin(photo_ids)
                ]
            source_summary[str(source_name)] = {
                "photo_counts": {
                    name: int((source_photos["split"] == name).sum())
                    for name in SPLIT_ORDER
                },
                "group_counts": {
                    name: int((source_groups["split"] == name).sum())
                    for name in SPLIT_ORDER
                },
            }
        report["source_summary"] = source_summary
    return report


def run(config_path: Path) -> tuple[Path, Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.resolve().parent.parent
    processed_dir = project_root / config["processed_dir"]
    photos_df = pd.read_csv(processed_dir / "photos.csv", dtype={"photo_id": str})
    groups_df = pd.read_csv(processed_dir / "groups.csv", dtype={"photo_id": str})

    split_df = create_source_aware_block_split(
        photos_df,
        block_size=int(config.get("block_size", 10)),
        ratios=config.get("ratios"),
        provisional=bool(config.get("provisional", True)),
    )
    report = build_split_report(split_df, groups_df)

    output_path = project_root / config["output_file"]
    report_path = project_root / config["report_file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/split.json"))
    args = parser.parse_args()
    split_path, report_path = run(args.config)
    print(f"Wrote split assignments: {split_path}")
    print(f"Wrote split report: {report_path}")


if __name__ == "__main__":
    main()
