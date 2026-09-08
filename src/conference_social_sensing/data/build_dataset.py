"""Command-line entry point for the Stage 0 dataset build."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .annotations import read_annotations
from .schemas import issue
from .svg_parser import extract_images, parse_svg_directory
from .validation import build_validation_report, validate_alignment, write_json


def _resolve(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    return path if path.is_absolute() else project_root / path


def _relative(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _configured_sources(config: dict[str, Any]) -> list[dict[str, str]]:
    """Return normalized source configs while retaining legacy single-source support."""
    if "sources" in config:
        sources = config["sources"]
        if not isinstance(sources, list) or not sources:
            raise ValueError("config.sources must be a non-empty list")
        normalized: list[dict[str, str]] = []
        for index, source in enumerate(sources):
            if not isinstance(source, dict):
                raise TypeError(f"config.sources[{index}] must be an object")
            missing = {"name", "svg_dir", "annotations_path"} - set(source)
            if missing:
                raise ValueError(
                    f"config.sources[{index}] is missing {sorted(missing)}"
                )
            normalized.append(
                {
                    "name": str(source["name"]),
                    "svg_dir": str(source["svg_dir"]),
                    "annotations_path": str(source["annotations_path"]),
                }
            )
        names = [source["name"] for source in normalized]
        if len(names) != len(set(names)):
            raise ValueError("config.sources names must be unique")
        return normalized
    return [
        {
            "name": str(config.get("source_name", "default")),
            "svg_dir": str(config["svg_dir"]),
            "annotations_path": str(config["annotations_path"]),
        }
    ]


def _mark_source(issues: list[dict[str, Any]], source_name: str) -> None:
    for item in issues:
        item.setdefault("source_batch", source_name)


def _duplicate_key_issues(
    frame: pd.DataFrame,
    *,
    key_columns: list[str],
    code: str,
    message: str,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    duplicates = frame.loc[frame.duplicated(key_columns, keep=False), key_columns]
    results: list[dict[str, Any]] = []
    for values in duplicates.drop_duplicates().to_dict("records"):
        results.append(issue("ERROR", code, message, **values))
    return results


def build_dataset(
    config_path: Path,
    *,
    output_dir_override: Path | None = None,
    extract_images_override: bool | None = None,
) -> dict[str, Any]:
    """Build normalized CSV tables and return the validation report."""

    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parent.parent.resolve()
    configured_sources = _configured_sources(config)
    output_dir = (
        Path(output_dir_override).resolve()
        if output_dir_override is not None
        else _resolve(project_root, config["output_dir"])
    )
    should_extract = (
        bool(extract_images_override)
        if extract_images_override is not None
        else bool(config.get("extract_images", True))
    )
    low_confidence_threshold = float(config.get("low_confidence_threshold", 0.5))

    photo_rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    group_frames: list[pd.DataFrame] = []
    person_frames: list[pd.DataFrame] = []
    membership_frames: list[pd.DataFrame] = []
    issues: list[dict[str, Any]] = []
    source_metadata: list[dict[str, str]] = []
    image_dir = output_dir / "images"
    for source in configured_sources:
        source_name = source["name"]
        svg_dir = _resolve(project_root, source["svg_dir"])
        annotations_path = _resolve(project_root, source["annotations_path"])
        parsed_svgs, source_issues = parse_svg_directory(svg_dir)
        _mark_source(source_issues, source_name)
        issues.extend(source_issues)
        if not parsed_svgs:
            raise RuntimeError(f"No valid SVG files found in {svg_dir}")
        if should_extract:
            extract_images(parsed_svgs, image_dir)

        for parsed in parsed_svgs:
            photo = dict(parsed.photo)
            photo["source_batch"] = source_name
            photo["svg_path"] = _relative(parsed.svg_path.resolve(), project_root)
            image_path = image_dir / f"{photo['photo_id']}{parsed.image_extension}"
            photo["image_path"] = _relative(image_path, project_root)
            photo_rows.append(photo)
            for detection in parsed.detections:
                detection_row = dict(detection)
                detection_row["source_batch"] = source_name
                detection_rows.append(detection_row)

        annotations = read_annotations(annotations_path)
        _mark_source(annotations.issues, source_name)
        issues.extend(annotations.issues)
        for frame, collection in (
            (annotations.groups, group_frames),
            (annotations.persons, person_frames),
            (annotations.memberships, membership_frames),
        ):
            enriched = frame.copy()
            enriched["source_batch"] = source_name
            enriched["source_workbook"] = _relative(
                annotations_path.resolve(), project_root
            )
            collection.append(enriched)
        source_metadata.append(
            {
                "name": source_name,
                "svg_dir": _relative(svg_dir.resolve(), project_root),
                "annotations_path": _relative(annotations_path.resolve(), project_root),
            }
        )

    photos = (
        pd.DataFrame(photo_rows)
        .sort_values(["source_batch", "photo_id"])
        .reset_index(drop=True)
    )
    detections = (
        pd.DataFrame(detection_rows)
        .sort_values(["source_batch", "photo_id", "detection_id"])
        .reset_index(drop=True)
    )
    groups = (
        pd.concat(group_frames, ignore_index=True)
        .sort_values(["source_batch", "photo_id", "group_number"])
        .reset_index(drop=True)
    )
    persons = (
        pd.concat(person_frames, ignore_index=True)
        .sort_values(["source_batch", "photo_id", "detection_id"])
        .reset_index(drop=True)
    )
    memberships = (
        pd.concat(membership_frames, ignore_index=True)
        .sort_values(["source_batch", "photo_id", "group_id", "member_order"])
        .reset_index(drop=True)
    )

    issues.extend(
        _duplicate_key_issues(
            photos,
            key_columns=["photo_id"],
            code="DUPLICATE_PHOTO_ID_ACROSS_SOURCES",
            message="The same photo_id occurs in more than one configured source.",
        )
    )
    issues.extend(
        _duplicate_key_issues(
            detections,
            key_columns=["photo_id", "detection_id"],
            code="DUPLICATE_DETECTION_KEY_ACROSS_SOURCES",
            message="The same photo/detection key occurs in more than one configured source.",
        )
    )
    issues.extend(
        _duplicate_key_issues(
            groups,
            key_columns=["group_id"],
            code="DUPLICATE_GROUP_ID_ACROSS_SOURCES",
            message="The same group_id occurs in more than one configured source.",
        )
    )

    detections, alignment_issues = validate_alignment(
        photos,
        detections,
        groups,
        persons,
        low_confidence_threshold=low_confidence_threshold,
    )
    issues.extend(alignment_issues)

    group_counts = groups.groupby("photo_id").size().to_dict()
    person_counts = persons.groupby("photo_id").size().to_dict()
    photos["has_group_annotations"] = (
        photos["photo_id"].map(group_counts).notna().astype(int)
    )
    photos["n_annotated_groups"] = (
        photos["photo_id"].map(group_counts).fillna(0).astype(int)
    )
    photos["n_annotated_persons"] = (
        photos["photo_id"].map(person_counts).fillna(0).astype(int)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "photos": photos,
        "detections": detections,
        "persons": persons,
        "groups": groups,
        "memberships": memberships,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    report = build_validation_report(
        photos=photos,
        detections=detections,
        groups=groups,
        persons=persons,
        memberships=memberships,
        issues=issues,
        low_confidence_threshold=low_confidence_threshold,
        source_paths={
            "data_sources": source_metadata,
            "config_path": _relative(config_path, project_root),
        },
    )
    write_json(output_dir / "validation_report.json", report)
    write_json(
        output_dir / "dataset_metadata.json",
        {
            "schema_version": "1.1",
            "data_sources": source_metadata,
            "table_files": {name: f"{name}.csv" for name in tables},
            "group_form_order": ["A", "B", "C", "D"],
            "race_order": ["W", "B", "SA", "EA", "MENA", "O"],
            "age_order": ["1", "2", "3", "O"],
            "unknown_policy": (
                "Detections absent from the human-coded workbook remain unknown "
                "and must not be treated as negative interaction labels."
            ),
        },
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract SVG images and build normalized conference annotation tables."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to JSON config."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory override, primarily for testing.",
    )
    parser.add_argument(
        "--no-extract-images",
        action="store_true",
        help="Build manifests without writing embedded images.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_dataset(
        args.config,
        output_dir_override=args.output_dir,
        extract_images_override=False if args.no_extract_images else None,
    )
    counts = report["counts"]
    validation = report["validation"]
    print(
        "Built dataset: "
        f"{counts['photos']} photos, {counts['detections']} detections, "
        f"{counts['groups']} groups, {counts['annotated_persons']} annotated people."
    )
    print(
        f"Validation: {validation['errors']} errors, {validation['warnings']} warnings."
    )
    if report["status"] == "failed":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
