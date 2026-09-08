"""Cross-source validation and compact dataset reporting."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import GROUP_FORMS, issue


def validate_alignment(
    photos: pd.DataFrame,
    detections: pd.DataFrame,
    groups: pd.DataFrame,
    persons: pd.DataFrame,
    *,
    low_confidence_threshold: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Validate annotation-to-SVG alignment and mark annotation membership."""

    issues: list[dict[str, Any]] = []
    photo_ids = set(photos["photo_id"].astype(str))
    detection_keys = set(
        zip(
            detections["photo_id"].astype(str),
            detections["detection_id"].astype(int),
            strict=True,
        )
    )

    for photo_id in sorted(set(groups["photo_id"].astype(str)) - photo_ids):
        issues.append(
            issue(
                "ERROR",
                "ANNOTATION_PHOTO_WITHOUT_SVG",
                "The workbook references a photo with no SVG file.",
                photo_id=photo_id,
            )
        )

    for record in persons[["person_id", "photo_id", "detection_id"]].to_dict("records"):
        key = (str(record["photo_id"]), int(record["detection_id"]))
        if key not in detection_keys:
            issues.append(
                issue(
                    "ERROR",
                    "ANNOTATED_PERSON_WITHOUT_DETECTION",
                    "The workbook person label does not exist in the corresponding SVG.",
                    photo_id=record["photo_id"],
                    person_id=record["person_id"],
                    detection_id=int(record["detection_id"]),
                )
            )

    annotated_person_ids = set(persons["person_id"].astype(str))
    enriched = detections.copy()
    enriched["membership_status"] = enriched["person_id"].map(
        lambda person_id: (
            "annotated_group_member"
            if str(person_id) in annotated_person_ids
            else "unknown"
        )
    )
    enriched["is_annotated_group_member"] = (
        enriched["membership_status"] == "annotated_group_member"
    ).astype(int)

    annotated_low_confidence = enriched[
        (enriched["is_annotated_group_member"] == 1)
        & (enriched["detector_confidence"] < low_confidence_threshold)
    ]
    for record in annotated_low_confidence[
        ["photo_id", "person_id", "detection_id", "detector_confidence"]
    ].to_dict("records"):
        issues.append(
            issue(
                "WARNING",
                "ANNOTATED_LOW_CONFIDENCE_DETECTION",
                "Human annotation references a detector box below the configured threshold.",
                **record,
            )
        )

    return enriched, issues


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    counts = frame[column].fillna("<missing>").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def build_validation_report(
    *,
    photos: pd.DataFrame,
    detections: pd.DataFrame,
    groups: pd.DataFrame,
    persons: pd.DataFrame,
    memberships: pd.DataFrame,
    issues: list[dict[str, Any]],
    low_confidence_threshold: float,
    source_paths: dict[str, Any],
) -> dict[str, Any]:
    """Build the human-readable and machine-readable Stage 0 report."""

    severity_counts = Counter(str(item["severity"]) for item in issues)
    issue_code_counts = Counter(str(item["code"]) for item in issues)
    form_counts = {
        label: int(groups.get(f"form_{label}", pd.Series(dtype=int)).sum())
        for label in GROUP_FORMS
    }
    combination_counts = _value_counts(groups, "group_forms")
    annotated = detections[detections["is_annotated_group_member"] == 1]
    unannotated_photo_ids = sorted(
        set(photos["photo_id"].astype(str)) - set(groups["photo_id"].astype(str))
    )

    confidence = detections["detector_confidence"].astype(float)
    coverage = len(persons) / len(detections) if len(detections) else 0.0
    report = {
        "status": "failed" if severity_counts["ERROR"] else "passed",
        "sources": source_paths,
        "counts": {
            "photos": len(photos),
            "detections": len(detections),
            "annotated_photos": int(groups["photo_id"].nunique()),
            "unannotated_photos": len(unannotated_photo_ids),
            "groups": len(groups),
            "annotated_persons": len(persons),
            "memberships": len(memberships),
            "unknown_detections": int(
                (detections["membership_status"] == "unknown").sum()
            ),
            "multilabel_groups": int(groups["is_multilabel"].sum()),
        },
        "annotation_coverage": round(coverage, 6),
        "unannotated_photo_ids": unannotated_photo_ids,
        "distributions": {
            "groups_per_photo": {
                str(key): int(value)
                for key, value in groups.groupby("photo_id")
                .size()
                .value_counts()
                .items()
            },
            "group_size": _value_counts(groups, "group_size"),
            "group_form_atomic": form_counts,
            "group_form_combinations": combination_counts,
            "gender": _value_counts(persons, "gender"),
            "image_orientation": _value_counts(photos, "orientation"),
        },
        "detector_quality": {
            "confidence_min": float(confidence.min()),
            "confidence_max": float(confidence.max()),
            "below_threshold": int((confidence < low_confidence_threshold).sum()),
            "threshold": float(low_confidence_threshold),
            "annotated_below_threshold": int(
                (annotated["detector_confidence"] < low_confidence_threshold).sum()
            ),
        },
        "validation": {
            "errors": int(severity_counts["ERROR"]),
            "warnings": int(severity_counts["WARNING"]),
            "issue_codes": dict(sorted(issue_code_counts.items())),
            "issues": issues,
        },
    }
    return report


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
