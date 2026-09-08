"""Normalize the human-coded Excel workbook into person/group tables."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import (
    AGE_CATEGORIES,
    GROUP_FORMS,
    RACE_CATEGORIES,
    VALID_GENDERS,
    issue,
)


@dataclass(frozen=True)
class AnnotationTables:
    groups: pd.DataFrame
    persons: pd.DataFrame
    memberships: pd.DataFrame
    issues: list[dict[str, Any]]


def _clean(value: Any) -> Any:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _as_int(value: Any, field: str, row_number: int) -> int:
    value = _clean(value)
    if value is None:
        raise ValueError(f"Excel row {row_number}: missing {field}")
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"Excel row {row_number}: {field} must be an integer")
    return int(number)


def parse_group_forms(raw_value: Any) -> tuple[str, ...]:
    """Parse single or multi-label group forms into canonical A/B/C/D order."""

    raw_value = _clean(raw_value)
    if raw_value is None or not str(raw_value).strip():
        raise ValueError("group_form is blank")

    text = str(raw_value).upper().strip().replace(" AND ", ";")
    invalid = re.sub(r"[ABCD\s,;/+&|]+", "", text)
    if invalid:
        raise ValueError(f"group_form contains unsupported text: {raw_value!r}")
    present = {label for label in GROUP_FORMS if label in text}
    if not present:
        raise ValueError(f"group_form contains no A/B/C/D label: {raw_value!r}")
    return tuple(label for label in GROUP_FORMS if label in present)


def validate_soft_distribution(
    values: dict[str, float | None],
    *,
    max_active_categories: int,
    tolerance: float = 1e-6,
) -> tuple[bool, str | None]:
    """Validate a sparse probability distribution from the coding workbook."""

    numeric = [0.0 if value is None else float(value) for value in values.values()]
    if any(value < 0 or value > 1 for value in numeric):
        return False, "probabilities must be between 0 and 1"
    if abs(sum(numeric) - 1.0) > tolerance:
        return False, f"probabilities sum to {sum(numeric):.6f}, expected 1"
    if sum(value > 0 for value in numeric) > max_active_categories:
        return False, f"more than {max_active_categories} active categories"
    return True, None


def _person_slots(columns: list[str]) -> list[int]:
    slots = {
        int(match.group(1))
        for column in columns
        if (match := re.fullmatch(r"per(\d+)_label", str(column)))
    }
    if not slots:
        raise ValueError("No perN_label columns found")
    return sorted(slots)


def read_annotations(excel_path: Path) -> AnnotationTables:
    """Read the first worksheet and normalize the human-coded annotations."""

    excel_path = Path(excel_path)
    frame = pd.read_excel(excel_path, sheet_name=0)
    slots = _person_slots([str(column) for column in frame.columns])

    group_rows: list[dict[str, Any]] = []
    person_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    seen_people: set[str] = set()

    for dataframe_index, row in frame.iterrows():
        excel_row = int(dataframe_index) + 2
        photo_value = _clean(row.get("photo_ID"))
        if photo_value is None:
            issues.append(
                issue(
                    "ERROR",
                    "MISSING_PHOTO_ID",
                    "Annotation row has no photo_ID.",
                    excel_row=excel_row,
                )
            )
            continue
        photo_id = str(photo_value).strip()

        try:
            group_number = _as_int(row.get("group_number"), "group_number", excel_row)
            declared_size = _as_int(row.get("group_size"), "group_size", excel_row)
            forms = parse_group_forms(row.get("group_form"))
        except (TypeError, ValueError) as exc:
            issues.append(
                issue(
                    "ERROR",
                    "INVALID_GROUP_ROW",
                    str(exc),
                    photo_id=photo_id,
                    excel_row=excel_row,
                )
            )
            continue

        group_id = f"{photo_id}__g{group_number}"
        members: list[int] = []
        for slot in slots:
            raw_label = _clean(row.get(f"per{slot}_label"))
            if raw_label is None:
                continue
            try:
                detection_id = _as_int(raw_label, f"per{slot}_label", excel_row)
            except (TypeError, ValueError) as exc:
                issues.append(
                    issue(
                        "ERROR",
                        "INVALID_PERSON_LABEL",
                        str(exc),
                        photo_id=photo_id,
                        group_id=group_id,
                        excel_row=excel_row,
                    )
                )
                continue

            members.append(detection_id)
            person_id = f"{photo_id}__p{detection_id}"
            membership_rows.append(
                {
                    "group_id": group_id,
                    "person_id": person_id,
                    "photo_id": photo_id,
                    "detection_id": detection_id,
                    "member_order": len(members),
                }
            )

            if person_id in seen_people:
                issues.append(
                    issue(
                        "ERROR",
                        "PERSON_IN_MULTIPLE_GROUPS",
                        "A person is assigned to more than one group in the same photo.",
                        photo_id=photo_id,
                        person_id=person_id,
                        group_id=group_id,
                        excel_row=excel_row,
                    )
                )
                continue
            seen_people.add(person_id)

            gender_value = _clean(row.get(f"per{slot}_gender"))
            gender = (
                str(gender_value).upper().strip() if gender_value is not None else None
            )
            if gender not in VALID_GENDERS:
                issues.append(
                    issue(
                        "ERROR",
                        "INVALID_GENDER",
                        f"Gender must be one of {sorted(VALID_GENDERS)}.",
                        photo_id=photo_id,
                        person_id=person_id,
                        value=gender,
                        excel_row=excel_row,
                    )
                )

            race = {
                category: _clean(row.get(f"per{slot}_race_{category}"))
                for category in RACE_CATEGORIES
            }
            age = {
                category: _clean(row.get(f"per{slot}_age_{category}"))
                for category in AGE_CATEGORIES
            }
            race_valid, race_message = validate_soft_distribution(
                race, max_active_categories=3
            )
            age_valid, age_message = validate_soft_distribution(
                age, max_active_categories=2
            )
            if not race_valid:
                issues.append(
                    issue(
                        "ERROR",
                        "INVALID_RACE_DISTRIBUTION",
                        race_message or "Invalid race distribution.",
                        photo_id=photo_id,
                        person_id=person_id,
                        excel_row=excel_row,
                    )
                )
            if not age_valid:
                issues.append(
                    issue(
                        "ERROR",
                        "INVALID_AGE_DISTRIBUTION",
                        age_message or "Invalid age distribution.",
                        photo_id=photo_id,
                        person_id=person_id,
                        excel_row=excel_row,
                    )
                )

            hispanic = _clean(row.get(f"per{slot}_hispanic"))
            if hispanic is not None and (float(hispanic) < 0 or float(hispanic) > 1):
                issues.append(
                    issue(
                        "ERROR",
                        "INVALID_HISPANIC_PROBABILITY",
                        "Hispanic probability must be between 0 and 1.",
                        photo_id=photo_id,
                        person_id=person_id,
                        excel_row=excel_row,
                    )
                )
            if hispanic is not None and float(race.get("W") or 0) != 1.0:
                issues.append(
                    issue(
                        "WARNING",
                        "HISPANIC_WITHOUT_CERTAIN_WHITE_LABEL",
                        "Coding memo expects Hispanic coding only when race_W equals 1.",
                        photo_id=photo_id,
                        person_id=person_id,
                        excel_row=excel_row,
                    )
                )

            person_row: dict[str, Any] = {
                "person_id": person_id,
                "photo_id": photo_id,
                "detection_id": detection_id,
                "gender": gender,
                "gender_mask": int(gender in VALID_GENDERS),
                "hispanic_probability": hispanic,
                "hispanic_mask": int(hispanic is not None),
                "race_mask": int(race_valid),
                "age_mask": int(age_valid),
                "race_distribution": json.dumps(
                    [float(race[category] or 0) for category in RACE_CATEGORIES]
                ),
                "age_distribution": json.dumps(
                    [float(age[category] or 0) for category in AGE_CATEGORIES]
                ),
                "source_excel_row": excel_row,
            }
            person_row.update(
                {f"race_{category}": race[category] for category in RACE_CATEGORIES}
            )
            person_row.update(
                {f"age_{category}": age[category] for category in AGE_CATEGORIES}
            )
            person_rows.append(person_row)

        if len(members) != declared_size:
            issues.append(
                issue(
                    "ERROR",
                    "GROUP_SIZE_MISMATCH",
                    "group_size does not match the number of populated person labels.",
                    photo_id=photo_id,
                    group_id=group_id,
                    declared_size=declared_size,
                    member_count=len(members),
                    excel_row=excel_row,
                )
            )

        group_row: dict[str, Any] = {
            "group_id": group_id,
            "photo_id": photo_id,
            "group_number": group_number,
            "group_size": declared_size,
            "member_count": len(members),
            "group_forms": json.dumps(list(forms)),
            "group_form_raw": str(_clean(row.get("group_form"))),
            "is_multilabel": int(len(forms) > 1),
            "within_group_note": _clean(row.get("within_group")),
            "other_note": _clean(row.get("other_note")),
            "source_excel_row": excel_row,
        }
        group_row.update(
            {f"form_{label}": int(label in forms) for label in GROUP_FORMS}
        )
        group_rows.append(group_row)

    return AnnotationTables(
        groups=pd.DataFrame(group_rows),
        persons=pd.DataFrame(person_rows),
        memberships=pd.DataFrame(membership_rows),
        issues=issues,
    )
