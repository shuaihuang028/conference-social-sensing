"""Shared constants and lightweight records for the data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

GROUP_FORMS = ("A", "B", "C", "D")
RACE_CATEGORIES = ("W", "B", "SA", "EA", "MENA", "O")
AGE_CATEGORIES = ("1", "2", "3", "O")
VALID_GENDERS = {"M", "F", "O"}


@dataclass(frozen=True)
class ParsedSvg:
    photo: dict[str, Any]
    detections: list[dict[str, Any]]
    image_bytes: bytes
    image_extension: str
    svg_path: Path


def issue(
    severity: str,
    code: str,
    message: str,
    **context: Any,
) -> dict[str, Any]:
    """Create a JSON-serializable validation issue."""

    return {
        "severity": severity.upper(),
        "code": code,
        "message": message,
        **{key: value for key, value in context.items() if value is not None},
    }
