"""Parse detector SVG files and optionally extract their embedded images."""

from __future__ import annotations

import base64
import hashlib
import io
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

from PIL import Image

from .schemas import ParsedSvg, issue

_MIME_TO_EXTENSION = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _number(value: str | None, field: str, svg_path: Path) -> float:
    if value is None:
        raise ValueError(f"{svg_path.name}: missing {field}")
    match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*(?:px)?\s*", value)
    if not match:
        raise ValueError(f"{svg_path.name}: invalid {field}={value!r}")
    return float(match.group(1))


def _embedded_image(root: ET.Element, svg_path: Path) -> tuple[bytes, str, int, int]:
    image_element = next(
        (element for element in root.iter() if _local_name(element.tag) == "image"),
        None,
    )
    if image_element is None:
        raise ValueError(f"{svg_path.name}: no embedded image element")

    href = next(
        (value for key, value in image_element.attrib.items() if key.endswith("href")),
        None,
    )
    if not href or not href.startswith("data:") or ";base64," not in href:
        raise ValueError(f"{svg_path.name}: image is not an embedded base64 data URL")

    header, encoded = href.split(",", 1)
    mime_type = header[5:].split(";", 1)[0].lower()
    extension = _MIME_TO_EXTENSION.get(mime_type)
    if extension is None:
        raise ValueError(
            f"{svg_path.name}: unsupported embedded image MIME type {mime_type}"
        )

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError(f"{svg_path.name}: invalid base64 image") from exc

    with Image.open(io.BytesIO(image_bytes)) as image:
        image.verify()
    with Image.open(io.BytesIO(image_bytes)) as image:
        embedded_width, embedded_height = image.size

    return image_bytes, extension, embedded_width, embedded_height


def parse_svg(svg_path: Path) -> tuple[ParsedSvg, list[dict[str, object]]]:
    """Parse one SVG into a photo record, detections, and validation issues."""

    svg_path = Path(svg_path)
    svg_bytes = svg_path.read_bytes()
    root = ET.fromstring(svg_bytes)
    photo_id = svg_path.stem

    width = round(_number(root.attrib.get("width"), "width", svg_path))
    height = round(_number(root.attrib.get("height"), "height", svg_path))
    image_bytes, image_extension, embedded_width, embedded_height = _embedded_image(
        root, svg_path
    )

    issues: list[dict[str, object]] = []
    if (width, height) != (embedded_width, embedded_height):
        issues.append(
            issue(
                "ERROR",
                "SVG_IMAGE_DIMENSION_MISMATCH",
                "SVG canvas dimensions differ from the embedded image.",
                photo_id=photo_id,
                svg_width=width,
                svg_height=height,
                image_width=embedded_width,
                image_height=embedded_height,
            )
        )

    rectangles: list[dict[str, float]] = []
    for element in root.iter():
        if _local_name(element.tag) != "rect":
            continue
        title = next(
            (
                child.text or ""
                for child in element
                if _local_name(child.tag) == "title"
            ),
            "",
        )
        confidence_match = re.fullmatch(r"\s*conf:([0-9.]+)\s*", title)
        if confidence_match is None:
            continue
        rectangles.append(
            {
                "x": _number(element.attrib.get("x"), "rect.x", svg_path),
                "y": _number(element.attrib.get("y"), "rect.y", svg_path),
                "width": _number(element.attrib.get("width"), "rect.width", svg_path),
                "height": _number(
                    element.attrib.get("height"), "rect.height", svg_path
                ),
                "detector_confidence": float(confidence_match.group(1)),
            }
        )

    labels: list[int] = []
    for element in root.iter():
        if _local_name(element.tag) != "text" or element.text is None:
            continue
        label_text = element.text.strip()
        if re.fullmatch(r"\d+", label_text):
            labels.append(int(label_text))

    if len(rectangles) != len(labels):
        raise ValueError(
            f"{svg_path.name}: {len(rectangles)} detector rectangles but "
            f"{len(labels)} numeric labels"
        )
    if len(labels) != len(set(labels)):
        raise ValueError(f"{svg_path.name}: duplicate detection IDs")

    detections: list[dict[str, object]] = []
    for label, rectangle in zip(labels, rectangles, strict=True):
        x1 = rectangle["x"]
        y1 = rectangle["y"]
        x2 = x1 + rectangle["width"]
        y2 = y1 + rectangle["height"]
        valid_bbox = (
            rectangle["width"] > 0
            and rectangle["height"] > 0
            and 0 <= x1 < x2 <= width
            and 0 <= y1 < y2 <= height
        )
        if not valid_bbox:
            issues.append(
                issue(
                    "ERROR",
                    "INVALID_BBOX",
                    "Detection bounding box is empty or outside the image.",
                    photo_id=photo_id,
                    detection_id=label,
                    bbox_xyxy=[x1, y1, x2, y2],
                )
            )

        detections.append(
            {
                "photo_id": photo_id,
                "detection_id": label,
                "person_id": f"{photo_id}__p{label}",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "bbox_width": rectangle["width"],
                "bbox_height": rectangle["height"],
                "x1_norm": x1 / width,
                "y1_norm": y1 / height,
                "x2_norm": x2 / width,
                "y2_norm": y2 / height,
                "detector_confidence": rectangle["detector_confidence"],
                "bbox_valid": valid_bbox,
            }
        )

    photo = {
        "photo_id": photo_id,
        "svg_path": str(svg_path),
        "width": width,
        "height": height,
        "orientation": "landscape" if width > height else "portrait",
        "n_detections": len(detections),
        "svg_sha256": hashlib.sha256(svg_bytes).hexdigest(),
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "image_extension": image_extension,
    }
    return (
        ParsedSvg(
            photo=photo,
            detections=detections,
            image_bytes=image_bytes,
            image_extension=image_extension,
            svg_path=svg_path,
        ),
        issues,
    )


def parse_svg_directory(
    svg_dir: Path,
) -> tuple[list[ParsedSvg], list[dict[str, object]]]:
    """Parse every SVG in a directory in natural numeric order."""

    svg_dir = Path(svg_dir)

    def numeric_key(path: Path) -> tuple[str, int]:
        match = re.search(r"(\d+)$", path.stem)
        return path.stem, int(match.group(1)) if match else -1

    parsed: list[ParsedSvg] = []
    issues: list[dict[str, object]] = []
    for svg_path in sorted(svg_dir.glob("*.svg"), key=numeric_key):
        try:
            item, item_issues = parse_svg(svg_path)
            parsed.append(item)
            issues.extend(item_issues)
        except (ET.ParseError, OSError, ValueError) as exc:
            issues.append(
                issue(
                    "ERROR",
                    "SVG_PARSE_ERROR",
                    str(exc),
                    svg_path=str(svg_path),
                )
            )
    return parsed, issues


def extract_images(parsed_svgs: Iterable[ParsedSvg], image_dir: Path) -> None:
    """Write embedded images using deterministic photo IDs."""

    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    for parsed in parsed_svgs:
        destination = image_dir / f"{parsed.photo['photo_id']}{parsed.image_extension}"
        if not destination.exists() or destination.read_bytes() != parsed.image_bytes:
            destination.write_bytes(parsed.image_bytes)
