"""FastAPI application factory for group-form prediction."""

from __future__ import annotations

import os
import time
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from conference_social_sensing.modeling.predictor import GroupFormPredictor

MAX_IMAGE_BYTES = 10 * 1024 * 1024


class GroupGeometry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_size: int = Field(gt=0)
    geom_center_x: float = Field(ge=0, le=1)
    geom_center_y: float = Field(ge=0, le=1)
    geom_union_width: float = Field(gt=0, le=1)
    geom_union_height: float = Field(gt=0, le=1)
    geom_union_area: float = Field(gt=0, le=1)
    geom_mean_member_width: float = Field(gt=0, le=1)
    geom_mean_member_height: float = Field(gt=0, le=1)
    geom_center_std_x: float = Field(ge=0, le=1)
    geom_center_std_y: float = Field(ge=0, le=1)
    geom_pair_distance_mean: float = Field(ge=0, le=1)
    geom_pair_distance_max: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    labels: list[str]
    probabilities: dict[str, float]
    threshold: float
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    labels: list[str]


def _default_checkpoint() -> Path:
    configured = os.environ.get("CONFERENCE_MODEL_CHECKPOINT")
    if configured:
        return Path(configured).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "artifacts/group_model/best_model.pt"


def create_app(
    checkpoint_path: Path | str | None = None,
    *,
    device: str = "auto",
    predictor: GroupFormPredictor | Any | None = None,
) -> FastAPI:
    """Build an app; predictor injection keeps API tests fast and isolated."""
    selected_device = (
        os.environ.get("CONFERENCE_MODEL_DEVICE", "auto")
        if device == "auto"
        else device
    )
    loaded_predictor = predictor or GroupFormPredictor(
        checkpoint_path or _default_checkpoint(), device=selected_device
    )
    app = FastAPI(
        title="Conference Group Form API",
        version="0.1.0",
        description=(
            "MVP inference service for an already-proposed interaction group. "
            "It predicts the A/B/C/D group-form labels from a context crop and geometry."
        ),
    )
    app.state.predictor = loaded_predictor

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model="frozen_resnet18_plus_geometry",
            device=str(loaded_predictor.device),
            labels=[
                label.removeprefix("form_") for label in loaded_predictor.label_columns
            ],
        )

    @app.post("/predict-group", response_model=PredictionResponse)
    async def predict_group(
        image: Annotated[UploadFile, File(description="JPEG, PNG, or WebP group crop")],
        geometry_json: Annotated[
            str,
            Form(
                description="JSON object containing the required normalized geometry features"
            ),
        ],
        threshold: Annotated[float | None, Form(ge=0, le=1)] = None,
    ) -> PredictionResponse:
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=415, detail="Uploaded file must be an image"
            )
        payload = await image.read(MAX_IMAGE_BYTES + 1)
        if len(payload) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit")
        try:
            geometry = GroupGeometry.model_validate_json(geometry_json)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=error.errors()) from error
        try:
            with Image.open(BytesIO(payload)) as source:
                crop = source.convert("RGB")
                crop.load()
        except (UnidentifiedImageError, OSError) as error:
            raise HTTPException(
                status_code=400, detail="Image payload could not be decoded"
            ) from error

        started = time.perf_counter()
        prediction = loaded_predictor.predict(
            crop,
            geometry.model_dump(),
            threshold=threshold,
        )
        return PredictionResponse(
            **prediction,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    return app
