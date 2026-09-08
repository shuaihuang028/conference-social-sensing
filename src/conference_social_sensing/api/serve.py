"""Local development server entry point."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/group_model/best_model.pt")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    os.environ["CONFERENCE_MODEL_CHECKPOINT"] = str(args.checkpoint.resolve())
    os.environ["CONFERENCE_MODEL_DEVICE"] = args.device
    uvicorn.run(
        "conference_social_sensing.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
