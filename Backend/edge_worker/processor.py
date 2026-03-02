from __future__ import annotations

import base64
from datetime import datetime, timezone
from typing import Any, Optional

import cv2
import numpy as np


def _to_iso_utc(value: datetime | str) -> str:
    if isinstance(value, str):
        return value

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _clamp_bbox(
    bbox: dict[str, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1 = int(max(0, min(width - 1, bbox.get("x1", 0))))
    y1 = int(max(0, min(height - 1, bbox.get("y1", 0))))
    x2 = int(max(1, min(width, bbox.get("x2", width))))
    y2 = int(max(1, min(height, bbox.get("y2", height))))

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return x1, y1, x2, y2


def build_alert_payload(
    frame_bgr: np.ndarray,
    confidence: float,
    timestamp_utc: datetime | str,
    incident_type: str,
    camera_id: str,
    location: dict[str, float],
    bbox: Optional[dict[str, float]] = None,
    confidence_threshold: float = 0.70,
) -> dict[str, Any] | None:
    if confidence < confidence_threshold:
        return None

    working_frame = frame_bgr
    if bbox:
        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = _clamp_bbox(bbox, width=width, height=height)
        cropped = frame_bgr[y1:y2, x1:x2]
        if cropped.size > 0:
            working_frame = cropped

    ok, encoded = cv2.imencode(".jpg", working_frame)
    if not ok:
        return None

    snapshot_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

    return {
        "type": incident_type,
        "confidence": float(confidence),
        "snapshot": snapshot_b64,
        "status": "pending",
        "timestamp": _to_iso_utc(timestamp_utc),
        "camera_id": camera_id,
        "location": {
            "latitude": float(location["latitude"]),
            "longitude": float(location["longitude"]),
        },
    }
