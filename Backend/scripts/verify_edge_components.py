from __future__ import annotations

import base64
import gc
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from edge_worker.offline_db import OfflineAlertStore
from edge_worker.processor import build_alert_payload


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_processor() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[:, :] = (30, 120, 220)

    dropped = build_alert_payload(
        frame_bgr=frame,
        confidence=0.40,
        timestamp_utc=datetime.now(timezone.utc),
        incident_type="fight",
        camera_id="cam_test",
        location={"latitude": 1.3000, "longitude": 103.8000},
        confidence_threshold=0.70,
    )
    assert_true(dropped is None, "Low-confidence payload should be dropped")

    kept = build_alert_payload(
        frame_bgr=frame,
        confidence=0.95,
        timestamp_utc=datetime.now(timezone.utc),
        incident_type="fight",
        camera_id="cam_test",
        location={"latitude": 1.3000, "longitude": 103.8000},
        bbox={"x1": 10, "y1": 10, "x2": 80, "y2": 80},
        confidence_threshold=0.70,
    )
    assert_true(kept is not None, "High-confidence payload should be kept")
    assert_true("snapshot" in kept, "Payload must contain snapshot")

    image_bytes = base64.b64decode(kept["snapshot"])
    assert_true(image_bytes[:2] == b"\xff\xd8", "Snapshot must be JPEG bytes")


def test_offline_store() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "buffer.db"
        store = OfflineAlertStore(db_path=str(db_path))

        payload = {
            "incident_type": "fight",
            "confidence": 0.88,
            "camera_id": "cam_test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "location": {"latitude": 1.3, "longitude": 103.8},
        }

        row_id = store.save_locally(payload)
        assert_true(row_id > 0, "Inserted row id should be positive")

        rows = store.get_pending(limit=10)
        assert_true(len(rows) == 1, "Expected one pending row")
        assert_true(rows[0][0] == row_id, "Expected matching row id")

        store.remove_local_alert(row_id)
        rows_after_delete = store.get_pending(limit=10)
        assert_true(len(rows_after_delete) == 0, "Pending rows should be empty after delete")
        del store
        gc.collect()


def main() -> int:
    test_processor()
    test_offline_store()
    print("PASS: edge processor and offline store checks succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
