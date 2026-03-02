from __future__ import annotations

import threading
import time
from typing import Any, Optional

import requests

from edge_worker.offline_db import OfflineAlertStore


class AlertSyncWorker:
    def __init__(
        self,
        backend_base_url: str,
        alert_endpoint: str = "/api/incidents/alert",
        sync_interval_seconds: float = 5.0,
        db_path: str = "buffer.db",
        request_timeout_seconds: float = 8.0,
    ) -> None:
        self.backend_base_url = backend_base_url.rstrip("/")
        self.alert_endpoint = alert_endpoint if alert_endpoint.startswith("/") else f"/{alert_endpoint}"
        self.sync_interval_seconds = max(1.0, float(sync_interval_seconds))
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        self.store = OfflineAlertStore(db_path=db_path)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def endpoint_url(self) -> str:
        return f"{self.backend_base_url}{self.alert_endpoint}"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[edge] sync worker started ({self.endpoint_url})")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        print("[edge] sync worker stopped")

    def send_or_buffer(self, payload: dict[str, Any]) -> bool:
        if self._try_send(payload):
            return True

        row_id = self.store.save_locally(payload)
        print(f"[edge] alert buffered locally (id={row_id})")
        return False

    def _try_send(self, payload: dict[str, Any]) -> bool:
        try:
            response = requests.post(self.endpoint_url, json=payload, timeout=self.request_timeout_seconds)
            response.raise_for_status()
            return True
        except Exception as exc:
            print(f"[edge] alert send failed: {exc}")
            return False

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            pending_rows = self.store.get_pending(limit=100)
            if pending_rows:
                print(f"[edge] retrying {len(pending_rows)} buffered alert(s)")

            for row_id, payload in pending_rows:
                if self._try_send(payload):
                    self.store.remove_local_alert(row_id)
                    print(f"[edge] buffered alert delivered (id={row_id})")

            self._stop_event.wait(self.sync_interval_seconds)
