from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import requests
from dotenv import load_dotenv


def load_keras_model_compat(model_path: str):
    """Load Keras model with fallbacks for legacy .h5 compatibility."""
    errors: list[str] = []

    try:
        from tensorflow.keras.models import load_model as tf_load_model

        return tf_load_model(model_path, compile=False)
    except Exception as exc:
        errors.append(f"tensorflow.keras failed: {exc}")

    try:
        import tf_keras

        return tf_keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        errors.append(f"tf_keras failed: {exc}")

    try:
        from keras.models import load_model as keras_load_model

        return keras_load_model(model_path, compile=False)
    except Exception as exc:
        errors.append(f"keras failed: {exc}")

    raise RuntimeError(
        "Unable to load model. This is usually a Keras version mismatch for legacy .h5 files. "
        "Re-save model as .keras using the current environment, or install tf-keras. "
        f"Details: {' | '.join(errors)}"
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CameraConfig:
    camera_id: str
    stream_url: str
    latitude: float
    longitude: float
    active: bool = True
    reconnect_delay_seconds: float = 2.0


@dataclass
class EdgeInferenceConfig:
    backend_base_url: str = "http://127.0.0.1:8000"
    yolo_weights: str = "yolov8n.pt"
    yolo_conf_threshold: float = 0.25
    yolo_device: str = "cpu"

    # Fight gating
    min_people_for_action: int = 2
    proximity_distance_px_threshold: float = 80.0
    overlap_ratio_threshold: float = 0.1
    motion_score_threshold: float = 0.25

    # Temporal windows
    target_fps: float = 12.0
    buffer_seconds: float = 12.0
    action_window_seconds: float = 4.0
    inference_interval_seconds: float = 1.0

    # Fight decision
    fight_threshold_t: float = 0.3
    smoothing_alpha: float = 0.6
    cooldown_seconds: int = 30

    # Evidence package
    keyframe_count: int = 3
    incident_clip_seconds: int = 10
    artifacts_root: str = "./artifacts"
    heartbeat_interval_seconds: float = 5.0

    # Armed threat detection (YOLOv8 weapon detector)
    weapon_weights: Optional[str] = "best.pt"
    weapon_conf_threshold: float = 0.5
    weapon_target_fps: float = 7.0  # run weapon model at ~5-10 FPS
    weapon_persist_n: int = 3
    weapon_window_m: int = 5
    armed_min_people_context: int = 2
    armed_cooldown_seconds: int = 45
    weapon_class_keywords: tuple[str, ...] = (
        "weapon",
        "gun",
        "pistol",
        "rifle",
        "knife",
        "machete",
        "firearm",
    )

    # MobileNetV2 violence classifier (trained on violent video dataset)
    mobilenet_model_path: Optional[str] = None
    mobilenet_backend: str = "auto"  # auto | keras | onnx
    mobilenet_input_size: int = 64
    mobilenet_violence_class_index: int = 1
    mobilenet_preprocess_mode: str = "zero_one"  # keras | zero_one
    max_clip_frames_for_action: int = 16
    sequence_length: int = 16


@dataclass
class CameraRuntime:
    prev_gray: Optional[np.ndarray] = None
    frame_buffer: deque[np.ndarray] = field(default_factory=deque)
    ts_buffer: deque[datetime] = field(default_factory=deque)
    ema_fight_score: float = 0.0
    last_inference_at: Optional[datetime] = None
    last_weapon_inference_at: Optional[datetime] = None
    last_heartbeat_at: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    armed_cooldown_until: Optional[datetime] = None
    fight_incident_active: bool = False
    armed_incident_active: bool = False
    weapon_hits: deque[bool] = field(default_factory=deque)


class YoloPersonDetector:
    def __init__(self, weights: str, conf_threshold: float, device: str = "cpu") -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - dependency/runtime error
            raise RuntimeError("ultralytics is required for YOLOv8 detection") from exc

        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold
        self.device = device

    def detect_people(self, frame: np.ndarray) -> list[dict[str, float]]:
        """Return person bboxes as [{x1,y1,x2,y2,confidence}, ...]."""
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            classes=[0],  # class 0: person
            verbose=False,
            device=self.device,
        )
        if not results:
            return []

        out: list[dict[str, float]] = []
        boxes = results[0].boxes
        if boxes is None:
            return out

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            out.append(
                {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "confidence": float(c),
                }
            )
        return out


class YoloWeaponDetector:
    """Weapon detector wrapper for YOLOv8 custom weights (e.g., best.pt)."""

    def __init__(
        self,
        weights: str,
        conf_threshold: float,
        device: str = "cpu",
        class_keywords: tuple[str, ...] = (),
    ) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("ultralytics is required for YOLOv8 weapon detection") from exc

        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold
        self.device = device
        self.class_keywords = tuple(k.lower() for k in class_keywords)

    def detect_weapons(self, frame: np.ndarray) -> list[dict[str, Any]]:
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
        )
        if not results:
            return []

        out: list[dict[str, Any]] = []
        boxes = results[0].boxes
        if boxes is None:
            return out

        names = getattr(results[0], "names", {}) or {}
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(confs), dtype=int)

        for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clss):
            class_name = str(names.get(int(cls_id), f"class_{int(cls_id)}")).lower()

            if self.class_keywords and not any(k in class_name for k in self.class_keywords):
                # If names are available and no keyword matches, skip.
                # For single-class custom weapon models, class_name usually still contains weapon-like text.
                continue

            out.append(
                {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "confidence": float(c),
                    "class_id": int(cls_id),
                    "class_name": class_name,
                }
            )

        # Fallback: if class-name filtering removed everything, keep raw detections above threshold.
        if not out and len(xyxy) > 0:
            for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clss):
                out.append(
                    {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "confidence": float(c),
                        "class_id": int(cls_id),
                        "class_name": str(names.get(int(cls_id), f"class_{int(cls_id)}")).lower(),
                    }
                )

        return out


class MobileNetV2FightClassifier:
    """Clip classifier that uses a MobileNetV2 violence model and returns p_fight in [0,1]."""

    def __init__(
        self,
        model_path: Optional[str],
        backend: str,
        input_size: int,
        violence_class_index: int,
        preprocess_mode: str,
        max_clip_frames: int,
        sequence_length: int,
    ) -> None:
        self.enabled = bool(model_path)
        self.model_path = model_path
        self.backend = backend
        self.input_size = input_size
        self.violence_class_index = violence_class_index
        self.preprocess_mode = preprocess_mode
        self.max_clip_frames = max_clip_frames

        self.model = None
        self.session = None
        self.onnx_input_name: Optional[str] = None
        self._keras_preprocess = None
        self.sequence_mode = False
        self.sequence_length = max(1, sequence_length)

        if not self.enabled:
            return

        chosen = self.backend
        if chosen == "auto":
            chosen = "onnx" if str(self.model_path).lower().endswith(".onnx") else "keras"

        if chosen == "onnx":
            try:
                import onnxruntime as ort
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("onnxruntime is required for ONNX MobileNetV2 models") from exc

            self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            self.onnx_input_name = self.session.get_inputs()[0].name
            self.backend = "onnx"
        else:
            try:
                from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("tensorflow is required for Keras MobileNetV2 models") from exc

            self.model = load_keras_model_compat(str(self.model_path))
            self._keras_preprocess = preprocess_input
            input_shape = getattr(self.model, "input_shape", None)
            if isinstance(input_shape, tuple) and len(input_shape) == 5:
                # Expected shape: (None, T, H, W, C)
                self.sequence_mode = True
                if input_shape[1] is not None:
                    self.sequence_length = int(input_shape[1])
            self.backend = "keras"

    def predict_fight_probability(self, clip_path: str, person_boxes: Optional[list[dict[str, float]]] = None) -> float:
        if not self.enabled:
            return 0.0

        frames = self._sample_clip_frames(clip_path, self.max_clip_frames)
        if not frames:
            return 0.0

        if self.backend == "keras" and self.sequence_mode:
            return self._predict_sequence_probability(frames, person_boxes)

        probs: list[float] = []
        for frame in frames:
            crops = self._extract_person_crops(frame, person_boxes)
            if not crops:
                crops = [frame]

            frame_max = 0.0
            for crop in crops:
                inp = self._prepare_input(crop)

                if self.backend == "onnx":
                    output = self.session.run(None, {self.onnx_input_name: inp})[0]
                else:
                    output = self.model.predict(inp, verbose=0)

                p = self._parse_probability(output)
                frame_max = max(frame_max, p)

            probs.append(frame_max)

        if not probs:
            return 0.0

        k = max(1, len(probs) // 4)
        top_scores = sorted(probs)[-k:]
        return float(np.mean(top_scores))

    def _predict_sequence_probability(
        self,
        frames: list[np.ndarray],
        person_boxes: Optional[list[dict[str, float]]],
    ) -> float:
        seq_len = self.sequence_length
        sampled = self._resample_to_sequence(frames, seq_len)

        # If people are detected, focus on their union region for the whole sequence.
        if person_boxes:
            sampled = [self._crop_union_box(fr, person_boxes) for fr in sampled]

        processed = [self._prepare_frame(fr) for fr in sampled]
        clip_tensor = np.stack(processed, axis=0)  # (T, H, W, C)
        clip_tensor = np.expand_dims(clip_tensor, axis=0).astype(np.float32)  # (1, T, H, W, C)

        output = self.model.predict(clip_tensor, verbose=0)
        return self._parse_probability(output)

    def _prepare_input(self, image_bgr: np.ndarray) -> np.ndarray:
        arr = self._prepare_frame(image_bgr)
        return np.expand_dims(arr, axis=0).astype(np.float32)

    def _prepare_frame(self, image_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.preprocess_mode == "zero_one":
            return rgb / 255.0

        if self._keras_preprocess is not None:
            return self._keras_preprocess(rgb)

        return (rgb / 127.5) - 1.0

    def _parse_probability(self, output: Any) -> float:
        arr = np.array(output)
        arr = np.squeeze(arr)

        if arr.ndim == 0:
            return float(np.clip(arr.item(), 0.0, 1.0))

        if arr.ndim == 1:
            if arr.shape[0] == 1:
                return float(np.clip(arr[0], 0.0, 1.0))
            idx = max(0, min(self.violence_class_index, arr.shape[0] - 1))
            return float(np.clip(arr[idx], 0.0, 1.0))

        flat = arr.reshape(-1)
        if flat.shape[0] == 1:
            return float(np.clip(flat[0], 0.0, 1.0))

        idx = max(0, min(self.violence_class_index, flat.shape[0] - 1))
        return float(np.clip(flat[idx], 0.0, 1.0))

    @staticmethod
    def _sample_clip_frames(clip_path: str, max_frames: int) -> list[np.ndarray]:
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            return []

        all_frames: list[np.ndarray] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                all_frames.append(frame)
        finally:
            cap.release()

        if not all_frames:
            return []

        if len(all_frames) <= max_frames:
            return all_frames

        idxs = np.linspace(0, len(all_frames) - 1, num=max_frames, dtype=int)
        return [all_frames[i] for i in idxs.tolist()]

    @staticmethod
    def _resample_to_sequence(frames: list[np.ndarray], sequence_length: int) -> list[np.ndarray]:
        if not frames:
            return []

        if len(frames) == sequence_length:
            return frames

        idxs = np.linspace(0, len(frames) - 1, num=sequence_length, dtype=int)
        return [frames[i] for i in idxs.tolist()]

    @staticmethod
    def _crop_union_box(frame: np.ndarray, boxes: list[dict[str, float]]) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = min(max(0, int(b.get("x1", 0))) for b in boxes)
        y1 = min(max(0, int(b.get("y1", 0))) for b in boxes)
        x2 = max(min(w, int(b.get("x2", w))) for b in boxes)
        y2 = max(min(h, int(b.get("y2", h))) for b in boxes)

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_w = int(0.2 * bw)
        pad_h = int(0.2 * bh)

        xx1 = max(0, x1 - pad_w)
        yy1 = max(0, y1 - pad_h)
        xx2 = min(w, x2 + pad_w)
        yy2 = min(h, y2 + pad_h)

        crop = frame[yy1:yy2, xx1:xx2]
        return crop if crop.size > 0 else frame

    @staticmethod
    def _extract_person_crops(frame: np.ndarray, boxes: Optional[list[dict[str, float]]]) -> list[np.ndarray]:
        if not boxes:
            return []

        h, w = frame.shape[:2]
        crops: list[np.ndarray] = []
        for b in boxes:
            x1 = int(max(0, b.get("x1", 0)))
            y1 = int(max(0, b.get("y1", 0)))
            x2 = int(min(w, b.get("x2", w)))
            y2 = int(min(h, b.get("y2", h)))

            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            pad_w = int(0.15 * bw)
            pad_h = int(0.15 * bh)

            xx1 = max(0, x1 - pad_w)
            yy1 = max(0, y1 - pad_h)
            xx2 = min(w, x2 + pad_w)
            yy2 = min(h, y2 + pad_h)

            crop = frame[yy1:yy2, xx1:xx2]
            if crop.size > 0:
                crops.append(crop)

        return crops


class MultiCameraEdgeService:
    def __init__(self, cameras: list[CameraConfig], cfg: EdgeInferenceConfig) -> None:
        self.cameras = [c for c in cameras if c.active]
        self.cfg = cfg
        self.runtime: dict[str, CameraRuntime] = {c.camera_id: CameraRuntime() for c in self.cameras}
        for rt in self.runtime.values():
            rt.weapon_hits = deque(maxlen=cfg.weapon_window_m)

        self.detector = YoloPersonDetector(
            weights=cfg.yolo_weights,
            conf_threshold=cfg.yolo_conf_threshold,
            device=cfg.yolo_device,
        )
        self.weapon_detector = None
        if cfg.weapon_weights:
            self.weapon_detector = YoloWeaponDetector(
                weights=cfg.weapon_weights,
                conf_threshold=cfg.weapon_conf_threshold,
                device=cfg.yolo_device,
                class_keywords=cfg.weapon_class_keywords,
            )
        self.classifier = MobileNetV2FightClassifier(
            model_path=cfg.mobilenet_model_path,
            backend=cfg.mobilenet_backend,
            input_size=cfg.mobilenet_input_size,
            violence_class_index=cfg.mobilenet_violence_class_index,
            preprocess_mode=cfg.mobilenet_preprocess_mode,
            max_clip_frames=cfg.max_clip_frames_for_action,
            sequence_length=cfg.sequence_length,
        )

        Path(cfg.artifacts_root).mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        threads: list[threading.Thread] = []
        for cam in self.cameras:
            th = threading.Thread(target=self._run_camera_loop, args=(cam,), daemon=True)
            th.start()
            threads.append(th)

        for th in threads:
            th.join()

    def _run_camera_loop(self, cam: CameraConfig) -> None:
        interval = 1.0 / max(self.cfg.target_fps, 1.0)
        while True:
            cap = cv2.VideoCapture(cam.stream_url)
            if not cap.isOpened():
                print(f"[{cam.camera_id}] stream open failed; retrying...")
                time.sleep(cam.reconnect_delay_seconds)
                continue

            print(f"[{cam.camera_id}] stream connected")
            rt = self.runtime[cam.camera_id]
            try:
                while True:
                    t0 = time.time()
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        print(f"[{cam.camera_id}] frame read failed; reconnecting...")
                        break

                    now = utc_now()
                    self._process_frame(cam, rt, frame, now)

                    elapsed = time.time() - t0
                    sleep_for = interval - elapsed
                    if sleep_for > 0:
                        time.sleep(sleep_for)
            finally:
                cap.release()
                time.sleep(cam.reconnect_delay_seconds)

    def _process_frame(self, cam: CameraConfig, rt: CameraRuntime, frame: np.ndarray, now: datetime) -> None:
        boxes = self.detector.detect_people(frame)
        people_count = len(boxes)

        min_dist, overlap_ratio = self._bbox_proximity_metrics(boxes)
        motion_score, gray = self._motion_score(frame, rt.prev_gray)
        rt.prev_gray = gray

        self._maybe_post_heartbeat(cam, rt, now, people_count, motion_score)

        self._push_buffer(rt, frame, now)

        # Armed threat path (higher priority)
        armed_triggered = self._maybe_process_armed_threat(
            cam=cam,
            rt=rt,
            now=now,
            people_count=people_count,
            person_boxes=boxes,
        )
        if armed_triggered:
            return

        in_cooldown = rt.cooldown_until is not None and now < rt.cooldown_until
        if in_cooldown:
            rt.fight_incident_active = False
            return

        close_people = (
            min_dist is not None
            and (min_dist <= self.cfg.proximity_distance_px_threshold or overlap_ratio >= self.cfg.overlap_ratio_threshold)
        )
        high_motion = motion_score >= self.cfg.motion_score_threshold

        should_infer = people_count >= self.cfg.min_people_for_action and (close_people or high_motion)
        if not should_infer:
            return

        if rt.last_inference_at and (now - rt.last_inference_at).total_seconds() < self.cfg.inference_interval_seconds:
            return

        clip_frames, clip_start, clip_end = self._latest_clip(rt)
        if not clip_frames:
            return

        rt.last_inference_at = now
        clip_path = self._save_clip(cam.camera_id, clip_frames, clip_start, suffix="window")

        p_fight = self.classifier.predict_fight_probability(clip_path, person_boxes=boxes)
        rt.ema_fight_score = self.cfg.smoothing_alpha * p_fight + (1.0 - self.cfg.smoothing_alpha) * rt.ema_fight_score

        triggered = rt.ema_fight_score > self.cfg.fight_threshold_t
        if not triggered:
            return

        rt.fight_incident_active = True

        evidence_clip_frames, ev_start, _ = self._latest_clip(rt, seconds=self.cfg.incident_clip_seconds)
        evidence_clip_path = self._save_clip(cam.camera_id, evidence_clip_frames, ev_start, suffix="incident")
        keyframe_paths = self._save_keyframes(cam.camera_id, evidence_clip_frames)

        payload = {
            "incident_type": "fight",
            "confidence": float(rt.ema_fight_score),
            "camera_id": cam.camera_id,
            "timestamp": now.isoformat(),
            "location": {
                "latitude": cam.latitude,
                "longitude": cam.longitude,
            },
            "pipeline_stage": "edge",
            "frame_stats": {
                "timestamp": now.isoformat(),
                "people_count": people_count,
                "person_bboxes": boxes,
                "min_bbox_distance_px": min_dist,
                "overlap_ratio": overlap_ratio,
                "motion_score": motion_score,
            },
            "inference_window": {
                "start_ts": clip_start.isoformat(),
                "end_ts": clip_end.isoformat(),
                "duration_seconds": self.cfg.action_window_seconds,
                "sampled_fps": self.cfg.target_fps,
                "frame_count": len(clip_frames),
                "source_clip_uri": clip_path,
            },
            "fight_inference": {
                "model_name": "mobilenetv2-violence",
                "raw_fight_probability": float(p_fight),
                "smoothed_fight_probability": float(rt.ema_fight_score),
                "threshold": float(self.cfg.fight_threshold_t),
                "triggered": True,
                "inferred_at": now.isoformat(),
            },
            "evidence": {
                "keyframe_uris": keyframe_paths,
                "clip_uri": evidence_clip_path,
                "clip_duration_seconds": self.cfg.incident_clip_seconds,
                "generated_at": now.isoformat(),
            },
        }

        self._post_alert(payload)
        rt.cooldown_until = now + timedelta(seconds=self.cfg.cooldown_seconds)
        rt.fight_incident_active = False

    def _maybe_process_armed_threat(
        self,
        cam: CameraConfig,
        rt: CameraRuntime,
        now: datetime,
        people_count: int,
        person_boxes: list[dict[str, float]],
    ) -> bool:
        if self.weapon_detector is None:
            return False

        in_armed_cooldown = rt.armed_cooldown_until is not None and now < rt.armed_cooldown_until
        if in_armed_cooldown:
            rt.armed_incident_active = False
            return False

        min_interval = 1.0 / max(self.cfg.weapon_target_fps, 0.1)
        if rt.last_weapon_inference_at and (now - rt.last_weapon_inference_at).total_seconds() < min_interval:
            return False

        rt.last_weapon_inference_at = now
        weapon_detections = self.weapon_detector.detect_weapons(self.runtime[cam.camera_id].frame_buffer[-1]) if rt.frame_buffer else []
        has_weapon = len(weapon_detections) > 0
        rt.weapon_hits.append(has_weapon)

        has_required_window = len(rt.weapon_hits) >= self.cfg.weapon_window_m
        persistent_weapon = has_required_window and sum(rt.weapon_hits) >= self.cfg.weapon_persist_n
        context_ok = people_count >= self.cfg.armed_min_people_context

        if not (persistent_weapon and context_ok):
            return False

        rt.armed_incident_active = True

        evidence_clip_frames, ev_start, ev_end = self._latest_clip(rt, seconds=self.cfg.incident_clip_seconds)
        evidence_clip_path = self._save_clip(cam.camera_id, evidence_clip_frames, ev_start, suffix="armed_incident")
        keyframe_paths = self._save_keyframes(cam.camera_id, evidence_clip_frames)

        max_weapon_conf = max((float(d.get("confidence", 0.0)) for d in weapon_detections), default=self.cfg.weapon_conf_threshold)

        payload = {
            "incident_type": "armed_threat",
            "confidence": float(max_weapon_conf),
            "camera_id": cam.camera_id,
            "timestamp": now.isoformat(),
            "location": {
                "latitude": cam.latitude,
                "longitude": cam.longitude,
            },
            "pipeline_stage": "edge",
            "frame_stats": {
                "timestamp": now.isoformat(),
                "people_count": people_count,
                "person_bboxes": person_boxes,
                "motion_score": None,
            },
            "inference_window": {
                "start_ts": ev_start.isoformat(),
                "end_ts": ev_end.isoformat(),
                "duration_seconds": self.cfg.incident_clip_seconds,
                "sampled_fps": self.cfg.target_fps,
                "frame_count": len(evidence_clip_frames),
                "source_clip_uri": evidence_clip_path,
            },
            "fight_inference": {
                "model_name": "yolov8-weapon",
                "raw_fight_probability": float(max_weapon_conf),
                "smoothed_fight_probability": float(max_weapon_conf),
                "threshold": float(self.cfg.weapon_conf_threshold),
                "triggered": True,
                "inferred_at": now.isoformat(),
            },
            "evidence": {
                "keyframe_uris": keyframe_paths,
                "clip_uri": evidence_clip_path,
                "clip_duration_seconds": self.cfg.incident_clip_seconds,
                "generated_at": now.isoformat(),
            },
        }

        self._post_alert(payload)
        rt.armed_cooldown_until = now + timedelta(seconds=self.cfg.armed_cooldown_seconds)
        rt.armed_incident_active = False
        rt.weapon_hits.clear()
        return True

    def _maybe_post_heartbeat(
        self,
        cam: CameraConfig,
        rt: CameraRuntime,
        now: datetime,
        people_count: int,
        motion_score: float,
    ) -> None:
        if rt.last_heartbeat_at and (now - rt.last_heartbeat_at).total_seconds() < self.cfg.heartbeat_interval_seconds:
            return

        payload = {
            "camera_id": cam.camera_id,
            "timestamp": now.isoformat(),
            "input_fps": self.cfg.target_fps,
            "processed_fps": self.cfg.target_fps,
            "dropped_frames": 0,
            "latest_people_count": people_count,
            "latest_motion_score": motion_score,
        }

        url = f"{self.cfg.backend_base_url.rstrip('/')}/api/edge/cameras/{cam.camera_id}/heartbeat"
        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            rt.last_heartbeat_at = now
        except Exception as exc:
            print(f"[edge] heartbeat failed ({cam.camera_id}): {exc}")

    def _post_alert(self, payload: dict[str, Any]) -> None:
        url = f"{self.cfg.backend_base_url.rstrip('/')}/api/incidents/alert"
        try:
            response = requests.post(url, json=payload, timeout=8)
            response.raise_for_status()
            print(f"[edge] alert sent: {response.json()}")
        except Exception as exc:
            print(f"[edge] failed to send alert: {exc}")

    def _push_buffer(self, rt: CameraRuntime, frame: np.ndarray, ts: datetime) -> None:
        rt.frame_buffer.append(frame)
        rt.ts_buffer.append(ts)

        horizon = timedelta(seconds=self.cfg.buffer_seconds)
        while rt.ts_buffer and (ts - rt.ts_buffer[0]) > horizon:
            rt.ts_buffer.popleft()
            rt.frame_buffer.popleft()

    def _latest_clip(self, rt: CameraRuntime, seconds: Optional[int] = None) -> tuple[list[np.ndarray], datetime, datetime]:
        if not rt.ts_buffer:
            now = utc_now()
            return [], now, now

        window_seconds = float(seconds if seconds is not None else self.cfg.action_window_seconds)
        end_ts = rt.ts_buffer[-1]
        start_ts = end_ts - timedelta(seconds=window_seconds)

        clip_frames: list[np.ndarray] = []
        for ts, frame in zip(rt.ts_buffer, rt.frame_buffer):
            if ts >= start_ts:
                clip_frames.append(frame)

        if not clip_frames:
            clip_frames = [rt.frame_buffer[-1]]
            start_ts = rt.ts_buffer[-1]

        return clip_frames, start_ts, end_ts

    def _save_clip(self, camera_id: str, frames: list[np.ndarray], start_ts: datetime, suffix: str) -> str:
        if not frames:
            return ""

        cam_dir = Path(self.cfg.artifacts_root) / camera_id
        cam_dir.mkdir(parents=True, exist_ok=True)
        name = f"{start_ts.strftime('%Y%m%dT%H%M%S')}_{suffix}.mp4"
        out_path = cam_dir / name

        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.cfg.target_fps,
            (w, h),
        )
        for fr in frames:
            writer.write(fr)
        writer.release()
        return str(out_path)

    def _save_keyframes(self, camera_id: str, frames: list[np.ndarray]) -> list[str]:
        if not frames:
            return []

        cam_dir = Path(self.cfg.artifacts_root) / camera_id
        cam_dir.mkdir(parents=True, exist_ok=True)

        count = min(self.cfg.keyframe_count, len(frames))
        idxs = np.linspace(0, len(frames) - 1, num=count, dtype=int).tolist()

        out_paths: list[str] = []
        stamp = utc_now().strftime("%Y%m%dT%H%M%S")
        for i, idx in enumerate(idxs):
            path = cam_dir / f"{stamp}_keyframe_{i+1}.jpg"
            cv2.imwrite(str(path), frames[idx])
            out_paths.append(str(path))
        return out_paths

    @staticmethod
    def _motion_score(frame: np.ndarray, prev_gray: Optional[np.ndarray]) -> tuple[float, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            return 0.0, gray

        diff = cv2.absdiff(gray, prev_gray)
        score = float(np.mean(diff) / 255.0)
        return score, gray

    @staticmethod
    def _bbox_proximity_metrics(boxes: list[dict[str, float]]) -> tuple[Optional[float], float]:
        if len(boxes) < 2:
            return None, 0.0

        def center(b: dict[str, float]) -> tuple[float, float]:
            return ((b["x1"] + b["x2"]) / 2.0, (b["y1"] + b["y2"]) / 2.0)

        def iou(a: dict[str, float], b: dict[str, float]) -> float:
            inter_x1 = max(a["x1"], b["x1"])
            inter_y1 = max(a["y1"], b["y1"])
            inter_x2 = min(a["x2"], b["x2"])
            inter_y2 = min(a["y2"], b["y2"])
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter = inter_w * inter_h
            area_a = max(0.0, (a["x2"] - a["x1"])) * max(0.0, (a["y2"] - a["y1"]))
            area_b = max(0.0, (b["x2"] - b["x1"])) * max(0.0, (b["y2"] - b["y1"]))
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        min_dist = float("inf")
        max_iou = 0.0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                c1 = center(boxes[i])
                c2 = center(boxes[j])
                dist = float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))
                min_dist = min(min_dist, dist)
                max_iou = max(max_iou, iou(boxes[i], boxes[j]))

        return min_dist, max_iou


def load_cameras_from_json(file_path: str) -> list[CameraConfig]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        CameraConfig(
            camera_id=item["camera_id"],
            stream_url=item["stream_url"],
            latitude=float(item["latitude"]),
            longitude=float(item["longitude"]),
            active=bool(item.get("active", True)),
        )
        for item in data
    ]


def load_config_from_env() -> EdgeInferenceConfig:
    backend_dir = Path(__file__).resolve().parents[1]
    load_dotenv(backend_dir / ".env")
    return EdgeInferenceConfig(
        backend_base_url=os.getenv("EDGE_BACKEND_URL", "http://127.0.0.1:8000"),
        yolo_weights=os.getenv("YOLO_WEIGHTS", "yolov8n.pt"),
        yolo_conf_threshold=float(os.getenv("YOLO_CONF_THRESHOLD", "0.25")),
        yolo_device=os.getenv("YOLO_DEVICE", "cpu"),
        min_people_for_action=int(os.getenv("MIN_PEOPLE_FOR_ACTION", "2")),
        proximity_distance_px_threshold=float(os.getenv("PROXIMITY_DISTANCE_PX_THRESHOLD", "80")),
        overlap_ratio_threshold=float(os.getenv("OVERLAP_RATIO_THRESHOLD", "0.1")),
        motion_score_threshold=float(os.getenv("MOTION_SCORE_THRESHOLD", "0.25")),
        target_fps=float(os.getenv("TARGET_FPS", "12")),
        buffer_seconds=float(os.getenv("BUFFER_SECONDS", "12")),
        action_window_seconds=float(os.getenv("ACTION_WINDOW_SECONDS", "4")),
        inference_interval_seconds=float(os.getenv("INFERENCE_INTERVAL_SECONDS", "1")),
        fight_threshold_t=float(os.getenv("FIGHT_THRESHOLD_T", "0.3")),
        smoothing_alpha=float(os.getenv("SMOOTHING_ALPHA", "0.6")),
        cooldown_seconds=int(os.getenv("COOLDOWN_SECONDS", "30")),
        keyframe_count=int(os.getenv("KEYFRAME_COUNT", "3")),
        incident_clip_seconds=int(os.getenv("INCIDENT_CLIP_SECONDS", "10")),
        artifacts_root=os.getenv("ARTIFACTS_ROOT", "./artifacts"),
        heartbeat_interval_seconds=float(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "5")),
        weapon_weights=os.getenv("WEAPON_WEIGHTS", "best.pt") or None,
        weapon_conf_threshold=float(os.getenv("WEAPON_CONF_THRESHOLD", "0.5")),
        weapon_target_fps=float(os.getenv("WEAPON_TARGET_FPS", "7")),
        weapon_persist_n=int(os.getenv("WEAPON_PERSIST_N", "3")),
        weapon_window_m=int(os.getenv("WEAPON_WINDOW_M", "5")),
        armed_min_people_context=int(os.getenv("ARMED_MIN_PEOPLE_CONTEXT", "2")),
        armed_cooldown_seconds=int(os.getenv("ARMED_COOLDOWN_SECONDS", "45")),
        mobilenet_model_path=os.getenv("MOBILENET_MODEL_PATH") or None,
        mobilenet_backend=os.getenv("MOBILENET_BACKEND", "auto"),
        mobilenet_input_size=int(os.getenv("MOBILENET_INPUT_SIZE", "224")),
        mobilenet_violence_class_index=int(os.getenv("MOBILENET_VIOLENCE_CLASS_INDEX", "1")),
        mobilenet_preprocess_mode=os.getenv("MOBILENET_PREPROCESS_MODE", "zero_one"),
        max_clip_frames_for_action=int(os.getenv("MAX_CLIP_FRAMES_FOR_ACTION", "16")),
        sequence_length=int(os.getenv("SEQUENCE_LENGTH", "16")),
    )
