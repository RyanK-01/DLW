from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


def load_keras_model_compat(model_path: Path):
    """Load Keras model with fallbacks for legacy .h5 compatibility."""
    errors: list[str] = []

    try:
        from tensorflow.keras.models import load_model as tf_load_model

        return tf_load_model(str(model_path), compile=False)
    except Exception as exc:
        errors.append(f"tensorflow.keras failed: {exc}")

    try:
        import tf_keras

        return tf_keras.models.load_model(str(model_path), compile=False)
    except Exception as exc:
        errors.append(f"tf_keras failed: {exc}")

    try:
        from keras.models import load_model as keras_load_model

        return keras_load_model(str(model_path), compile=False)
    except Exception as exc:
        errors.append(f"keras failed: {exc}")

    raise RuntimeError(
        "Unable to load model. This is usually a Keras version mismatch for legacy .h5 files. "
        "Re-save model as .keras using the current environment, or install tf-keras. "
        f"Details: {' | '.join(errors)}"
    )


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def ensure_yolo_weights(weights_path: str) -> None:
    # Ultralytics auto-downloads known model aliases like yolov8n.pt on first load.
    from ultralytics import YOLO

    model = YOLO(weights_path)
    _ = model.model  # force load
    print(f"YOLO ready: {weights_path}")


def ensure_mobilenet_model(model_path: Path, model_url: str | None) -> None:
    if model_path.exists():
        print(f"MobileNetV2 model exists: {model_path}")
        return

    if not model_url:
        raise RuntimeError(
            f"Model not found at {model_path}. Provide --mobilenet-model-url to download."
        )

    print(f"Downloading MobileNetV2 model to: {model_path}")
    download_file(model_url, model_path)
    print("MobileNetV2 model downloaded")


def validate_mobilenet_model(model_path: Path, backend: str) -> None:
    suffix = model_path.suffix.lower()
    chosen = backend
    if chosen == "auto":
        chosen = "onnx" if suffix == ".onnx" else "keras"

    if chosen == "onnx":
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        print(f"ONNX input name: {input_meta.name}, shape: {input_meta.shape}")
        print(f"ONNX output name: {output_meta.name}, shape: {output_meta.shape}")
        print("MobileNetV2 ONNX model initialized successfully")
        return

    model = load_keras_model_compat(model_path)
    print(f"Keras input shape: {model.input_shape}")
    print(f"Keras output shape: {model.output_shape}")
    print("MobileNetV2 Keras model initialized successfully")
    print("Expected violence score index: 1 (for classes [NonViolence, Violence])")


def main() -> int:
    backend_dir = Path(__file__).resolve().parents[1]
    load_dotenv(backend_dir / ".env")

    parser = argparse.ArgumentParser(description="Prepare YOLOv8 and MobileNetV2 model files")
    parser.add_argument("--yolo-weights", default="models/yolov8n.pt")
    parser.add_argument("--weapon-weights", default=os.getenv("WEAPON_WEIGHTS"))
    parser.add_argument("--mobilenet-model", default=os.getenv("MOBILENET_MODEL_PATH"))
    parser.add_argument("--mobilenet-model-url", default=None)
    parser.add_argument(
        "--mobilenet-backend",
        default=os.getenv("MOBILENET_BACKEND", "auto"),
        choices=["auto", "keras", "onnx"],
    )
    args = parser.parse_args()

    ensure_yolo_weights(args.yolo_weights)
    if args.weapon_weights:
        ensure_yolo_weights(args.weapon_weights)

    if args.mobilenet_model:
        model_path = Path(args.mobilenet_model)
        ensure_mobilenet_model(model_path, args.mobilenet_model_url)
        validate_mobilenet_model(model_path, args.mobilenet_backend)
    else:
        print(
            "MobileNetV2 skipped. Set MOBILENET_MODEL_PATH in Backend/.env "
            "or pass --mobilenet-model <path>."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
