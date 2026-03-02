from __future__ import annotations

import argparse
import sys

from edge_worker.service import MultiCameraEdgeService, load_cameras_from_json, load_config_from_env


def main() -> int:
    parser = argparse.ArgumentParser(description="DLW multi-camera edge inference worker")
    parser.add_argument(
        "--cameras",
        required=True,
        help="Path to JSON file containing camera stream configs",
    )
    args = parser.parse_args()

    cameras = load_cameras_from_json(args.cameras)
    if not cameras:
        print("No active cameras found in config")
        return 1

    cfg = load_config_from_env()
    service = MultiCameraEdgeService(cameras=cameras, cfg=cfg)
    service.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
