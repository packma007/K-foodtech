"""카메라 영상에서 음식/제품을 찾아 영양성분을 띄운다.

    python -m foodcam.main                  # 웹캠 0번
    python -m foodcam.main --camera 1
    python -m foodcam.main --image test.jpg # 사진 한 장으로 확인
    python -m foodcam.main --no-yolo        # 바코드만

종료: q 또는 ESC
"""
from __future__ import annotations

import argparse
import logging
import os
import time

import cv2
from dotenv import load_dotenv

from .detectors import BarcodeScanner, Detection, FoodDetector
from .nutrition import NutritionResolver
from .overlay import Overlay

BARCODE_HOLD = 1.5  # 초. 바코드는 흔들리면 금방 안 읽히므로 잠깐 화면에 유지


def build_resolver() -> NutritionResolver:
    load_dotenv()
    return NutritionResolver(
        mfds_key=os.getenv("MFDS_SERVICE_KEY", ""),
        foodsafety_key=os.getenv("FOODSAFETY_KEY", "sample"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--image", help="카메라 대신 이미지 파일")
    ap.add_argument("--model", default="yolo11n.pt", help="YOLO 가중치 (AI Hub로 학습한 한식 모델 등)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--no-yolo", action="store_true", help="음식 탐지 끄고 바코드만")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    resolver = build_resolver()
    scanner = BarcodeScanner()
    detector = None if args.no_yolo else FoodDetector(args.model, args.conf)
    overlay = Overlay()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"이미지를 열 수 없음: {args.image}")
        dets = scanner(frame) + (detector(frame) if detector else [])
        for d in dets:  # 사진 모드는 조회가 끝날 때까지 기다린다
            resolver.get(d.kind, d.query)
        items = [(d, resolver.resolve(d.kind, d.query)) for d in dets]
        cv2.imshow("foodcam", overlay.draw(frame, items))
        cv2.waitKey(0)
        return

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not cap.isOpened():
        raise SystemExit(f"카메라 {args.camera}번을 열 수 없음")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    recent_barcodes: dict[str, tuple[Detection, float]] = {}
    last = time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()

        for d in scanner(frame):
            recent_barcodes[d.query] = (d, now)
        recent_barcodes = {k: v for k, v in recent_barcodes.items() if now - v[1] < BARCODE_HOLD}
        dets = [d for d, _ in recent_barcodes.values()]
        if detector:
            dets += detector(frame)

        items = [(d, resolver.get(d.kind, d.query)) for d in dets]
        out = overlay.draw(frame, items)

        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        cv2.putText(out, f"{fps:.1f} fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("foodcam", out)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
