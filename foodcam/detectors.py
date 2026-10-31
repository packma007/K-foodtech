"""프레임에서 음식/제품 찾기.

- BarcodeScanner: 포장 제품의 바코드(EAN-13 등)를 읽는다 → 실제 제품을 정확히 특정.
- FoodDetector:   YOLO로 음식 객체를 찾는다 → 음식 이름(한글)만 알 수 있음.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# COCO 사전학습 모델의 음식 클래스 → 식약처 DB 검색용 한글 이름
COCO_FOOD_KO = {
    "banana": "바나나",
    "apple": "사과",
    "sandwich": "샌드위치",
    "orange": "오렌지",
    "broccoli": "브로콜리",
    "carrot": "당근",
    "hot dog": "핫도그",
    "pizza": "피자",
    "donut": "도넛",
    "cake": "케이크",
}


@dataclass
class Detection:
    kind: str  # "barcode" | "food"
    query: str  # 바코드 숫자 또는 음식 이름
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    score: float = 1.0


class BarcodeScanner:
    def __init__(self):
        self._det = cv2.barcode.BarcodeDetector()

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        ok, infos, _types, points = self._det.detectAndDecodeWithType(frame)
        if not ok or points is None:
            return []
        out = []
        for code, pts in zip(infos, points):
            if not code:  # 위치는 찾았지만 해독 실패
                continue
            xs, ys = pts[:, 0], pts[:, 1]
            out.append(Detection("barcode", code, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))
        return out


class FoodDetector:
    def __init__(self, model_path: str = "yolo11n.pt", conf: float = 0.4):
        from ultralytics import YOLO  # 무거워서 필요할 때만 불러옴

        self.model = YOLO(model_path)
        self.conf = conf
        self.names: dict[int, str] = self.model.names
        # COCO 모델이면 음식 클래스만 남기고, 직접 학습한 음식 모델이면 클래스 이름을 그대로 쓴다.
        self.is_coco = "person" in self.names.values()

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        r = self.model(frame, conf=self.conf, verbose=False)[0]
        out = []
        for xyxy, cls, score in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            name = self.names[int(cls)]
            if self.is_coco:
                name = COCO_FOOD_KO.get(name)
                if name is None:
                    continue
            x1, y1, x2, y2 = map(int, xyxy)
            out.append(Detection("food", name, (x1, y1, x2, y2), score))
        return out
