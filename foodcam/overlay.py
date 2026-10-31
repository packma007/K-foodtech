"""영상 위에 박스와 영양성분 패널 그리기. OpenCV는 한글을 못 그려서 글자는 Pillow로 그린다."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detectors import Detection
from .nutrition import FOUND, Lookup

FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]
COLORS = {"barcode": (0, 200, 255), "food": (80, 220, 80)}  # BGR


def _font(size: int):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


class Overlay:
    def __init__(self, font_size: int = 16):
        self.font = _font(font_size)
        self.title_font = _font(font_size + 2)
        self.line_h = font_size + 5

    def draw(self, frame: np.ndarray, items: list[tuple[Detection, Lookup]]) -> np.ndarray:
        for det, _ in items:
            x1, y1, x2, y2 = det.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[det.kind], 2)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img, "RGBA")
        h, w = frame.shape[:2]
        for det, lk in items:
            title, lines = self._text(det, lk)
            box_w = max(draw.textlength(s, font=self.font) for s in [title, *lines]) + 16
            box_h = self.line_h * (len(lines) + 1) + 10
            x1, y1, x2, _ = det.box
            # 박스 오른쪽에 두고, 화면 밖으로 나가면 왼쪽으로
            px = x2 + 6 if x2 + 6 + box_w <= w else max(0, x1 - 6 - box_w)
            py = min(max(0, y1), max(0, h - box_h))
            draw.rectangle([px, py, px + box_w, py + box_h], fill=(0, 0, 0, 170))
            b, g, r = COLORS[det.kind]
            draw.text((px + 8, py + 4), title, font=self.title_font, fill=(r, g, b))
            for i, s in enumerate(lines):
                draw.text((px + 8, py + 6 + self.line_h * (i + 1)), s, font=self.font, fill=(255, 255, 255))
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _text(det: Detection, lk: Lookup) -> tuple[str, list[str]]:
        label = det.query if det.kind == "food" else f"바코드 {det.query}"
        if lk.status != FOUND:
            return label, [lk.status + (f" · {lk.message}" if lk.message else "")]
        n = lk.nutrition
        title = n.name + (f" · {n.maker}" if n.maker else "")
        basis = f"{n.basis} 기준" if n.basis else ""
        if n.weight:
            basis += f" (총 {n.weight})"
        lines = [f"{k} {v}" for k, v in n.values]
        lines.append(f"[{n.source}] {basis}".strip())
        return title, lines
