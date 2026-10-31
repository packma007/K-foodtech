"""양 추정: 500원 동전으로 축척을 잡고, 그릇 속 음식 표면 넓이로 부피를 계산한다.

카메라는 상판을 위에서 수직으로 내려다본다고 가정한다.

1) 축척     동전 지름(px) / 26.5mm = px_per_mm (상판 높이 기준, 최근 값들의 중앙값)
2) 높이보정 그릇 윗면은 카메라에 더 가까워 크게 보임
            → 윗면 축척 = px_per_mm × Z / (Z − 그릇깊이)   (Z = 카메라 높이)
3) 그릇     테두리 원 지름(px) → mm
4) 음식     빈 그릇 사진(없으면 테두리 색)과 달라진 부분·결이 있는 부분 = 음식 표면 → 같은 넓이의 원 반지름 r
5) 부피     그릇을 원뿔대로 보고  h = (r − b)/(R − b)·H,  V = π·h·(b² + b·r + r²)/3
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .meal import BowlType

MIN_LAYER_MM = 5.0  # 음식이 바닥만 덮을 때 두께 가정


# ── 1) 동전 축척 ───────────────────────────────────────────────────
COIN_MIN_EDGE = 3.0  # 둘레의 밝기 변화(반지름 방향) ÷ 바탕 결의 세기. 동전 테두리는 10 이상, 천의 결·뚜껑은 1~6
COIN_MAX_SAT = 100  # 은색: 채도가 낮다 (실내 조명에선 약간 노랗게 찍혀 75 안팎까지 나온다)
COIN_MIN_VAL = 60  # 너무 어둡지 않다
COIN_MIN_CONTRAST = 18  # 동전 안쪽과 바로 바깥 바닥의 색 차이 (Lab 거리). 밝은 나무 식탁은 밝기는 비슷해도 색이 다르다
COIN_BOWL_RATIO = (0.06, 0.45)  # 동전 반지름 / 그릇 반지름 (500원 26.5mm ÷ 그릇 100~250mm, 높이 차 감안)


def find_coin(frame: np.ndarray, bowls: list[tuple[int, int, int]],
              prev: tuple[int, int, int] | None = None) -> tuple[int, int, int] | None:
    """사진에서 500원 동전 원 (x, y, r)을 찾는다. 확실한 후보가 없으면 None.
    prev(직전 장면의 동전)가 있으면 그 근처·비슷한 크기 후보를 우선한다 (동전은 움직이지 않는다)."""
    h, w = frame.shape[:2]
    m = min(h, w)
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 1.5)
    # 원 후보는 줄인 사진에서 찾는다. 원래 크기에선 천의 결이 원 후보를 수천 개 만들어 진짜 동전을 밀어낸다
    # (특히 가까이 찍었을 때). 검증은 아래에서 원래 크기 사진으로 한다.
    k = min(1.0, 480 / m)
    small = cv2.medianBlur(cv2.resize(gray, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA), 5)
    sm = min(small.shape[:2])
    circles = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, dp=1.0, minDist=max(6, sm // 40), param1=100, param2=24,
                               minRadius=max(4, sm // 80), maxRadius=sm // 4)
    if circles is None:
        return None
    circles = circles / k
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    best, best_score = None, 0.0
    # 바닥 천의 결 때문에 후보가 수백 개 나오기도 해서 전부 보되, 싼 검사(둘레 경계)부터 한다.
    # 결이 있는 천에서는 어디에 원을 그려도 '경계선'이 걸리므로, 경계가 있는지가 아니라
    # 둘레의 반지름 방향 밝기 변화가 바로 바깥 바탕보다 얼마나 센지를 본다.
    for x, y, r in np.round(circles[0]).astype(int):
        # 그릇 안의 둥근 것(음식 재료 등)이나, 그릇에 비해 너무 크거나 작은 원은 동전이 아니다
        if any(math.hypot(x - bx, y - by) < br * 0.9 for bx, by, br in bowls):
            continue
        if bowls and not COIN_BOWL_RATIO[0] <= r / max(br for _, _, br in bowls) <= COIN_BOWL_RATIO[1]:
            continue
        # 1) 둘레가 바탕보다 뚜렷한 경계인가
        px, py = (x + r * np.cos(ang)).astype(int), (y + r * np.sin(ang)).astype(int)
        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if inside.mean() < 0.9:
            continue
        a = ang[inside]
        radial = np.abs(gx[py[inside], px[inside]] * np.cos(a) + gy[py[inside], px[inside]] * np.sin(a))
        bx = (x + 1.4 * r * np.cos(ang)).astype(int).clip(0, w - 1)
        by = (y + 1.4 * r * np.sin(ang)).astype(int).clip(0, h - 1)
        edge = float(np.median(radial)) / (float(np.median(np.hypot(gx[by, bx], gy[by, bx]))) + 5)
        if edge < COIN_MIN_EDGE:
            continue
        # 2) 은색인가  3) 바로 바깥 바닥과 색이 다른가
        R = int(1.6 * r) + 1  # 후보 주변만 잘라서 계산 (사진 전체로 하면 장당 몇 초)
        x0, y0 = max(0, x - R), max(0, y - R)
        win = hsv[y0:y + R + 1, x0:x + R + 1]
        yy, xx = np.ogrid[y0:y0 + win.shape[0], x0:x0 + win.shape[1]]
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        disk = d2 <= (0.8 * r) ** 2
        ring = (d2 >= (1.2 * r) ** 2) & (d2 <= (1.6 * r) ** 2)
        if not disk.any() or not ring.any():
            continue
        sat, val = float(win[..., 1][disk].mean()), float(win[..., 2][disk].mean())
        if sat > COIN_MAX_SAT or val < COIN_MIN_VAL:
            continue
        lwin = lab[y0:y0 + win.shape[0], x0:x0 + win.shape[1]]
        contrast = float(np.linalg.norm(lwin[disk].mean(axis=0) - lwin[ring].mean(axis=0)))
        if contrast < COIN_MIN_CONTRAST:
            continue
        score = edge * min(contrast / 60, 1.0)
        if prev and math.hypot(x - prev[0], y - prev[1]) < 2 * prev[2] and abs(r / prev[2] - 1) < 0.25:
            score *= 2
        if score > best_score:
            best, best_score = (int(x), int(y), int(r)), score
    return best


class CoinScale:
    """동전으로 px/mm 축척을 잡는다. 휴대폰을 들고 찍으면 높이가 계속 바뀌므로 최근 몇 장만 쓴다."""

    HOLD_S = 3.0  # 동전이 잠깐 안 보여도 이 시간 동안은 마지막 축척을 쓴다
    JUMP = 0.15  # 크기가 이만큼 넘게 바뀌면 카메라 높이가 바뀐 것으로 보고 새로 잡는다

    def __init__(self, coin_diameter_mm: float = 26.5, history: int = 5):
        self.coin_mm = coin_diameter_mm
        self._recent: deque[float] = deque(maxlen=history)
        self._seen_at = -1e9
        self.last_circle: tuple[int, int, int] | None = None

    @property
    def px_per_mm(self) -> float | None:
        return float(np.median(self._recent)) if self._recent else None

    def update(self, frame: np.ndarray, bowls: list[tuple[int, int, int]], now: float | None = None) -> None:
        now = time.time() if now is None else now
        c = find_coin(frame, bowls, self.last_circle)
        if c:
            v = 2 * c[2] / self.coin_mm
            cur = self.px_per_mm
            if cur and abs(v / cur - 1) > self.JUMP:
                self._recent.clear()
            self._recent.append(v)
            self.last_circle, self._seen_at = c, now
        elif now - self._seen_at > self.HOLD_S:
            self._recent.clear()
            self.last_circle = None


# ── 3) 그릇 찾기 ───────────────────────────────────────────────────
class BowlFinder:
    """mode="yolo": COCO 'bowl' 클래스,  mode="hough": 큰 원 검출 (모델 없이, 흰 그릇에 잘 맞음)."""

    def __init__(self, mode: str = "yolo", model_path: str = "yolo11n.pt", conf: float = 0.2):
        self.mode = mode
        if mode == "yolo":
            from ultralytics import YOLO

            self.model = YOLO(model_path)
            self.bowl_ids = [i for i, n in self.model.names.items() if n == "bowl"]
            self.conf = conf

    def __call__(self, frame: np.ndarray) -> list[tuple[int, int, int]]:
        if self.mode == "yolo":
            r = self.model(frame, conf=self.conf, classes=self.bowl_ids, verbose=False)[0]
            out = []
            for x1, y1, x2, y2 in r.boxes.xyxy.tolist():
                out.append((int((x1 + x2) / 2), int((y1 + y2) / 2), int((x2 - x1 + y2 - y1) / 4)))
            return out
        h, w = frame.shape[:2]
        gray = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 7)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=min(h, w) // 6, param1=100,
                                   param2=60, minRadius=min(h, w) // 12, maxRadius=min(h, w) // 2)
        if circles is None:
            return []
        out: list[tuple[int, int, int]] = []
        for x, y, r in sorted(np.round(circles[0]).astype(int).tolist(), key=lambda c: -c[2]):
            if all(math.hypot(x - ox, y - oy) > max(r, orr) for ox, oy, orr in out):  # 겹치는 원 제거
                out.append((x, y, r))
        return out


# ── 4) 음식 표면 ───────────────────────────────────────────────────
# 그릇마다 그릇 원을 NORM×NORM 크기로 잘라 맞춘 뒤 비교한다 (그릇이 조금 움직여도 같은 좌표).
NORM = 160
_yy, _xx = np.mgrid[0:NORM, 0:NORM]
_D = np.hypot(_xx - (NORM - 1) / 2, _yy - (NORM - 1) / 2) / (NORM / 2)  # 그릇 중심에서 거리 / 반지름
INNER = _D < 0.85  # 테두리 색과 비교할 때 볼 범위
INNER_REF = _D < 0.95  # 빈 그릇 사진과 비교할 때 볼 범위 (가득 찬 경우까지)
RIM = (_D > 0.86) & (_D <= 0.95)  # 그릇 안쪽 벽 위쪽. 바깥 그림자가 섞이지 않게 테두리보다 약간 안쪽


def bowl_crop(frame: np.ndarray, cx: int, cy: int, R: int) -> np.ndarray | None:
    """그릇 원을 감싸는 정사각형을 NORM×NORM으로 (화면 밖은 가장자리 복사). 원이 화면 밖이면 None."""
    pad = R + 2
    big = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = big[max(0, cy + pad - R):max(0, cy + pad + R + 1), max(0, cx + pad - R):max(0, cx + pad + R + 1)]
    if crop.shape[0] < 2 * R + 1 or crop.shape[1] < 2 * R + 1:
        return None
    return cv2.resize(crop, (NORM, NORM), interpolation=cv2.INTER_AREA)


def _texture(crop: np.ndarray) -> np.ndarray:
    """가장자리 밀도 (밥알 같은 결이 있으면 높음)."""
    edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 30, 90)
    return cv2.blur(edges.astype(np.float32) / 255, (9, 9))


def _lab(crop: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)


def _color_dist(lab: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d = lab - ref
    return np.sqrt((0.5 * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)  # 그림자 영향 줄이려 밝기 가중 ↓


def looks_empty(crop: np.ndarray) -> bool:
    """그릇 안쪽이 고르게 한 색이고 결이 거의 없으면 빈 그릇."""
    lab = _lab(crop)
    inner = lab[INNER | RIM]
    return (float(np.percentile(_color_dist(inner, np.median(inner, axis=0)), 95)) < 15
            and float(_texture(crop)[INNER].mean()) < 0.03)


def food_mask(crop: np.ndarray, empty_ref: np.ndarray | None, color_thr: float = 14.0) -> np.ndarray:
    """NORM×NORM 마스크. empty_ref(빈 그릇 사진)가 있으면 그것과 비교, 없으면 테두리 색과 비교."""
    lab = _lab(crop)
    tex = _texture(crop)
    if empty_ref is not None:
        by_color = _color_dist(lab, _lab(empty_ref)) > color_thr
        by_texture = tex - _texture(empty_ref) > 0.06
        area = INNER_REF
    else:
        by_color = _color_dist(lab, np.median(lab[RIM], axis=0)) > color_thr
        by_texture = tex > 0.06
        area = INNER
    # 결 밀도는 9×9로 흐려서 구했기 때문에 경계 밖으로 4px 번진다. 같은 만큼 깎아 되돌린다
    # (선 하나짜리 경계는 사라지고, 밥알처럼 결이 넓게 퍼진 곳만 남는다)
    by_texture = cv2.erode(by_texture.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)

    m = ((by_color | by_texture) & area).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    # 그릇 중심 쪽의 큰 덩어리만 (음식은 바닥 가운데부터 쌓인다) → 볼록 껍질로 채움
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    if n > 1:
        c = np.array([(NORM - 1) / 2] * 2)
        idx = min(range(1, n), key=lambda i: np.hypot(*(cents[i] - c)) - math.sqrt(stats[i, cv2.CC_STAT_AREA]))
        if stats[idx, cv2.CC_STAT_AREA] >= 30:
            cnts, _ = cv2.findContours((labels == idx).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, [cv2.convexHull(np.vstack(cnts))], -1, 255, -1)
    return out


# ── 5) 부피 ────────────────────────────────────────────────────────
def fill_volume_ml(rim_d_mm: float, food_d_mm: float, shape: BowlType) -> float:
    R = rim_d_mm / 2
    b = R * shape.bottom_ratio
    H = rim_d_mm * shape.depth_ratio
    r = min(max(food_d_mm / 2, 0.0), R)
    if r <= b:  # 바닥만 일부 덮음
        return math.pi * r * r * MIN_LAYER_MM / 1000
    hgt = (r - b) / (R - b) * H
    return math.pi * hgt * (b * b + b * r + r * r) / 3 / 1000


def surface_height_mm(rim_d_mm: float, food_d_mm: float, shape: BowlType) -> float:
    R, H = rim_d_mm / 2, rim_d_mm * shape.depth_ratio
    b = R * shape.bottom_ratio
    r = min(food_d_mm / 2, R)
    return 0.0 if r <= b else (r - b) / (R - b) * H


# ── 그릇 추적 + 평활 ───────────────────────────────────────────────
@dataclass
class TrackedBowl:
    id: int
    cx: int
    cy: int
    last_seen: float
    hits: int = 0  # 연속으로 보인 횟수. CONFIRM_HITS 이상이어야 그릇으로 인정
    radii: deque = field(default_factory=lambda: deque(maxlen=30), repr=False)
    empty_ref: np.ndarray | None = field(default=None, repr=False)  # 빈 그릇일 때 사진
    ref_r: int | None = None  # 빈 그릇 사진을 찍을 때의 반지름. 이후 이 값으로 고정해 사진과 맞춘다
    grams: float | None = None  # 지수이동평균
    rim_mm: float | None = None
    hull: np.ndarray | None = field(default=None, repr=False)  # 화면 좌표 음식 윤곽
    food_source: str | None = None  # 영양성분 출처 (식약처 / AI 추정 / meal.toml)

    @property
    def r(self) -> int:
        # 그릇 크기는 변하지 않는다. 빈 그릇 사진이 있으면 그때 반지름, 없으면 최근 값들의 중앙값
        # (음식이 가득 차면 음식 원을 그릇으로 잘못 잡기도 함)
        return self.ref_r or int(np.median(self.radii))


CONFIRM_HITS = 5


class BowlTracker:
    def __init__(self, lost_after: float = 1.0, alpha: float = 0.25, confirm_hits: int = CONFIRM_HITS):
        self.bowls: list[TrackedBowl] = []
        self._next = 0
        self.lost_after = lost_after
        self.alpha = alpha
        self.confirm_hits = confirm_hits

    def update(self, circles: list[tuple[int, int, int]], now: float) -> list[TrackedBowl]:
        used = set()
        for x, y, r in circles:
            match = min((b for b in self.bowls if b.id not in used),
                        key=lambda b: math.hypot(b.cx - x, b.cy - y), default=None)
            # 손으로 들고 찍으면 장면 사이에 그릇이 꽤 움직인다 → 반지름만큼까지는 같은 그릇으로 본다
            if match and math.hypot(match.cx - x, match.cy - y) < match.r:
                match.cx, match.cy = int(0.7 * match.cx + 0.3 * x), int(0.7 * match.cy + 0.3 * y)
                match.radii.append(r)
                match.last_seen = now
                match.hits += 1
                used.add(match.id)
            elif not any(math.hypot(b.cx - x, b.cy - y) < b.r + r * 0.5 for b in self.bowls):
                # 기존 그릇과 크게 겹치는 원(그릇 두 개를 감싸는 원 등)은 새 그릇으로 보지 않는다
                b = TrackedBowl(self._next, x, y, now)
                b.radii.append(r)
                self._next += 1
                self.bowls.append(b)
                used.add(b.id)
        self.bowls = [b for b in self.bowls if now - b.last_seen < self.lost_after]
        return sorted((b for b in self.bowls if b.hits >= self.confirm_hits), key=lambda b: b.cx)  # 왼쪽부터 = meal.toml의 bowls 순서

    def reset_references(self) -> None:
        for b in self.bowls:
            b.empty_ref, b.ref_r = None, None

    def smooth(self, b: TrackedBowl, grams: float) -> float:
        b.grams = grams if b.grams is None else (1 - self.alpha) * b.grams + self.alpha * grams
        return b.grams


def measure(frame: np.ndarray, b: TrackedBowl, px_per_mm: float, shape: BowlType,
            camera_height_mm: float, density: float) -> float:
    """그릇 하나의 음식 무게(g) 추정. b.rim_mm, b.hull도 채운다."""
    def scale_at(height_mm: float) -> float:  # 상판보다 height_mm 높은 면의 px/mm
        return px_per_mm * camera_height_mm / max(camera_height_mm - height_mm, 1.0)

    R = b.r
    rim_mm = 2 * R / px_per_mm
    for _ in range(3):  # 그릇 지름 ↔ 깊이 ↔ 높이보정을 몇 번 반복해 맞춘다
        rim_mm = 2 * R / scale_at(rim_mm * shape.depth_ratio)
    b.rim_mm = rim_mm
    rim_scale = scale_at(rim_mm * shape.depth_ratio)

    crop = bowl_crop(frame, b.cx, b.cy, R)
    if crop is None:
        return b.grams or 0.0  # 그릇이 화면 밖: 이전 값 유지
    if b.empty_ref is None and looks_empty(crop):
        b.empty_ref, b.ref_r = crop, R
    m = food_mask(crop, b.empty_ref)

    area_frac = cv2.countNonZero(m) / (math.pi * (NORM / 2) ** 2)
    food_r_px = R * math.sqrt(area_frac)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    b.hull = None
    if cnts:  # NORM 좌표 → 화면 좌표
        k = (2 * R + 1) / NORM
        b.hull = (cnts[0].astype(np.float32) * k + np.array([b.cx - R, b.cy - R])).astype(np.int32)

    # 음식 표면은 그릇 윗면보다 낮아 덜 커 보인다. 표면 높이에 맞춘 축척으로 다시 계산
    food_d_mm = 2 * food_r_px / rim_scale
    for _ in range(2):
        food_d_mm = 2 * food_r_px / scale_at(surface_height_mm(rim_mm, food_d_mm, shape))
    return fill_volume_ml(rim_mm, food_d_mm, shape) * density
