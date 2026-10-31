"""배식 안내: 그릇에 담긴 양을 실시간으로 추정해 목표 영양성분에 맞게 '덜어라/더 담아라'를 말한다.

    python -m foodcam.serving --meal meal.toml
    python -m foodcam.serving --meal meal.toml --bowls hough   # YOLO 대신 원 검출로 그릇 찾기
    python -m foodcam.serving --meal meal.toml --video test.mp4 --mute

준비: 카메라를 상판 위에서 수직으로, 500원 동전을 그릇 옆 상판에 둔다.
      빈 그릇을 먼저 보여 주면 그 모습을 기준으로 음식을 더 정확히 찾는다.
키:   r = 빈 그릇 기준 다시 찍기,  q/ESC = 종료
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .guide import Advice, Portion, advise, status_lines
from .main import build_resolver
from .meal import BowlPlan, Meal, load_meal
from .nutrition import FOUND, NutritionResolver
from .overlay import _font
from .portion import BowlFinder, BowlTracker, CoinScale, TrackedBowl, bowl_crop, measure
from .voice import AdviceAnnouncer, Speaker

STATE_COLORS = {"ok": (120, 230, 120), "over": (255, 90, 90), "under": (255, 210, 80)}  # RGB
SMALL_BOWL_MM = 135  # 자동 인식 모드에서 그릇 지름이 이보다 작으면 밥그릇, 크면 국그릇 모양으로 계산
MIN_FOOD_G = 15  # 이보다 적으면 빈 그릇으로 본다
DEFAULT_DENSITY = 0.9


@dataclass
class Analysis:
    rows: list  # [(TrackedBowl, BowlPlan | None, grams | None, per_gram | None)]
    portions: list[Portion]
    advice: Advice | None
    problem: str | None  # 안내를 못 하는 이유 (동전 못 찾음 등)
    scale: float | None  # px/mm


class ServingApp:
    def __init__(self, meal: Meal, resolver: NutritionResolver, finder: BowlFinder, speaker,
                 tracker: BowlTracker | None = None, recognizer=None):
        """speaker: say(text)가 있는 객체. PC는 Speaker, 모바일 서버는 말할 문장을 모아 앱에 돌려준다.
        recognizer(FoodRecognizer)가 있으면 그릇마다 음식을 인식하고, 없으면 meal.toml의 [[bowls]]를
        왼쪽 그릇부터 순서대로 붙인다."""
        self.meal = meal
        self.recognizer = recognizer
        self.key = f"{id(self):x}"
        self.resolver = resolver
        self.finder = finder
        self.coin = CoinScale(meal.coin_diameter_mm)
        self.tracker = tracker or BowlTracker()
        self.announcer = AdviceAnnouncer(speaker)
        self.font = _font(18)
        self.small = _font(15)
        self._frame_shape: tuple[int, int] | None = None
        for plan in meal.bowls:  # 영양정보 미리 조회 시작
            if not plan.per100g:
                resolver.get("food", plan.food)

    def per_gram(self, i: int) -> dict[str, float] | None:
        plan = self.meal.bowls[i]
        if plan.per100g:
            return {k: v / 100 for k, v in plan.per100g.items()}
        lk = self.resolver.get("food", plan.food)
        return lk.nutrition.per_gram() if lk.status == FOUND else None

    def process(self, frame: np.ndarray, now: float) -> tuple[np.ndarray, Advice | None]:
        a = self.analyze(frame, now)
        return self._draw(frame, a.rows, a.portions, a.advice, a.problem, a.scale), a.advice

    def analyze(self, frame: np.ndarray, now: float) -> Analysis:
        # 사진 크기가 바뀌면(휴대폰을 돌림 등) 그릇 좌표와 동전 축척이 더는 맞지 않으므로 처음부터 다시 잡는다
        shape = frame.shape[:2]
        if self._frame_shape is not None and shape != self._frame_shape:
            t = self.tracker
            self.tracker = BowlTracker(t.lost_after, t.alpha, t.confirm_hits)
            self.coin = CoinScale(self.meal.coin_diameter_mm)
        self._frame_shape = shape

        circles = self.finder(frame)
        self.coin.update(frame, circles)
        bowls = self.tracker.update(circles, now)
        scale = self.coin.px_per_mm

        rows = []  # (bowl, plan, grams, per_g)
        advice, problem = None, None
        if self.recognizer:
            for b in bowls:
                rows.append(self._recognized_row(frame, b, scale) if scale else (b, None, None, None))
            if not scale:
                problem = "500원 동전을 찾는 중…"
            elif not bowls:
                problem = "그릇을 찾는 중…"
            elif any(g is not None and g >= MIN_FOOD_G and pg is None for _, _, g, pg in rows):
                problem = "음식 인식 중…"
        else:
            for i, b in enumerate(bowls):
                plan = self.meal.bowls[i] if i < len(self.meal.bowls) else None
                grams = None
                if plan and scale:
                    g = measure(frame, b, scale, self.meal.bowl_types[plan.type],
                                self.meal.camera_height_mm, plan.density)
                    grams = self.tracker.smooth(b, g)
                    b.food_source = "meal.toml 순서"
                rows.append((b, plan, grams, self.per_gram(i) if plan else None))
            if not scale:
                problem = "500원 동전을 찾는 중…"
            elif len(bowls) < len(self.meal.bowls):
                problem = f"그릇 {len(self.meal.bowls)}개 중 {len(bowls)}개만 보임"
            elif any(r[3] is None for r in rows if r[1]):
                problem = "영양정보 조회 중…"
        portions = [Portion(p.food, g, pg) for _, p, g, pg in rows if p and g is not None and pg and g >= MIN_FOOD_G]
        if not problem:
            advice = advise(portions, self.meal.targets, self.meal.tolerance)
            self.announcer.update(advice, now)
        return Analysis(rows, portions, advice, problem, scale)

    def _recognized_row(self, frame: np.ndarray, b: TrackedBowl, scale: float):
        """자동 인식 모드에서 그릇 하나: 모양 고르기 → 양 재기 → 음식 인식 → 영양성분."""
        types = self.meal.bowl_types
        rim_guess = 2 * b.r / scale
        type_name = "밥그릇" if rim_guess < SMALL_BOWL_MM else "국그릇"
        if type_name not in types:
            type_name = next(iter(types))

        key = (self.key, b.id)
        st = self.recognizer.state(key)
        res = st.result if st.result and st.result.is_food else None
        density = res.density if res else DEFAULT_DENSITY
        grams = self.tracker.smooth(b, measure(frame, b, scale, types[type_name], self.meal.camera_height_mm, density))

        small = bowl_crop(frame, b.cx, b.cy, b.r)
        if small is not None:
            x0, y0 = max(0, b.cx - int(b.r * 1.05)), max(0, b.cy - int(b.r * 1.05))
            photo = frame[y0:b.cy + int(b.r * 1.05), x0:b.cx + int(b.r * 1.05)]
            st = self.recognizer.lookup(key, small, photo, grams >= MIN_FOOD_G)
            res = st.result if st.result and st.result.is_food else None

        if grams < MIN_FOOD_G:
            b.food_source = None
            return b, BowlPlan("빈 그릇", type_name, density), grams, None
        if res is None:
            b.food_source = "인식 실패: " + st.error if st.error and not st.pending else None
            return b, BowlPlan("인식 중…", type_name, density), grams, None

        lk = self.resolver.get("food", res.db_name)
        if lk.status == FOUND and lk.nutrition.amounts:
            pg, b.food_source = lk.nutrition.per_gram(), "식약처"
        else:
            pg, b.food_source = {k: v / 100 for k, v in res.per100g.items()}, "AI 추정"
        return b, BowlPlan(res.name, type_name, density), grams, pg

    def _draw(self, frame, rows, portions, advice, problem, scale) -> np.ndarray:
        if self.coin.last_circle:
            x, y, r = self.coin.last_circle
            cv2.circle(frame, (x, y), r, (0, 215, 255), 2)
        for b, plan, grams, _ in rows:
            cv2.circle(frame, (b.cx, b.cy), b.r, (255, 255, 255) if plan else (128, 128, 128), 2)
            if b.hull is not None and plan:
                cv2.drawContours(frame, [b.hull], -1, (80, 220, 80), 2)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img, "RGBA")
        for b, plan, grams, pg in rows:
            if not plan:
                text = "설정에 없는 그릇"
            elif grams is None:
                text = plan.food
            else:
                kcal = f" · {grams * pg['열량']:.0f}kcal" if pg and "열량" in pg else ""
                text = f"{plan.food} {grams:.0f}g{kcal}"
            if b.rim_mm:
                text += f"\n그릇 ⌀{b.rim_mm:.0f}mm"
            d.multiline_text((b.cx - b.r, b.cy + b.r + 6), text, font=self.small, fill=(255, 255, 255),
                             stroke_width=2, stroke_fill=(0, 0, 0))

        # 오른쪽 위 패널: 합계 vs 목표 + 안내
        lines = [(s, STATE_COLORS[st]) for s, st in status_lines(portions, self.meal.targets, self.meal.tolerance)]
        if problem:
            lines.append((problem, (200, 200, 200)))
        elif advice:
            lines.append(("▶ " + advice.speech(), (255, 255, 255)))
        if scale:
            lines.append((f"축척 {scale:.2f}px/mm", (160, 160, 160)))
        w = max(d.textlength(s, font=self.font) for s, _ in lines) + 20
        x0 = img.width - w - 10
        d.rectangle([x0, 10, x0 + w, 20 + 26 * len(lines)], fill=(0, 0, 0, 170))
        for i, (s, c) in enumerate(lines):
            d.text((x0 + 10, 15 + 26 * i), s, font=self.font, fill=c)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meal", default="meal.toml")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="카메라 대신 영상 파일")
    ap.add_argument("--bowls", choices=["yolo", "hough"], default="yolo")
    ap.add_argument("--mute", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    from .recognize import FoodRecognizer

    resolver = build_resolver()  # .env도 여기서 읽는다 → 그 뒤에 API 키 확인
    recognizer = FoodRecognizer() if FoodRecognizer.available() else None
    app = ServingApp(load_meal(args.meal), resolver, BowlFinder(args.bowls), Speaker(not args.mute),
                     recognizer=recognizer)
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("카메라/영상을 열 수 없음")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out, _ = app.process(frame, time.time())
        cv2.imshow("foodcam - serving", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):  # 그릇을 비우거나 바꾼 뒤 빈 그릇 사진 다시 찍기
            app.tracker.reset_references()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
