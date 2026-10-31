"""모바일 앱용 서버. 앱이 사진(JPEG)을 보내면 배식 분석 결과를 JSON으로 돌려준다.

    python -m foodcam.server --meal meal.toml              # 0.0.0.0:8000
    python -m foodcam.server --meal meal.toml --bowls hough

클라우드(Render)에서는 환경 변수로 설정한다: PORT(Render가 넣어 줌), BOWLS_MODE=hough, MEAL_PATH.
SAVE_FRAMES=<폴더> 를 주면 받은 사진을 2초에 한 장씩 저장한다 (실제 사진으로 동전·그릇 찾기를 맞출 때).
torch를 설치하지 않으므로(requirements-server.txt) 그릇은 hough로만 찾는다.

앱(휴대폰)마다 세션을 따로 둔다 (그릇 추적·빈 그릇 사진·말할 타이밍이 기기별로 달라서).
말할 문장은 서버가 정하고, 실제 소리는 휴대폰이 낸다.
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps

from .guide import status_lines
from .main import build_resolver
from .meal import Meal, load_meal
from .portion import BowlFinder, BowlTracker
from .recognize import FoodRecognizer
from .serving import Analysis, ServingApp

log = logging.getLogger(__name__)
SAVE_DIR = os.getenv("SAVE_FRAMES")
SAVE_EVERY = 2.0  # 초
MAX_SIDE = 1280  # 이보다 큰 사진은 줄여서 계산
SESSION_TTL = 600  # 초


class CollectSpeaker:
    """말할 문장을 모아 두었다가 다음 응답에 실어 보낸다."""

    def __init__(self):
        self.pending: str | None = None

    def say(self, text: str) -> None:
        self.pending = text

    def pop(self) -> str | None:
        text, self.pending = self.pending, None
        return text


@dataclass
class Session:
    app: ServingApp
    speaker: CollectSpeaker
    last_used: float = field(default_factory=time.time)


class Backend:
    def __init__(self, meal: Meal, bowls_mode: str):
        self.meal = meal
        self.resolver = build_resolver()
        self.finder = BowlFinder(bowls_mode)  # YOLO 모델은 하나를 같이 쓴다
        self.recognizer = FoodRecognizer() if FoodRecognizer.available() else None
        log.info("음식 인식: %s", "비전 LLM 사용" if self.recognizer else "꺼짐 (ANTHROPIC_API_KEY 없음 → meal.toml 순서)")
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()  # YOLO·OpenCV 계산은 한 번에 하나씩
        self._last_save = 0.0

    def maybe_save(self, data: bytes, session: str) -> None:
        if not SAVE_DIR or time.time() - self._last_save < SAVE_EVERY:
            return
        self._last_save = time.time()
        os.makedirs(SAVE_DIR, exist_ok=True)
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{session}.jpg"
        with open(os.path.join(SAVE_DIR, name), "wb") as f:
            f.write(data)

    def session(self, sid: str) -> Session:
        now = time.time()
        for k in [k for k, s in self.sessions.items() if now - s.last_used > SESSION_TTL]:
            if self.recognizer:
                self.recognizer.forget(self.sessions[k].app.key)
            del self.sessions[k]
        s = self.sessions.get(sid)
        if s is None:
            speaker = CollectSpeaker()
            # 휴대폰은 초당 몇 장만 보내므로 PC보다 오래 기다리고, 덜 매끄럽게 평균한다
            tracker = BowlTracker(lost_after=3.0, alpha=0.5, confirm_hits=2)
            s = Session(ServingApp(self.meal, self.resolver, self.finder, speaker, tracker, self.recognizer), speaker)
            self.sessions[sid] = s
            log.info("새 세션 %s", sid)
        s.last_used = now
        return s


def decode(data: bytes) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def to_json(a: Analysis, app: ServingApp, frame: np.ndarray, speak: str | None) -> dict:
    h, w = frame.shape[:2]
    bowls = []
    for b, plan, grams, pg in a.rows:
        hull = []
        if b.hull is not None:
            pts = b.hull.reshape(-1, 2)
            hull = pts[:: max(1, len(pts) // 40)].tolist()  # 점 40개 정도로 줄임
        bowls.append({
            "x": int(b.cx), "y": int(b.cy), "r": int(b.r),
            "food": plan.food if plan else None,
            "grams": None if grams is None else round(float(grams)),
            "kcal": round(float(grams * pg["열량"])) if grams is not None and pg and "열량" in pg else None,
            "rimMm": None if b.rim_mm is None else round(float(b.rim_mm)),
            "source": b.food_source,
            "hull": hull,
        })
    coin = app.coin.last_circle
    return {
        "width": w,
        "height": h,
        "scale": None if a.scale is None else round(float(a.scale), 3),
        "problem": a.problem,
        "coin": {"x": int(coin[0]), "y": int(coin[1]), "r": int(coin[2])} if coin else None,
        "bowls": bowls,
        "status": [{"text": t, "state": st} for t, st in
                   status_lines(a.portions, app.meal.targets, app.meal.tolerance)],
        "advice": None if a.advice is None else {"action": a.advice.action, "text": a.advice.speech()},
        "speak": speak,
        "recognition": app.recognizer is not None,
    }


def create_app(backend: Backend) -> FastAPI:
    api = FastAPI(title="foodcam")

    @api.get("/health")
    def health():
        return {"ok": True}

    @api.get("/meal")
    def meal():
        m = backend.meal
        return {
            "targets": {k: {"min": t.min, "max": t.max} for k, t in m.targets.items()},
            "bowls": [{"food": b.food, "type": b.type} for b in m.bowls],
        }

    @api.post("/frame")
    def frame(image: UploadFile = File(...), session: str = Form(...)):
        data = image.file.read()
        backend.maybe_save(data, session)
        try:
            img = decode(data)
        except Exception as e:
            raise HTTPException(400, f"사진을 읽을 수 없음: {e}") from e
        with backend.lock:
            s = backend.session(session)
            a = s.app.analyze(img, time.time())
            return to_json(a, s.app, img, s.speaker.pop())

    @api.post("/reset")
    def reset(session: str = Form(...)):
        with backend.lock:
            backend.session(session).app.tracker.reset_references()
        return {"ok": True}

    return api


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meal", default=os.getenv("MEAL_PATH", "meal.toml"))
    ap.add_argument("--bowls", choices=["yolo", "hough"], default=os.getenv("BOWLS_MODE", "yolo"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    uvicorn.run(create_app(Backend(load_meal(args.meal), args.bowls)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
