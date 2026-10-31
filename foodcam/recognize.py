"""그릇에 담긴 음식이 무엇인지 비전 LLM(Claude)으로 알아낸다.

매 사진마다 묻지 않는다. 그릇마다 음식이 처음 보일 때, 그리고 내용물이 크게 바뀌었을 때만
백그라운드에서 한 번 묻고 결과를 그릇에 붙여 둔다 (한 끼에 몇 번 정도).

영양성분은 돌려받은 음식 이름으로 식약처 DB를 먼저 찾고, 없을 때만 여기서 받은 AI 추정값을 쓴다.
API 키: 환경 변수 ANTHROPIC_API_KEY (.env 또는 Render 환경 변수).
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODEL = os.getenv("FOODCAM_VISION_MODEL", "claude-opus-5-5")
CROP_MAX_SIDE = 512  # 이미지 토큰을 줄이려고 그릇 부분만 이 크기로
RECHECK_DIFF = 25.0  # 마지막으로 물어본 뒤 그릇 모습이 이만큼(평균 화소 차) 바뀌면 다시 묻는다
RECHECK_MIN_S = 6.0  # 같은 그릇은 이 간격보다 자주 묻지 않는다

NUTRIENTS = ["열량", "탄수화물", "당류", "단백질", "지방", "포화지방", "나트륨"]

SCHEMA = {
    "type": "object",
    "properties": {
        "is_food": {"type": "boolean", "description": "그릇에 음식이 담겨 있으면 true, 비었거나 음식이 아니면 false"},
        "name": {"type": "string", "description": "한국에서 부르는 음식 이름 (예: 마라샹궈, 쌀밥, 된장찌개)"},
        "db_name": {"type": "string", "description": "식약처 영양성분DB에서 찾을 일반 명칭. 브랜드·수식어 없이 (예: 마라샹궈, 쌀밥)"},
        "confidence": {"type": "number", "description": "0~1"},
        "density_g_per_ml": {"type": "number", "description": "그릇에 담긴 상태의 겉보기 밀도 g/mL (밥 약 0.85, 국 1.0, 볶음·샹궈류 0.6~0.8)"},
        "per100g": {
            "type": "object",
            "description": "100g당 영양성분 추정값. 열량 kcal, 나트륨 mg, 나머지 g",
            "properties": {k: {"type": "number"} for k in NUTRIENTS},
            "required": NUTRIENTS,
            "additionalProperties": False,
        },
    },
    "required": ["is_food", "name", "db_name", "confidence", "density_g_per_ml", "per100g"],
    "additionalProperties": False,
}

PROMPT = (
    "위에서 내려다본 그릇 사진입니다. 그릇에 담긴 음식이 무엇인지 한국어 이름으로 알려 주세요. "
    "한국 식당·가정에서 흔한 메뉴(중식·분식·외식 포함)일 가능성이 높습니다. "
    "확실하지 않으면 가장 그럴듯한 이름을 고르고 confidence를 낮게 주세요. "
    "그릇이 비어 있거나 음식이 아니면 is_food=false로 답하세요."
)


@dataclass
class Recognition:
    is_food: bool
    name: str
    db_name: str
    confidence: float
    density: float
    per100g: dict[str, float] = field(default_factory=dict)


@dataclass
class _BowlState:
    result: Recognition | None = None
    pending: bool = False
    asked_at: float = 0.0
    asked_crop: np.ndarray | None = None
    error: str | None = None


class FoodRecognizer:
    def __init__(self, api_key: str | None = None):
        import anthropic  # 키가 없는 환경에서도 서버는 뜨도록 여기서 불러온다

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vision")
        self._lock = threading.Lock()
        self._states: dict[tuple[str, int], _BowlState] = {}

    @staticmethod
    def available() -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))

    # ── 서버 루프에서 매 사진마다 부르는 함수 (막히지 않음) ──────────────
    def lookup(self, key: tuple[str, int], crop: np.ndarray, photo: np.ndarray, has_food: bool) -> _BowlState:
        """key=(세션, 그릇 id), crop=바뀜 비교용 작은 그릇 사진, photo=AI에 보낼 그릇 사진.
        필요하면 백그라운드 인식을 시작하고 지금까지의 상태를 돌려준다."""
        now = time.time()
        with self._lock:
            st = self._states.setdefault(key, _BowlState())
            if st.pending or not has_food:
                return st
            changed = st.asked_crop is None or float(np.mean(cv2.absdiff(crop, st.asked_crop))) > RECHECK_DIFF
            if changed and now - st.asked_at >= RECHECK_MIN_S:
                st.pending, st.asked_at, st.asked_crop = True, now, crop.copy()
                self._pool.submit(self._run, key, photo.copy())
            return st

    def state(self, key: tuple[str, int]) -> _BowlState:
        with self._lock:
            return self._states.setdefault(key, _BowlState())

    def forget(self, session: str) -> None:
        with self._lock:
            for k in [k for k in self._states if k[0] == session]:
                del self._states[k]

    # ── 내부 ──────────────────────────────────────────────────────
    def _run(self, key, crop: np.ndarray) -> None:
        try:
            result, error = self.identify(crop), None
            log.info("음식 인식 %s → %s (%.2f)", key, result.name if result.is_food else "음식 아님", result.confidence)
        except Exception as e:  # 인식 실패해도 배식 계산은 계속
            result, error = None, f"{type(e).__name__}: {e}"
            log.warning("음식 인식 실패 %s: %s", key, error)
        with self._lock:
            st = self._states.setdefault(key, _BowlState())
            st.pending, st.error = False, error
            if result is not None:
                st.result = result

    def identify(self, crop_bgr: np.ndarray) -> Recognition:
        h, w = crop_bgr.shape[:2]
        k = CROP_MAX_SIDE / max(h, w)
        if k < 1:
            crop_bgr = cv2.resize(crop_bgr, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError("JPEG 인코딩 실패")
        data = base64.standard_b64encode(buf.tobytes()).decode("ascii")

        response = self.client.beta.messages.create(
            model=MODEL,
            max_tokens=2000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # 안전 분류기가 거절하면 서버가 권장 모델로 다시 실행
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("모델이 응답을 거절함")
        text = next((b.text for b in response.content if b.type == "text"), "")
        d = json.loads(text)
        return Recognition(
            is_food=bool(d["is_food"]),
            name=d["name"].strip(),
            db_name=(d.get("db_name") or d["name"]).strip(),
            confidence=float(d["confidence"]),
            density=min(max(float(d["density_g_per_ml"]), 0.3), 1.3),
            per100g={k: float(v) for k, v in d["per100g"].items()},
        )
