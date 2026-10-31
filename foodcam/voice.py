"""음성 안내. 전용 스레드에서 말하고, 말하는 동안 들어온 요청은 가장 최근 것 하나만 남긴다."""
from __future__ import annotations

import logging
import queue
import threading

log = logging.getLogger(__name__)


class Speaker:
    def __init__(self, enabled: bool = True, rate: int = 180):
        self.enabled = enabled
        self.rate = rate
        self._q: queue.Queue[str] = queue.Queue(maxsize=1)
        if enabled:
            threading.Thread(target=self._loop, daemon=True, name="speaker").start()

    def say(self, text: str) -> None:
        log.info("🔊 %s", text)
        if not self.enabled:
            return
        try:  # 밀린 안내는 버리고 최신 것만
            self._q.get_nowait()
        except queue.Empty:
            pass
        self._q.put_nowait(text)

    def _loop(self) -> None:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        for v in engine.getProperty("voices"):
            if "ko" in (v.id + v.name).lower() or "heami" in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        else:
            log.warning("한국어 음성을 찾지 못했습니다. 기본 음성으로 말합니다.")
        while True:
            text = self._q.get()
            engine.say(text)
            engine.runAndWait()


class AdviceAnnouncer:
    """같은 판단이 stable_s 동안 유지될 때만 말하고, 같은 말은 repeat_s 뒤에 다시 한다."""

    def __init__(self, speaker: Speaker, stable_s: float = 1.2, repeat_s: float = 8.0):
        self.speaker = speaker
        self.stable_s = stable_s
        self.repeat_s = repeat_s
        self._cand_key = None
        self._cand_since = 0.0
        self._spoken_key = None
        self._spoken_at = -1e9

    def update(self, advice, now: float) -> None:
        if advice.key != self._cand_key:
            self._cand_key, self._cand_since = advice.key, now
            return
        if now - self._cand_since < self.stable_s:
            return
        if advice.key == self._spoken_key:
            if advice.action == "ok" or now - self._spoken_at < self.repeat_s:
                return  # '알맞아요'는 한 번만
        self.speaker.say(advice.speech())
        self._spoken_key, self._spoken_at = advice.key, now
