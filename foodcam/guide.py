"""목표 영양성분과 비교해 '덜어라 / 더 담아라'를 정한다.

1) 상한을 넘은 영양소가 있으면 (넘은 비율이 큰 것부터)
   → 그 영양소를 가장 많이 낸 그릇에서, 넘친 양 ÷ 그 음식의 g당 함량 만큼 덜기
2) 하한에 모자란 영양소가 있으면
   → 더 담아도 다른 상한을 넘지 않는 그릇 중 g당 함량이 가장 높은 그릇에 더 담기
3) 모두 범위 안이면 '알맞음'
상한 초과를 먼저 안내한다 (과다 섭취가 부족보다 우선).
"""
from __future__ import annotations

from dataclasses import dataclass

from .meal import Target
from .nutrition import UNITS


@dataclass
class Portion:
    food: str
    grams: float
    per_gram: dict[str, float]  # 영양소 → g당 함량


@dataclass
class Advice:
    action: str  # "less" | "more" | "ok"
    food: str = ""
    grams: float = 0.0
    nutrient: str = ""

    @property
    def key(self) -> tuple:
        """같은 안내인지 비교할 때 쓰는 값. 덜거나 담는 중에는 양이 계속 바뀌므로 양은 뺀다."""
        return self.action, self.food, self.nutrient

    def speech(self) -> str:
        if self.action == "ok":
            return "좋아요. 알맞게 담았어요."
        i = "이" if _has_batchim(self.nutrient) else "가"
        if not self.food:  # 모자라지만 담긴 음식 중 채울 수 있는 게 없음 (빈 그릇 등)
            return f"{self.nutrient}{i} 부족해요. 음식을 더 담아 주세요."
        g = max(10, int(round(self.grams / 10) * 10))
        eul = "을" if _has_batchim(self.food) else "를"
        if self.action == "less":
            return f"{self.food}{eul} 약 {g}그램 덜어 주세요. {self.nutrient}{i} 많아요."
        return f"{self.food}{eul} 약 {g}그램 더 담아 주세요. {self.nutrient}{i} 부족해요."


def _has_batchim(word: str) -> bool:
    c = word[-1] if word else ""
    return "가" <= c <= "힣" and (ord(c) - 0xAC00) % 28 != 0


def totals(portions: list[Portion]) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in portions:
        for n, v in p.per_gram.items():
            out[n] = out.get(n, 0.0) + v * p.grams
    return out


def advise(portions: list[Portion], targets: dict[str, Target], tol: float = 0.05) -> Advice:
    tot = totals(portions)

    over = [(tot.get(n, 0) / t.max, n) for n, t in targets.items()
            if t.max is not None and tot.get(n, 0) > t.max * (1 + tol)]
    for _, n in sorted(over, reverse=True):
        culprit = max(portions, key=lambda p: p.per_gram.get(n, 0) * p.grams)
        per_g = culprit.per_gram.get(n, 0)
        if per_g > 0:
            grams = min((tot[n] - targets[n].max) / per_g, culprit.grams)
            return Advice("less", culprit.food, grams, n)

    for n, t in targets.items():
        if t.min is None or tot.get(n, 0) >= t.min * (1 - tol):
            continue
        need = t.min - tot.get(n, 0)
        options = []
        for p in portions:
            per_g = p.per_gram.get(n, 0)
            if per_g <= 0:
                continue
            grams = need / per_g
            breaks = any(tg.max is not None and tot.get(m, 0) + p.per_gram.get(m, 0) * grams > tg.max * (1 + tol)
                         for m, tg in targets.items())
            options.append((breaks, -per_g, p, grams))
        if options:
            _, _, p, grams = min(options, key=lambda o: (o[0], o[1]))
            return Advice("more", p.food, grams, n)
        return Advice("more", "", 0.0, n)  # 담긴 음식으로는 채울 수 없음

    return Advice("ok")


def status_lines(portions: list[Portion], targets: dict[str, Target], tol: float = 0.05) -> list[tuple[str, str]]:
    """화면 표시용: [(문구, 상태)]  상태: ok | over | under"""
    tot = totals(portions)
    out = []
    for n, t in targets.items():
        v = tot.get(n, 0)
        rng = f"{t.min:g}" if t.min is not None else ""
        rng += "~" + (f"{t.max:g}" if t.max is not None else "")
        state = "ok"
        if t.max is not None and v > t.max * (1 + tol):
            state = "over"
        elif t.min is not None and v < t.min * (1 - tol):
            state = "under"
        out.append((f"{n} {v:.0f}{UNITS[n]} (목표 {rng})", state))
    return out
