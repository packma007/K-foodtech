"""영양성분 조회.

우선순위
  1순위  실제 판매 제품의 영양성분
         - 바코드 → 식품안전나라(C005/I2570) → 품목제조보고번호
           → 식약처 식품영양성분DB의 '상용제품' 데이터
         - 식약처에 없으면 Open Food Facts(바코드)
  2순위  식약처 식품영양성분DB의 '품목대표' 값 (제품명/식품유형/음식명으로 검색)

카메라 루프가 멈추지 않도록 조회는 백그라운드 스레드에서 하고 결과는 캐시한다.
"""
from __future__ import annotations

import difflib
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import requests

log = logging.getLogger(__name__)

FOODSAFETY_URL = "http://openapi.foodsafetykorea.go.kr/api/{key}/{svc}/json/1/{n}/{cond}"
NUTRI_URL = "https://apis.data.go.kr/1471000/FoodNtrCpntDbInfo03/getFoodNtrCpntDbInq03"
OFF_URL = "https://world.openfoodfacts.org/api/v2/product/{code}.json"

# 식약처 식품영양성분DB 응답 키 → (표시 이름, 단위). 식품 영양표시 9개 항목 순서.
MFDS_FIELDS = [
    ("AMT_NUM1", "열량", "kcal"),
    ("AMT_NUM6", "탄수화물", "g"),
    ("AMT_NUM7", "당류", "g"),
    ("AMT_NUM3", "단백질", "g"),
    ("AMT_NUM4", "지방", "g"),
    ("AMT_NUM24", "포화지방", "g"),
    ("AMT_NUM25", "트랜스지방", "g"),
    ("AMT_NUM23", "콜레스테롤", "mg"),
    ("AMT_NUM13", "나트륨", "mg"),
]

# Open Food Facts nutriments 키 → (표시 이름, 단위, 배율). 콜레스테롤·나트륨은 g 단위로 와서 mg로 바꾼다.
OFF_FIELDS = [
    ("energy-kcal_100g", "열량", "kcal", 1),
    ("carbohydrates_100g", "탄수화물", "g", 1),
    ("sugars_100g", "당류", "g", 1),
    ("proteins_100g", "단백질", "g", 1),
    ("fat_100g", "지방", "g", 1),
    ("saturated-fat_100g", "포화지방", "g", 1),
    ("trans-fat_100g", "트랜스지방", "g", 1),
    ("cholesterol_100g", "콜레스테롤", "mg", 1000),
    ("sodium_100g", "나트륨", "mg", 1000),
]

PENDING, FOUND, NOT_FOUND, ERROR = "조회 중", "찾음", "정보 없음", "오류"
RETRY_AFTER_ERROR = 30.0  # 초


@dataclass
class Nutrition:
    name: str
    source: str  # 화면에 보여줄 출처: "식약처 상용제품", "Open Food Facts", "식약처 품목대표"
    maker: str = ""
    basis: str = ""  # 영양성분 함량 기준량 (예: 100g)
    weight: str = ""  # 제품 총중량
    values: list[tuple[str, str]] = field(default_factory=list)  # [(표시 이름, "421 kcal"), ...]
    amounts: dict[str, float] = field(default_factory=dict)  # 표시 이름 → basis 기준 함량 (계산용)

    @property
    def basis_g(self) -> float:
        """영양성분 기준량을 g(mL)로. "100g", "100 mL", "1회(30g)" 등에서 숫자를 뽑고, 없으면 100."""
        m = re.search(r"([\d.]+)\s*(g|ml|mL|㎖)", self.basis or "")
        return float(m.group(1)) if m else 100.0

    def per_gram(self) -> dict[str, float]:
        return {k: v / self.basis_g for k, v in self.amounts.items()}


UNITS = {label: unit for _, label, unit in MFDS_FIELDS}  # "나트륨" → "mg"


@dataclass
class Lookup:
    status: str
    nutrition: Nutrition | None = None
    message: str = ""
    updated: float = field(default_factory=time.time)


@dataclass
class Product:
    report_no: str  # 품목제조보고번호
    name: str
    maker: str
    category: str  # 식품유형 (예: 유탕면)


def _num(v) -> str | None:
    if v is None or str(v).strip() in ("", "-", "N/A"):
        return None
    try:
        f = float(str(v).replace(",", ""))
    except ValueError:
        return None
    return f"{f:.0f}" if f >= 100 or f == int(f) else f"{f:.1f}"


def _norm(s: str) -> str:
    return re.sub(r"[\s()\[\]{}·,._\-/]", "", s or "").lower()


def _similarity(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    score = difflib.SequenceMatcher(None, a, b).ratio()
    if a in b or b in a:
        score = max(score, 0.75)
    return score


class NutritionResolver:
    def __init__(self, mfds_key: str = "", foodsafety_key: str = "sample", timeout: float = 8.0):
        self.mfds_key = mfds_key.strip()
        self.foodsafety_key = foodsafety_key.strip() or "sample"
        self.timeout = timeout
        self.session = requests.Session()
        self._cache: dict[str, Lookup] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="nutri")
        if not self.mfds_key:
            log.warning("MFDS_SERVICE_KEY가 없어 식약처 영양성분DB를 건너뜁니다 (Open Food Facts만 사용).")

    # ── 외부에서 쓰는 함수 ─────────────────────────────────────────
    def get(self, kind: str, query: str) -> Lookup:
        """kind: "barcode" 또는 "food". 결과가 없으면 조회를 시작하고 PENDING을 돌려준다."""
        key = f"{kind}:{query}"
        with self._lock:
            hit = self._cache.get(key)
            stale = hit and hit.status == ERROR and time.time() - hit.updated > RETRY_AFTER_ERROR
            if hit is None or stale:
                hit = Lookup(PENDING)
                self._cache[key] = hit
                self._pool.submit(self._run, key, kind, query)
        return hit

    def resolve(self, kind: str, query: str) -> Lookup:
        """동기 조회 (CLI·테스트용)."""
        return self._lookup(kind, query)

    # ── 내부 ──────────────────────────────────────────────────────
    def _run(self, key: str, kind: str, query: str) -> None:
        result = self._lookup(kind, query)
        with self._lock:
            self._cache[key] = result
        if result.status == FOUND:
            n = result.nutrition
            log.info("[%s] %s → %s (%s)", kind, query, n.name, n.source)
        else:
            log.info("[%s] %s → %s %s", kind, query, result.status, result.message)

    def _lookup(self, kind: str, query: str) -> Lookup:
        try:
            if kind == "barcode":
                return self._by_barcode(query)
            return self._by_food_name(query)
        except requests.RequestException as e:
            return Lookup(ERROR, message=f"네트워크 오류: {e.__class__.__name__}")
        except Exception as e:  # API 응답 형식이 바뀌어도 화면 루프는 살아 있게
            log.exception("조회 실패: %s %s", kind, query)
            return Lookup(ERROR, message=str(e))

    def _by_barcode(self, code: str) -> Lookup:
        products = self._products_for_barcode(code)

        # 1순위-a: 식약처 상용제품 (품목제조보고번호로 정확히 찾기)
        for p in products:
            rows = [r for r in self._nutri(ITEM_REPORT_NO=p.report_no) if self._is_commercial(r)]
            best = self._best(rows, p.name, p.maker, threshold=0.3)
            if best:
                return Lookup(FOUND, self._from_mfds(best, "식약처 상용제품"))

        # 1순위-b: 식약처 상용제품 (제품명 + 업체명으로 찾기)
        for p in products:
            rows = [r for r in self._nutri(FOOD_NM_KR=p.name) if self._is_commercial(r)]
            best = self._best(rows, p.name, p.maker, threshold=0.6)
            if best:
                return Lookup(FOUND, self._from_mfds(best, "식약처 상용제품"))

        # 1순위-c: Open Food Facts (바코드)
        off = self._open_food_facts(code)
        if off:
            if products:  # 식품안전나라의 한글 제품명·업체명을 우선
                off.name, off.maker = products[0].name, products[0].maker
            return Lookup(FOUND, off)

        # 2순위: 식약처 품목대표 (제품명 → 식품유형 순)
        for p in products:
            for name in (p.name, p.category):
                rep = self._representative(name)
                if rep:
                    rep.name = f"{p.name} (≈{rep.name})"
                    rep.maker = p.maker
                    return Lookup(FOUND, rep)

        if products:
            return Lookup(NOT_FOUND, message=f"{products[0].name}: 영양정보 없음")
        return Lookup(NOT_FOUND, message="등록되지 않은 바코드")

    def _by_food_name(self, name: str) -> Lookup:
        # 화면 인식만으로는 어느 회사 제품인지 모르므로 품목대표값을 쓴다.
        rep = self._representative(name)
        if rep:
            return Lookup(FOUND, rep)
        return Lookup(NOT_FOUND, message=f"{name}: 식약처 DB에 없음")

    def _representative(self, name: str) -> Nutrition | None:
        if not name:
            return None
        rows = self._nutri(FOOD_NM_KR=name)
        reps = [r for r in rows if "대표" in (r.get("DB_CLASS_NM") or "")]
        best = self._best(reps, name, threshold=0.5)
        return self._from_mfds(best, "식약처 품목대표") if best else None

    # ── 데이터 소스 ───────────────────────────────────────────────
    def _foodsafety(self, svc: str, cond: str, n: int = 10) -> list[dict]:
        url = FOODSAFETY_URL.format(key=self.foodsafety_key, svc=svc, n=n, cond=cond)
        data = self.session.get(url, timeout=self.timeout).json().get(svc, {})
        code = data.get("RESULT", {}).get("CODE", "")
        if code not in ("INFO-000", "INFO-200"):
            log.warning("식품안전나라 %s 응답: %s", svc, data.get("RESULT"))
        return data.get("row") or []

    def _products_for_barcode(self, code: str) -> list[Product]:
        found: dict[str, Product] = {}
        for r in self._foodsafety("C005", f"BAR_CD={code}"):
            p = Product(r.get("PRDLST_REPORT_NO", ""), r.get("PRDLST_NM", ""),
                        r.get("BSSH_NM", ""), r.get("PRDLST_DCNM", ""))
            found.setdefault(p.report_no, p)
        if not found:  # 유통바코드 DB에 한 번 더
            for r in self._foodsafety("I2570", f"BRCD_NO={code}"):
                p = Product(r.get("PRDLST_REPORT_NO", ""), r.get("PRDT_NM", ""),
                            r.get("CMPNY_NM", ""), r.get("PRDLST_NM", ""))
                found.setdefault(p.report_no, p)
        return [p for p in found.values() if p.report_no or p.name]

    def _nutri(self, **cond) -> list[dict]:
        if not self.mfds_key or not any(cond.values()):
            return []
        params = {"serviceKey": self.mfds_key, "type": "json", "pageNo": 1, "numOfRows": 50, **cond}
        resp = self.session.get(NUTRI_URL, params=params, timeout=self.timeout)
        try:
            data = resp.json()
        except ValueError:  # 인증키 오류 등은 XML로 온다
            log.warning("식약처 영양성분DB 응답이 JSON이 아님: %s", resp.text[:200])
            return []
        body = data.get("body") or data.get("response", {}).get("body") or {}
        items = body.get("items") or []
        if isinstance(items, dict):
            items = items.get("item") or []
        if isinstance(items, dict):
            items = [items]
        return items

    def _open_food_facts(self, code: str) -> Nutrition | None:
        resp = self.session.get(
            OFF_URL.format(code=code),
            params={"fields": "product_name,product_name_ko,brands,quantity,nutriments"},
            headers={"User-Agent": "kfoodtech-foodcam/0.1"},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            return None
        prod = resp.json().get("product") or {}
        nut = prod.get("nutriments") or {}
        values, amounts = [], {}
        for key, label, unit, scale in OFF_FIELDS:
            v = nut.get(key)
            if isinstance(v, (int, float)):
                amounts[label] = float(v) * scale
                values.append((label, f"{_num(amounts[label])} {unit}"))
        if not values:
            return None
        return Nutrition(
            name=prod.get("product_name_ko") or prod.get("product_name") or code,
            source="Open Food Facts",
            maker=prod.get("brands", ""),
            basis="100g",
            weight=prod.get("quantity", ""),
            values=values,
            amounts=amounts,
        )

    # ── 보조 ──────────────────────────────────────────────────────
    @staticmethod
    def _is_commercial(row: dict) -> bool:
        return "상용" in (row.get("DB_CLASS_NM") or "")

    @staticmethod
    def _best(rows: list[dict], name: str, maker: str = "", threshold: float = 0.5) -> dict | None:
        def score(r: dict) -> float:
            s = _similarity(r.get("FOOD_NM_KR", ""), name)
            if maker and _norm(maker)[:4] and _norm(maker)[:4] in _norm(r.get("MAKER_NM", "")):
                s += 0.2
            return s

        if not rows:
            return None
        best = max(rows, key=score)
        return best if score(best) >= threshold else None

    @staticmethod
    def _from_mfds(row: dict, source: str) -> Nutrition:
        values, amounts = [], {}
        for key, label, unit in MFDS_FIELDS:
            s = _num(row.get(key))
            if s is not None:
                values.append((label, f"{s} {unit}"))
                amounts[label] = float(str(row[key]).replace(",", ""))
        return Nutrition(
            amounts=amounts,
            name=row.get("FOOD_NM_KR", ""),
            source=source,
            maker=row.get("MAKER_NM") or "",
            basis=row.get("SERVING_SIZE") or "",
            weight=row.get("Z10500") or "",
            values=values,
        )
