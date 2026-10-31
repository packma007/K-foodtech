"""카메라 없이 영양성분 조회만 확인.

    python -m foodcam.lookup 8801043014809   # 바코드 (숫자만이면 바코드로 봄)
    python -m foodcam.lookup 김치찌개         # 음식 이름
"""
from __future__ import annotations

import logging
import sys

from .main import build_resolver
from .nutrition import FOUND


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    resolver = build_resolver()
    for q in sys.argv[1:]:
        kind = "barcode" if q.isdigit() else "food"
        lk = resolver.resolve(kind, q)
        print(f"\n■ {q} ({kind})")
        if lk.status != FOUND:
            print(f"  {lk.status} {lk.message}")
            continue
        n = lk.nutrition
        print(f"  {n.name} · {n.maker}  [{n.source}]  기준량 {n.basis or '-'}  총중량 {n.weight or '-'}")
        for k, v in n.values:
            print(f"  {k:<6} {v}")


if __name__ == "__main__":
    main()
