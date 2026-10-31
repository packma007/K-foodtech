"""배식 설정(meal.toml) 읽기."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .nutrition import UNITS


@dataclass
class BowlType:
    depth_ratio: float
    bottom_ratio: float


@dataclass
class BowlPlan:
    food: str
    type: str
    density: float
    per100g: dict[str, float] = field(default_factory=dict)


@dataclass
class Target:
    min: float | None = None
    max: float | None = None


@dataclass
class Meal:
    targets: dict[str, Target]
    bowls: list[BowlPlan]
    bowl_types: dict[str, BowlType]
    camera_height_mm: float = 600
    coin_diameter_mm: float = 26.5
    tolerance: float = 0.05


def load_meal(path: str | Path) -> Meal:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    s = raw.get("settings", {})

    targets = {}
    for name, t in raw.get("targets", {}).items():
        if name not in UNITS:
            raise ValueError(f"모르는 영양성분 이름: {name} (가능: {', '.join(UNITS)})")
        targets[name] = Target(t.get("min"), t.get("max"))

    types = {k: BowlType(v["depth_ratio"], v["bottom_ratio"]) for k, v in raw.get("bowl_types", {}).items()}
    bowls = []
    for b in raw.get("bowls", []):
        if b["type"] not in types:
            raise ValueError(f"[bowl_types]에 없는 그릇 종류: {b['type']}")
        bowls.append(BowlPlan(b["food"], b["type"], b.get("density", s.get("default_density", 0.9)),
                              b.get("per100g", {})))

    return Meal(
        targets=targets,
        bowls=bowls,
        bowl_types=types,
        camera_height_mm=s.get("camera_height_mm", 600),
        coin_diameter_mm=s.get("coin_diameter_mm", 26.5),
        tolerance=s.get("tolerance", 0.05),
    )
