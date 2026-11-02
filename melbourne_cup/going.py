"""
Going (track condition) model.

The race file carries a probability distribution over the expected rating. Each horse gets a going-fit
score in [-1, 1] that blends (a) the published wet-track record where one exists and (b) a prior from
connections' comments / pedigree (wet_rating). The adjustment to the logit is wetness x fit x scale, so
on a Good 4 nobody moves and on a Soft 7 the wet specialists and the firm-ground horses separate.
"""
from __future__ import annotations

import math

WETNESS = {"GOOD": 0.0, "SOFT5": 0.45, "SOFT6": 0.8, "SOFT7_PLUS": 1.0}

PRESETS = {
    "good4": {"GOOD": 1.0, "SOFT5": 0.0, "SOFT6": 0.0, "SOFT7_PLUS": 0.0},
    "soft5": {"GOOD": 0.0, "SOFT5": 1.0, "SOFT6": 0.0, "SOFT7_PLUS": 0.0},
    "soft6": {"GOOD": 0.0, "SOFT5": 0.0, "SOFT6": 1.0, "SOFT7_PLUS": 0.0},
    "soft7": {"GOOD": 0.0, "SOFT5": 0.0, "SOFT6": 0.0, "SOFT7_PLUS": 1.0},
    "heavy": {"GOOD": 0.0, "SOFT5": 0.0, "SOFT6": 0.0, "SOFT7_PLUS": 1.0},
    "forecast": None,  # use the distribution in the race file
}


def expected_wetness(dist: dict) -> float:
    total = sum(dist.values()) or 1.0
    return sum(WETNESS[k] * v for k, v in dist.items()) / total


def record_score(rec: dict | None) -> float | None:
    """Wet-record quality on [-1, 1]; None when there is no usable sample (< 3 starts)."""
    if not rec or rec.get("starts", 0) < 3:
        return None
    starts = rec["starts"]
    quality = (rec.get("wins", 0) + 0.5 * (rec.get("seconds", 0) + rec.get("thirds", 0))) / starts
    # 0.30 (a win-or-place every third start) is treated as neutral for a Cup-class stayer
    return math.tanh((quality - 0.30) * 4.0)


def going_fit(runner: dict) -> dict:
    prior = float(runner.get("wet_rating", 0.0))
    rec = record_score(runner.get("wet_record"))
    if rec is None:
        fit, basis = prior, "connections / pedigree only"
    else:
        fit, basis = 0.5 * prior + 0.5 * rec, f"wet record {runner['wet_record']['starts']} starts (score {rec:+.2f}) blended with prior {prior:+.1f}"
    return {"fit": max(-1.0, min(1.0, fit)), "basis": basis}


def going_adjustment(runner: dict, dist: dict, scale: float = 0.45) -> tuple[float, str]:
    g = going_fit(runner)
    wet = expected_wetness(dist)
    adj = scale * wet * g["fit"]
    return adj, f"going fit {g['fit']:+.2f} x wetness {wet:.2f} -> {adj:+.2f} ({g['basis']})"
