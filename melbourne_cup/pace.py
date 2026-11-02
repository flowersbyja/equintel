"""
Race shape: speed map, pace pressure, scenario adjustments and the Cup-Day bias switch.

Mirrors racing-predictor's race-shape module (habitual style -> pace pressure) but for one race with
hand-set predicted settling positions, and adds:
  * a style prior from the historical first-four records (how often each running style has figured),
  * a pace-scenario adjustment (hot early tempo favours the swoopers, an uncontested lead favours the leaders),
  * a day-of-meeting bias switch: set --bias on-pace if on-speed horses are winning the early races on Cup Day,
    --bias closers if they are being swamped, neutral otherwise.
"""
from __future__ import annotations

import math
from collections import Counter

STYLES = ["LEADER", "ON_PACE", "MIDFIELD", "BACKMARKER"]


def speed_map(runners: list[dict]) -> dict:
    ordered = sorted(runners, key=lambda r: r.get("settle", 12))
    groups = {"leaders": [], "on_pace": [], "midfield": [], "back": []}
    for r in ordered:
        s = r.get("settle", 12)
        key = "leaders" if s <= 3 else "on_pace" if s <= 8 else "midfield" if s <= 16 else "back"
        groups[key].append(r["horse"])
    return groups


def pace_pressure(runners: list[dict]) -> tuple[str, str]:
    """Same thresholds as racing-predictor summarizePaceShape: leaders + 0.5 x on-pace."""
    n_lead = sum(1 for r in runners if r["style"] == "LEADER")
    n_on = sum(1 for r in runners if r["style"] == "ON_PACE")
    score = n_lead + 0.5 * n_on
    if n_lead <= 1 and n_on == 0:
        return "uncontested", f"{n_lead} leader, no on-pacers"
    if score >= 3:
        return "hot", f"{n_lead} habitual leader(s) and {n_on} on-pace types drawn across the track"
    return "moderate", f"{n_lead} leader(s), {n_on} on-pace types"


SCENARIO_ADJ = {
    "hot": {"LEADER": -0.20, "ON_PACE": -0.08, "MIDFIELD": +0.04, "BACKMARKER": +0.12},
    "moderate": {"LEADER": 0.0, "ON_PACE": 0.0, "MIDFIELD": 0.0, "BACKMARKER": 0.0},
    "uncontested": {"LEADER": +0.30, "ON_PACE": +0.15, "MIDFIELD": -0.05, "BACKMARKER": -0.20},
}

BIAS_ADJ = {
    "neutral": {"LEADER": 0.0, "ON_PACE": 0.0, "MIDFIELD": 0.0, "BACKMARKER": 0.0},
    "on-pace": {"LEADER": +1.0, "ON_PACE": +0.7, "MIDFIELD": -0.1, "BACKMARKER": -0.8},
    "closers": {"LEADER": -0.8, "ON_PACE": -0.5, "MIDFIELD": +0.2, "BACKMARKER": +0.8},
}


def style_history_prior(history: list[dict], runners: list[dict], weight: float = 0.15, cap: float = 0.20) -> dict[str, float]:
    """log( share of past first-four finishers with this style / share of this field with this style ), damped."""
    hist = Counter()
    for t in history:
        hist[t["style"]] += {1: 1.0, 2: 0.6, 3: 0.4, 4: 0.3}.get(t["pos"], 0.2) * t.get("confidence", 0.8)
    field = Counter(r["style"] for r in runners)
    out = {}
    for s in STYLES:
        h = (hist[s] + 0.5) / (sum(hist.values()) + 2.0)
        f = (field[s] + 0.5) / (len(runners) + 2.0)
        out[s] = max(-cap, min(cap, weight * math.log(h / f)))
    return out


def pace_adjustment(runner: dict, pressure: str, bias: str, bias_strength: float, style_prior: dict[str, float]) -> tuple[float, list[str]]:
    s = runner["style"]
    notes = []
    adj = SCENARIO_ADJ[pressure][s]
    if adj:
        notes.append(f"{pressure} tempo scenario {adj:+.2f}")
    b = BIAS_ADJ[bias][s] * bias_strength
    if b:
        notes.append(f"day bias '{bias}' {b:+.2f}")
    p = style_prior.get(s, 0.0)
    if p:
        notes.append(f"style history {p:+.2f}")
    return adj + b + p, notes
