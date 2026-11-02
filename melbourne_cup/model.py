"""
Ensemble + first-four simulation.

  logit_i = w_market * log(market_i) + w_form * z(form_i) + w_pattern * z(pattern_i) + priors_i
  win_prob = softmax(logit)
  first-four: Plackett-Luce sampling (Gumbel-max trick) with win_prob as strengths.

Market weight < 1 shrinks the (well-informed) market towards the model's own opinion; the other two terms
are the tempo-adjusted form and the preparation-formula similarity. Priors are small, transparent
structural adjustments (weight, barrier, age, sex, expected going, slowly run lead-up).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .features import form_rating
from .going import PRESETS, going_adjustment
from .pace import pace_adjustment, pace_pressure, speed_map, style_history_prior
from .patterns import ArchetypeIndex, load_json, pattern_scores

LEADUP_GROUP_RULES = [
    ("caulfield cup", "CAULFIELD_CUP"),
    ("cox plate", "COX_PLATE"),
    ("moonee valley", "MV_CUP"),
    ("geelong cup", "GEELONG_CUP"),
    ("bendigo cup", "BENDIGO_CUP"),
    ("herbert power", "HERBERT_POWER"),
    ("bart cummings", "BART_CUMMINGS"),
    ("archer", "LEXUS_HOTHAM"),
    ("hotham", "LEXUS_HOTHAM"),
    ("lexus stakes", "LEXUS_HOTHAM"),
    ("mackinnon", "MACKINNON"),
    ("turnbull", "TURNBULL"),
]
EURO_VENUES = {"leopardstown", "curragh", "york", "ascot", "newmarket", "goodwood", "longchamp", "doncaster", "newbury", "epsom"}


def derive_leadup_group(runner: dict) -> str:
    lu = runner["leadups"][0]
    name = lu["race"].lower()
    if lu.get("venue", "").lower() in EURO_VENUES:
        return "EURO_DIRECT"
    for needle, group in LEADUP_GROUP_RULES:
        if needle in name:
            return group
    return "OTHER_AUS"


@dataclass
class Weights:
    market: float = 0.75
    form: float = 0.30
    pattern: float = 0.30
    slow_leadup_penalty: float = 0.30
    going_scale: float = 0.45
    bias: str = "neutral"          # neutral | on-pace | closers  (Cup Day track bias switch)
    bias_strength: float = 0.35
    track: str = "forecast"        # forecast | good4 | soft5 | soft6 | soft7 | heavy


@dataclass
class Prediction:
    race: dict
    runners: list[dict]
    win_prob: np.ndarray
    place_prob: np.ndarray
    top4_prob: np.ndarray
    components: list[dict]
    pattern: dict
    form: dict
    first_four_exact: list[tuple[tuple[str, ...], float]] = field(default_factory=list)
    first_four_boxed: list[tuple[frozenset, float]] = field(default_factory=list)
    weights: Weights = field(default_factory=Weights)
    sims: int = 0
    speed_map: dict = field(default_factory=dict)
    pace: tuple = ("moderate", "")
    track_distribution: dict = field(default_factory=dict)
    idx: object = None


def _z(values: np.ndarray) -> np.ndarray:
    sd = values.std()
    return (values - values.mean()) / sd if sd > 1e-9 else np.zeros_like(values)


def structural_priors(r: dict, race: dict, w: Weights, last_tempo: str, trainer_record: dict | None = None) -> tuple[float, list[str]]:
    notes = []
    p = 0.0
    if trainer_record is not None:
        tr = min(0.25, 0.08 * trainer_record.get("wins", 0) + 0.03 * trainer_record.get("placings", 0))
        if trainer_record.get("runners", 0) == 0:
            tr = -0.05
        if tr:
            p += tr
            notes.append(f"trainer Cup record {trainer_record.get('wins', 0)}w/{trainer_record.get('placings', 0)}p {tr:+.2f}")
    if r.get("jockey_cup_wins", 0) > 0:
        p += 0.05
        notes.append("Cup-winning jockey +0.05")
    if r["weight"] > 55.5:
        adj = -0.08 * (r["weight"] - 55.5)
        p += adj
        notes.append(f"weight {r['weight']}kg {adj:+.2f}")
    if r["barrier"] >= 20:
        p -= 0.35
        notes.append(f"barrier {r['barrier']} -0.35 (no winner from 20+ since 2015)")
    elif r["barrier"] >= 15:
        p -= 0.10
        notes.append(f"barrier {r['barrier']} -0.10")
    age_adj = {4: +0.10, 5: +0.05, 6: 0.0, 7: -0.10, 8: -0.35, 9: -0.60}.get(r["age"], -0.6)
    if age_adj:
        p += age_adj
        notes.append(f"age {r['age']} {age_adj:+.2f}")
    if r["sex"] == "M":
        p -= 0.20
        notes.append("mare -0.20")
    if last_tempo == "SLOW":
        p -= w.slow_leadup_penalty
        notes.append(f"last lead-up slowly run -{w.slow_leadup_penalty:.2f}")
    return p, notes


def predict(weights: Weights | None = None, sims: int = 20000, seed: int = 7,
            field_file: str = "field_2026.json", history_file: str = "historical_cups.json",
            archetypes_file: str = "prep_archetypes.json") -> Prediction:
    w = weights or Weights()
    field_doc = load_json(field_file)
    history = load_json(history_file)["records"]
    idx = ArchetypeIndex(load_json(archetypes_file))
    runners = field_doc["runners"]
    for r in runners:
        r["leadup_group"] = derive_leadup_group(r)

    odds = np.array([r["odds"] for r in runners], dtype=float)
    market = (1.0 / odds)
    market = market / market.sum()

    form = {r["horse"]: form_rating(r) for r in runners}
    form_vals = np.array([form[r["horse"]]["form_rating"] for r in runners])

    pattern = pattern_scores(runners, history, idx)
    pattern_vals = np.array([pattern[r["horse"]]["pattern_total"] for r in runners])

    form_z = _z(form_vals)
    pattern_z = _z(np.log(pattern_vals + 1e-9))

    track_dist = PRESETS.get(w.track) or field_doc["race"].get("track_distribution", {"GOOD": 1.0})
    pressure, pressure_note = pace_pressure(runners)
    style_prior = style_history_prior(history, runners)

    components = []
    logits = np.zeros(len(runners))
    for i, r in enumerate(runners):
        prior, notes = structural_priors(r, field_doc["race"], w, form[r["horse"]]["last_leadup_tempo"], idx.trainer_record(r["trainer"]))
        g_adj, g_note = going_adjustment(r, track_dist, scale=w.going_scale)
        p_adj, p_notes = pace_adjustment(r, pressure, w.bias, w.bias_strength, style_prior)
        logit = w.market * math.log(market[i]) + w.form * form_z[i] + w.pattern * pattern_z[i] + prior + g_adj + p_adj
        logits[i] = logit
        components.append({
            "horse": r["horse"], "market_prob": float(market[i]), "form_rating": float(form_vals[i]),
            "form_z": float(form_z[i]), "pattern_total": float(pattern_vals[i]), "pattern_z": float(pattern_z[i]),
            "prior": prior, "prior_notes": notes, "going_adj": g_adj, "going_note": g_note,
            "pace_adj": p_adj, "pace_notes": p_notes, "logit": float(logit),
        })
    win = np.exp(logits - logits.max())
    win = win / win.sum()

    names = [r["horse"] for r in runners]
    place, top4, _, _ = simulate_first_four(win, names, sims=sims, seed=seed)
    exact, boxed = analytic_first_four(win, names)
    return Prediction(race=field_doc["race"], runners=runners, win_prob=win, place_prob=place, top4_prob=top4,
                      components=components, pattern=pattern, form=form, first_four_exact=exact,
                      first_four_boxed=boxed, weights=w, sims=sims, speed_map=speed_map(runners),
                      pace=(pressure, pressure_note), track_distribution=track_dist, idx=idx)


def analytic_first_four(win: np.ndarray, names: list[str], temper: float = 0.9, top_n: int = 9, keep: int = 10):
    """Exact Plackett-Luce probability of every ordered first four drawn from the `top_n` strongest runners,
    P(a,b,c,d) = s_a/S * s_b/(S-s_a) * s_c/(S-s_a-s_b) * s_d/(S-s_a-s_b-s_c); boxed = sum over the 24 orders."""
    from itertools import permutations
    s = win ** temper
    S = s.sum()
    cand = list(np.argsort(-win)[:top_n])
    exact = {}
    for combo in permutations(cand, 4):
        prob, remaining = 1.0, S
        for i in combo:
            prob *= s[i] / remaining
            remaining -= s[i]
        exact[tuple(names[i] for i in combo)] = prob
    boxed = Counter()
    for combo, prob in exact.items():
        boxed[frozenset(combo)] += prob
    exact_list = sorted(exact.items(), key=lambda kv: -kv[1])[:keep]
    boxed_list = boxed.most_common(keep)
    return exact_list, boxed_list


def simulate_first_four(win: np.ndarray, names: list[str], sims: int = 20000, seed: int = 7, temper: float = 0.9):
    """Plackett-Luce sampling via the Gumbel-max trick. `temper` < 1 flattens strengths for minor placings
    (a 3200m handicap finish behind the winner is noisier than the market/model ordering implies)."""
    rng = np.random.default_rng(seed)
    strengths = np.log(win) * temper
    n = len(win)
    place_count = np.zeros(n)
    top4_count = np.zeros(n)
    exact = Counter()
    boxed = Counter()
    for _ in range(sims):
        g = rng.gumbel(size=n)
        order = np.argsort(-(strengths + g))
        first4 = order[:4]
        place_count[order[:3]] += 1
        top4_count[first4] += 1
        exact[tuple(names[i] for i in first4)] += 1
        boxed[frozenset(names[i] for i in first4)] += 1
    place = place_count / sims
    top4 = top4_count / sims
    exact_list = [(k, v / sims) for k, v in exact.most_common(10)]
    boxed_list = [(k, v / sims) for k, v in boxed.most_common(10)]
    return place, top4, exact_list, boxed_list


SCENARIOS = {
    "forecast / neutral": {},
    "Good 4": {"track": "good4"},
    "Soft 6": {"track": "soft6"},
    "Soft 7+": {"track": "soft7"},
    "on-pace bias": {"bias": "on-pace"},
    "closers bias": {"bias": "closers"},
}


def scenario_table(base: Weights | None = None, sims: int = 4000, top_n: int = 10) -> dict:
    """Win probabilities for the strongest runners under each going / bias scenario."""
    base = base or Weights()
    import dataclasses
    results = {}
    for name, override in SCENARIOS.items():
        w = dataclasses.replace(base, **override)
        pred = predict(weights=w, sims=sims)
        results[name] = {r["horse"]: float(pred.win_prob[i]) for i, r in enumerate(pred.runners)}
    ref = results["forecast / neutral"]
    horses = sorted(ref, key=lambda h: -ref[h])[:top_n]
    return {"horses": horses, "scenarios": results}
