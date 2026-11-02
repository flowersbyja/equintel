"""
Tempo-adjusted lead-up form.

A finishing position only means something relative to how the race was run. The rules here are
deliberately simple and transparent (every adjustment is logged so the report can explain it):

  * base merit from finishing position, beaten margin, field size and race class
  * tempo x running-style interaction: a backmarker beaten in a SLOW race is upgraded (the shape did
    not suit), an on-pace horse that faded in a FAST race is upgraded, a backmarker that won off a
    SLOW tempo is discounted (the shape handed it the race), an on-pacer that won a SLOW race is discounted
  * recognised excuses add back merit (first-up, wide, heavy track, interference, dropped whip, ...)
  * a short-distance sharpener (<= 2100m) is heavily down-weighted - it is a fitness run, not a form line
  * the most recent run counts most (exponential decay like racing-predictor's averageMarketResidual)
"""
from __future__ import annotations

CLASS_WEIGHT = {"G1": 1.00, "G2": 0.85, "G3": 0.72, "HCP": 0.70, "LR": 0.60, "OTHER": 0.50}

EXCUSE_CREDIT = {
    "FIRST_UP": 0.35,
    "WIDE": 0.35,
    "HEAVY_TRACK": 0.50,
    "INTERFERENCE": 0.45,
    "DROPPED_WHIP": 0.20,
    "OVERRACED": 0.25,
    "BIG_WEIGHT": 0.30,
    "WIDE_BARRIER": 0.20,
    "EASED": 0.40,
    "FELL": 0.60,
    "SHORT_SHARPENER": 0.0,  # handled by distance down-weighting instead
}


def base_merit(finish: int, field: int, margin_len: float, race_class: str) -> float:
    """Merit on roughly a -3..+3 scale. Winning a G1 ~ +3, last in a G3 beaten 10L ~ -2.5."""
    pos_term = 1.0 - 2.0 * (finish - 1) / max(field - 1, 1)  # +1 winner ... -1 last
    margin_term = -min(margin_len, 12.0) / 6.0  # 6L beaten = -1
    cls = CLASS_WEIGHT.get(race_class, CLASS_WEIGHT["OTHER"])
    top_bonus = 0.6 if finish <= 3 else 0.0
    return (1.6 * pos_term + margin_term + top_bonus) * cls


def tempo_adjustment(style: str, tempo: str, finish: int, field: int) -> tuple[float, str]:
    """How much the race shape helped (negative) or hurt (positive) this horse."""
    won_or_placed = finish <= 3
    well_beaten = finish > max(3, field // 2)
    if tempo == "SLOW":
        if style == "BACKMARKER":
            if won_or_placed:
                return -0.45, "closed off a slow tempo - race shape handed it the finish, discounted"
            return +0.55, "backmarker in a slowly run race - tempo did not suit, upgraded"
        if style in ("LEADER", "ON_PACE"):
            if won_or_placed:
                return -0.35, "on the speed in a slow race - soft lead, discounted"
            return -0.25, "on the speed in a slow race and still beaten - no excuse"
        if style == "MIDFIELD":
            return (+0.20 if well_beaten else -0.10), "midfield in a slow race"
    if tempo == "FAST":
        if style in ("LEADER", "ON_PACE"):
            if won_or_placed:
                return +0.40, "held on through a fast tempo - strong run, upgraded"
            return +0.45, "on the speed in a genuinely run race and faded - tempo excuse"
        if style == "BACKMARKER" and won_or_placed:
            return -0.15, "closer in a fast race - shape suited"
        return 0.0, "fast tempo"
    if tempo == "MODERATE":
        if style in ("LEADER", "ON_PACE") and won_or_placed:
            return +0.15, "on-pace and ran it out at an honest tempo"
        return 0.0, "honest tempo"
    return 0.0, "tempo unknown"


def leadup_merit(runner_style: str, leadup: dict) -> dict:
    merit = base_merit(leadup["finish"], leadup["field"], leadup["margin"], leadup.get("class", "OTHER"))
    tempo_adj, tempo_note = tempo_adjustment(runner_style, leadup.get("tempo", "UNKNOWN"), leadup["finish"], leadup["field"])
    excuses = leadup.get("excuses", [])
    excuse_credit = sum(EXCUSE_CREDIT.get(e, 0.0) for e in excuses)
    excuse_credit = min(excuse_credit, 0.8)
    # A sharpener over < 2100m tells us about fitness, not 3200m merit.
    dist = leadup.get("dist", 2400)
    relevance = 0.35 if dist <= 2100 else (0.7 if dist < 2300 else 1.0)
    adjusted = merit + tempo_adj + excuse_credit
    return {
        "race": leadup["race"],
        "finish": leadup["finish"],
        "field": leadup["field"],
        "margin": leadup["margin"],
        "tempo": leadup.get("tempo", "UNKNOWN"),
        "base": round(merit, 2),
        "tempo_adj": round(tempo_adj, 2),
        "tempo_note": tempo_note,
        "excuses": excuses,
        "excuse_credit": round(excuse_credit, 2),
        "relevance": relevance,
        "adjusted": round(adjusted, 2),
    }


def form_rating(runner: dict, max_runs: int = 3, decay: float = 0.55) -> dict:
    """Recency-weighted, tempo-adjusted form rating from the runner's lead-ups (most recent first)."""
    details = []
    num = 0.0
    den = 0.0
    for i, lu in enumerate(runner["leadups"][:max_runs]):
        d = leadup_merit(runner["style"], lu)
        w = (decay ** i) * d["relevance"]
        num += w * d["adjusted"]
        den += w
        d["weight"] = round(w, 3)
        details.append(d)
    rating = num / den if den > 0 else 0.0
    # Did the horse's most recent run come through a genuinely run race? (Racelab: none of the last
    # ten winners came out of a slowly run lead-up.)
    last_tempo = runner["leadups"][0].get("tempo", "UNKNOWN") if runner["leadups"] else "UNKNOWN"
    return {"form_rating": round(rating, 3), "last_leadup_tempo": last_tempo, "runs": details}
