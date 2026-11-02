"""Multi-variable form guide: one wide table plus a card per runner covering trainer method, campaign shape,
running style / speed map, going, gear, jockey and the model's view."""
from __future__ import annotations

import numpy as np

from .going import going_fit
from .model import Prediction


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _wet(rec: dict | None) -> str:
    if not rec:
        return "n/a"
    return f"{rec['starts']}: {rec['wins']}-{rec['seconds']}-{rec['thirds']}"


def _vec(v: dict) -> str:
    return " ".join(f"{k[:5]}={v[k]:.1f}" for k in v)


def build_form_guide(pred: Prediction) -> str:
    rs = pred.runners
    order = np.argsort(-pred.win_prob)
    race = pred.race
    L = []
    L.append(f"# {race['name']} - multi-variable form guide\n")
    L.append(f"*{race['course']} {race['distance_m']}m, {race['date']}. Going model: "
             + ", ".join(f"{k} {int(100 * v)}%" for k, v in pred.track_distribution.items())
             + f". Day bias setting: **{pred.weights.bias}**. Pace read: **{pred.pace[0]}** ({pred.pace[1]}).*\n")

    L.append("## Speed map (predicted settling order)\n")
    sm = pred.speed_map
    L.append(f"- **Leaders:** {', '.join(sm['leaders'])}")
    L.append(f"- **On pace:** {', '.join(sm['on_pace'])}")
    L.append(f"- **Midfield:** {', '.join(sm['midfield'])}")
    L.append(f"- **Back:** {', '.join(sm['back'])}\n")
    L.append("If the early Cup Day races are being won on the speed, re-run with `--bias on-pace`; if swoopers are dominating, `--bias closers`.\n")

    L.append("## The grid\n")
    L.append("| No | Horse | Bar | Wt | Age | Trainer (Cup W/P) | Jockey (Cup W) | Style | Settle | Last lead-up (pos, tempo) | Days | Aus runs | Prep formula | Trainer-method match | Wet rec | Going fit | Gear | Mkt | Model | Top-4 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i in order:
        r = rs[i]
        c = pred.components[i]
        p = pred.pattern[r["horse"]]
        tr = pred.idx.trainer_record(r["trainer"])
        best = p["best_matches"][0]
        lu = r["leadups"][0]
        gf = going_fit(r)["fit"]
        match = f"{best['horse']} {best['year']} (sim {best['similarity']:.2f}, method {best['trainer_style_similarity']:.2f}{', own' if best['same_trainer'] else ''})"
        L.append(f"| {r['no']} | **{r['horse']}** | {r['barrier']} | {r['weight']} | {r['age']}{r['sex']} | {r['trainer']} ({tr['wins']}/{tr['placings']}) | {r['jockey']} ({r.get('jockey_cup_wins', 0)}) | "
                 f"{r['style'].lower().replace('_', '-')} | {r.get('settle', '-')} | {lu['race']} {lu['finish']}/{lu['field']}, {lu['tempo'].lower()} | {r['days_since']} | {r['aus_prep_runs']} | "
                 f"{p['archetype']} | {match} | {_wet(r.get('wet_record'))} | {gf:+.2f} | {r.get('gear', '-')} | {_pct(c['market_prob'])} | **{_pct(pred.win_prob[i])}** | {_pct(pred.top4_prob[i])} |")
    L.append("")
    L.append("*Trainer-method match = how alike the matched horse's trainer method (and that campaign) is to this horse's trainer method and campaign, 0-1. 'own' = the same trainer's past Cup prep. Wet rec = starts: wins-2nds-3rds on rain-affected ground where published.*\n")

    L.append("## Runner cards\n")
    for i in order:
        r = rs[i]
        c = pred.components[i]
        p = pred.pattern[r["horse"]]
        tr = pred.idx.trainer_record(r["trainer"])
        L.append(f"### {r['no']}. {r['horse']} - model {_pct(pred.win_prob[i])}, top-4 {_pct(pred.top4_prob[i])}, market ${r['odds']:.1f}\n")
        L.append(f"**Trainer method ({r['trainer']}, Cup record {tr['runners']} runners / {tr['wins']} wins / {tr['placings']} placings):** {pred.idx.trainer_method(r['trainer'])}\n")
        L.append(f"**This campaign:** {r.get('campaign', '')} Departure from the trainer's normal method: {p['campaign_departure']:.2f} (0 = textbook, 1 = completely different).\n")
        L.append(f"**Running style / shape:** {r['style'].lower().replace('_', '-')}, predicted to settle about {r.get('settle', '?')}th. Pace adjustment {c['pace_adj']:+.2f} ({'; '.join(c['pace_notes']) or 'none'}).\n")
        L.append(f"**Going:** wet record {_wet(r.get('wet_record'))}, connections prior {r.get('wet_rating', 0):+.1f}. {c['going_note']}.\n")
        L.append(f"**Gear:** {r.get('gear', 'none reported')}. **Jockey:** {r['jockey']} ({r.get('jockey_cup_wins', 0)} Cup wins).\n")
        L.append(f"**Has this trainer done this before?** ({c['trainer_type_adj']:+.2f}) " + " ".join(c['trainer_type_notes']) + "\n")
        L.append("**Who it is modelled on (method-gated):**\n")
        for m in p["best_matches"]:
            own = f" - same trainer, {m['gate_note']} -> credit x{m['trainer_multiplier']:.2f}" if m["same_trainer"] else f" - different trainer, {m['gate_note']} -> credit x{m['trainer_multiplier']:.2f}"
            L.append(f"- {m['horse']} {m['year']} ({m['trainer']}, finished {m['pos']}, {m['archetype']}), horse-prep similarity {m['similarity']:.2f}{own}")
        L.append(f"\nMethod vectors (fresh, aus_runs, euro_base, wfa_sharpener, forward, young_light, ticket_target, data_driven): trainer {_vec(p['trainer_profile'])}; this campaign {_vec(p['campaign_style'])}.\n")
        L.append(f"**Model components:** market {_pct(c['market_prob'])}, form z {c['form_z']:+.2f}, pattern z {c['pattern_z']:+.2f}, trainer-done-it {c['trainer_type_adj']:+.2f}, structural {c['prior']:+.2f} ({'; '.join(c['prior_notes']) or 'none'}), going {c['going_adj']:+.2f}, pace {c['pace_adj']:+.2f}.\n")
        L.append(r["notes"] + "\n")
    return "\n".join(L) + "\n"
