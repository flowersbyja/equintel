"""Markdown report for a Prediction."""
from __future__ import annotations

import numpy as np

from .model import Prediction


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def build_report(pred: Prediction, scenarios: dict | None = None) -> str:
    rs = pred.runners
    order = np.argsort(-pred.win_prob)
    race = pred.race
    lines = []
    lines.append(f"# {race['name']} - model first four\n")
    lines.append(f"*{race['course']}, {race['distance_m']}m, {race['date']}. Expected track: {race['expected_track']}.*\n")
    lines.append(f"*Pace: {race['pace_outlook']} Model pace read: {pred.pace[0]} ({pred.pace[1]}). Day-bias setting: {pred.weights.bias}.*\n")
    lines.append("*Going model: " + ", ".join(f"{k} {int(100 * v)}%" for k, v in pred.track_distribution.items()) + ".*\n")
    lines.append(f"*Prices: {race['odds_snapshot']}. Model weights: market {pred.weights.market}, "
                 f"tempo-adjusted form {pred.weights.form}, prep-pattern similarity {pred.weights.pattern}; "
                 f"{pred.sims:,} Plackett-Luce simulations for the placings.*\n")

    top4 = [rs[i] for i in order[:4]]
    lines.append("## Selections\n")
    for k, r in enumerate(top4, 1):
        i = order[k - 1]
        lines.append(f"{k}. **{r['horse']}** (No.{r['no']}, barrier {r['barrier']}, {r['weight']}kg, {r['trainer']}, {r['jockey']}) "
                     f"- win {_pct(pred.win_prob[i])}, top-4 {_pct(pred.top4_prob[i])}, market ${r['odds']:.2f}")
    lines.append("")
    lines.append("**Most likely exact first four (model order):** " +
                 " / ".join(f"{' - '.join(c)} ({_pct(p)})" for c, p in pred.first_four_exact[:3]))
    lines.append("")
    lines.append("**Most likely boxed first four:** " +
                 " / ".join(f"{{{', '.join(sorted(c))}}} ({_pct(p)})" for c, p in pred.first_four_boxed[:3]))
    lines.append("")

    if scenarios:
        lines.append("## What changes with the going and the day's bias\n")
        names = list(scenarios["scenarios"].keys())
        lines.append("| Horse | " + " | ".join(names) + " |")
        lines.append("|---|" + "---|" * len(names))
        for h in scenarios["horses"]:
            lines.append(f"| {h} | " + " | ".join(_pct(scenarios["scenarios"][n][h]) for n in names) + " |")
        lines.append("")
        lines.append("*Re-run on the day: `--track soft6` (or good4 / soft7) once the rating is posted, and `--bias on-pace` or `--bias closers` after watching how the early races at Flemington are being won.*\n")

    lines.append("## Full field\n")
    lines.append("| Rank | No | Horse | Bar | Wt | Odds | Market | Model win | Place | Top-4 | Win edge | Form (tempo adj) | Pattern z | Trainer done it? | Prior | Going | Pace |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for rank, i in enumerate(order, 1):
        r = rs[i]
        c = pred.components[i]
        edge = pred.win_prob[i] * r["odds"] - 1
        lines.append(f"| {rank} | {r['no']} | {r['horse']} | {r['barrier']} | {r['weight']} | ${r['odds']:.1f} | {_pct(c['market_prob'])} | "
                     f"**{_pct(pred.win_prob[i])}** | {_pct(pred.place_prob[i])} | {_pct(pred.top4_prob[i])} | {100 * edge:+.0f}% | "
                     f"{c['form_rating']:+.2f} | {c['pattern_z']:+.2f} | {c['trainer_type_adj']:+.2f} | {c['prior']:+.2f} | {c['going_adj']:+.2f} | {c['pace_adj']:+.2f} |")
    lines.append("")
    lines.append("*Win edge = model probability x fixed odds - 1 (equintel value definition). Positive means the market is longer than the model.*\n")

    lines.append("## Value against the market\n")
    edges = [(pred.win_prob[i] * rs[i]["odds"] - 1, i) for i in range(len(rs))]
    edges.sort(reverse=True)
    for e, i in edges[:6]:
        if e <= 0:
            break
        r = rs[i]
        lines.append(f"- **{r['horse']}** ${r['odds']:.1f}: model {_pct(pred.win_prob[i])} vs market {_pct(pred.components[i]['market_prob'])} -> edge {100 * e:+.0f}%, "
                     f"top-4 {_pct(pred.top4_prob[i])}")
    lines.append("")

    lines.append("## Runner by runner: tempo, excuses and preparation pattern\n")
    for i in order:
        r = rs[i]
        c = pred.components[i]
        f = pred.form[r["horse"]]
        p = pred.pattern[r["horse"]]
        lines.append(f"### {r['no']}. {r['horse']} - model {_pct(pred.win_prob[i])} (market {_pct(c['market_prob'])}), top-4 {_pct(pred.top4_prob[i])}\n")
        lines.append(f"{r['trainer']} / {r['jockey']}, barrier {r['barrier']}, {r['weight']}kg, {r['age']}yo {r['sex']}, {r['country']}, style {r['style'].lower().replace('_', '-')}, "
                     f"prep archetype **{p['archetype'] or 'n/a'}** ({r['leadup_group']}, {r['days_since']} days since last run, {r['aus_prep_runs']} Australian prep runs).\n")
        lines.append(r["notes"] + "\n")
        lines.append("Lead-ups (most recent first), tempo-adjusted:\n")
        for run in f["runs"]:
            exc = f", excuses: {', '.join(run['excuses']).lower().replace('_', ' ')} (+{run['excuse_credit']})" if run["excuses"] else ""
            lines.append(f"- {run['race']}: {run['finish']}/{run['field']} beaten {run['margin']}L, tempo {run['tempo'].lower()} -> base {run['base']:+.2f}, "
                         f"tempo {run['tempo_adj']:+.2f} ({run['tempo_note']}){exc} => {run['adjusted']:+.2f} (weight {run['weight']})")
        lines.append(f"\nForm rating {c['form_rating']:+.2f} (z {c['form_z']:+.2f}). Structural prior {c['prior']:+.2f}: {'; '.join(c['prior_notes']) or 'none'}. "
                     f"Going {c['going_adj']:+.2f} ({c['going_note']}). Pace/shape {c['pace_adj']:+.2f} ({'; '.join(c['pace_notes']) or 'none'}).\n")
        lines.append(f"Trainer method: {pred.idx.trainer_method(r['trainer'])} This campaign: {r.get('campaign', '')} (departure from the trainer's norm {p['campaign_departure']:.2f}).\n")
        lines.append(f"**Has this trainer done this before?** ({c['trainer_type_adj']:+.2f}) " + " ".join(c['trainer_type_notes']) + "\n")
        lines.append(f"Closest historical preps (pattern z {c['pattern_z']:+.2f}; {_pct(p['own_formula_share'])} of the pattern score comes from the trainer's own past Cup preps, "
                     f"{_pct(p['winner_template_share'])} from winners rather than placegetters):\n")
        for m in p["best_matches"]:
            feats = ", ".join(f"{k} {v:.2f}" for k, v in m["top_feature_matches"])
            own = f" **(same trainer's formula; {m['gate_note']}, credit x{m['trainer_multiplier']:.2f})**" if m["same_trainer"] else f" (different trainer; {m['gate_note']}, credit x{m['trainer_multiplier']:.2f})"
            lines.append(f"- {m['horse']} {m['year']} ({'won' if m['pos'] == 1 else str(m['pos']) + 'th' if m['pos'] > 3 else ['', '', '2nd', '3rd'][m['pos']]}, {m['trainer']}, {m['archetype']}) "
                         f"similarity {m['similarity']:.2f}{own} - strongest matches: {feats}")
        lines.append("")

    lines.append("## Trainer formulas in this field\n")
    by_trainer: dict[str, list[int]] = {}
    for i, r in enumerate(rs):
        by_trainer.setdefault(r["trainer"], []).append(i)
    for trainer, idxs in sorted(by_trainer.items(), key=lambda kv: -len(kv[1])):
        if len(idxs) < 2:
            continue
        lines.append(f"**{trainer}** ({len(idxs)} runners) - {pred.idx.trainer_method(trainer)}\n")
        for i in sorted(idxs, key=lambda j: -pred.win_prob[j]):
            r = rs[i]
            p = pred.pattern[r["horse"]]
            own = [m for m in p["best_matches"] if m["same_trainer"]]
            own_txt = f"own formula match: {own[0]['horse']} {own[0]['year']} (sim {own[0]['similarity']:.2f})" if own else "no close match to this trainer's past Cup preps"
            cross = [m for m in p["best_matches"] if not m["same_trainer"]]
            cross_txt = f"; closest other-trainer style: {cross[0]['horse']} {cross[0]['year']} ({cross[0]['trainer']}, horse sim {cross[0]['similarity']:.2f}, method sim {cross[0]['trainer_style_similarity']:.2f})" if cross else ""
            lines.append(f"- {r['horse']} ({p['archetype']}): {own_txt}{cross_txt}")
        lines.append("")

    lines.append("## How the trainer layer works now\n")
    lines.append("- A past Cup horse only counts as a template for a 2026 runner if the two trainers' methods (blended with how each horse was actually campaigned) are alike: zero credit below 0.62 similarity, ramping to full credit at 1.0, and halved again when the trainers belong to different method schools.")
    lines.append("- The same trainer's own past Cup horse only counts if this campaign was run the same way (Maher's Point Of Law gets no Gold Trip credit because the campaign shape is different).")
    lines.append("- Separately, each runner carries a 'has this trainer done this before' term from the trainer's own record with this preparation type; a trainer with no such record inherits the method school's base rate with a first-attempt penalty.")
    lines.append("- Trainer type records and school base rates are approximate counts compiled from published Cup results (2010-2025) and are listed in melbourne_cup/cup_data/trainer_styles.json for checking.\n")
    lines.append("## Caveats\n")
    lines.append("- Historical prep templates for 2000-2009 and for some 2nd-4th placegetters are reconstructed from form memory and down-weighted via a confidence field; treat named similarities as indicative.")
    lines.append("- Lead-up tempo ratings are categorical (slow / moderate / fast) and were set to be consistent with Racelab's published count that only 11 of the 24 runners came through a moderate-or-faster lead-up.")
    lines.append("- Prices are a consensus snapshot; re-run with updated odds and the final track rating on race morning.")
    lines.append("- This is a research model, not betting advice. Probabilities are calibrated loosely by shrinking the market, not by a fitted backtest, because there is one Cup a year.")
    return "\n".join(lines) + "\n"
