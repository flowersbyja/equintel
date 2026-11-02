"""Printable 'best bets' form guide (HTML, A4 print styles, colour-coded tiers)."""
from __future__ import annotations

import datetime as dt
import html

import numpy as np

from .going import going_fit
from .model import Prediction

STYLE_LABEL = {"LEADER": "Leader", "ON_PACE": "On pace", "MIDFIELD": "Midfield", "BACKMARKER": "Swooper"}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _stars(win: float) -> str:
    n = 5 if win >= 0.12 else 4 if win >= 0.08 else 3 if win >= 0.05 else 2 if win >= 0.025 else 1
    return "★" * n + "☆" * (5 - n)


def _tier(rank: int) -> str:
    return "t1" if rank <= 4 else "t2" if rank <= 8 else "t3" if rank <= 14 else "t4"


def _going_word(fit: float) -> str:
    return "Loves wet" if fit >= 0.7 else "Handles wet" if fit > 0.1 else "Wants firm" if fit <= -0.7 else "Prefers dry" if fit < -0.1 else "Unknown"


def _done_it(pred: Prediction, r: dict) -> str:
    tr = pred.pattern[r["horse"]]["type_record"]
    arch = pred.pattern[r["horse"]]["archetype"] or "OTHER"
    own = pred.idx.trainer_type_record(r["trainer"], arch)
    if own and own["runners"] > 0:
        return f"Yes: {own['runners']} runs, {own['wins']}W {own['placings']}P ({tr['term']:+.2f})"
    return f"First time ({tr['term']:+.2f})"


def build_printable(pred: Prediction, scenarios: dict | None = None) -> str:
    rs = pred.runners
    order = list(np.argsort(-pred.win_prob))
    race = pred.race
    e = html.escape

    def runner(i):
        return rs[i]

    top = order[0]
    edges = sorted(range(len(rs)), key=lambda i: -(pred.win_prob[i] * rs[i]["odds"] - 1))
    value = next(i for i in edges if i != top and rs[i]["odds"] >= 10)
    each_way = next(i for i in sorted(range(len(rs)), key=lambda i: -pred.top4_prob[i]) if i not in (top, value) and rs[i]["odds"] >= 8)
    roughie = next(i for i in sorted(range(len(rs)), key=lambda i: -pred.top4_prob[i]) if rs[i]["odds"] >= 20 and i not in (top, value, each_way))
    first_four = [rs[i]["no"] for i in order[:4]]
    wide_box = [rs[i]["no"] for i in order[:6]]

    def bet_card(kind, i, cls):
        r = rs[i]
        edge = pred.win_prob[i] * r["odds"] - 1
        return f"""
        <article class="bet {cls}">
          <div class="bet-kind">{kind}</div>
          <div class="bet-horse"><span class="cloth">{r['no']}</span> {e(r['horse'])}</div>
          <div class="bet-meta">Bar {r['barrier']} · {r['weight']}kg · {e(r['jockey'])} · {e(r['trainer'])}</div>
          <div class="bet-nums"><span><b>{_pct(pred.win_prob[i])}</b> win</span><span><b>{_pct(pred.top4_prob[i])}</b> top-4</span><span><b>${r['odds']:.1f}</b> market</span><span><b>{100 * edge:+.0f}%</b> edge</span></div>
          <p class="bet-why">{e(r['verdict'])}</p>
        </article>"""

    sm = pred.speed_map
    no_of = {r["horse"]: r["no"] for r in rs}

    def chips(names, cls):
        return "".join(f'<span class="chip {cls}">{no_of[n]}<small>{e(n)}</small></span>' for n in names)

    rows = []
    for rank, i in enumerate(order, 1):
        r = rs[i]
        c = pred.components[i]
        lu = r["leadups"][0]
        gf = going_fit(r)["fit"]
        edge = pred.win_prob[i] * r["odds"] - 1
        best = pred.pattern[r["horse"]]["best_matches"][0]
        rows.append(f"""
        <tr class="{_tier(rank)}">
          <td class="num">{rank}</td>
          <td class="cloth-cell"><span class="cloth">{r['no']}</span></td>
          <td class="horse"><b>{e(r['horse'])}</b><small>{e(r['jockey'])} · {e(r['trainer'])}</small></td>
          <td class="num">{r['barrier']}</td>
          <td class="num">{r['weight']}</td>
          <td>{STYLE_LABEL[r['style']]}<small>settles ~{r.get('settle', '?')}</small></td>
          <td>{e(lu['race'].replace(' G1', '').replace(' G2', '').replace(' G3', ''))} {lu['finish']}/{lu['field']}<small>{lu['tempo'].lower()} tempo · {r['days_since']}d</small></td>
          <td class="going g{'p' if gf > 0.1 else 'n' if gf < -0.1 else 'z'}">{_going_word(gf)}</td>
          <td class="small">{e(best['horse'])} {best['year']}<small>credit x{best['trainer_multiplier']:.2f}{' · own' if best['same_trainer'] else ''}</small></td>
          <td class="small done">{e(_done_it(pred, r))}</td>
          <td class="num model"><b>{_pct(pred.win_prob[i])}</b><small>top-4 {_pct(pred.top4_prob[i])}</small></td>
          <td class="num">${r['odds']:.1f}<small class="{'pos' if edge > 0.1 else 'neg' if edge < -0.3 else ''}">{100 * edge:+.0f}%</small></td>
          <td class="stars">{_stars(pred.win_prob[i])}</td>
          <td class="verdict">{e(r['verdict'])}</td>
        </tr>""")

    scen_html = ""
    if scenarios:
        names = list(scenarios["scenarios"].keys())
        head = "".join(f"<th>{e(n)}</th>" for n in names)
        body = ""
        for h in scenarios["horses"][:8]:
            cells = "".join(f"<td>{_pct(scenarios['scenarios'][n][h])}</td>" for n in names)
            body += f"<tr><td class='horse'><b>{e(h)}</b></td>{cells}</tr>"
        scen_html = f"""
      <section class="panel">
        <h2>If the day changes</h2>
        <p class="lede">Win chance for the top eight under each going and track-bias reading. Watch the first three or four races: if on-speed horses keep winning, use the on-pace column; if the swoopers are getting over the top, the closers column.</p>
        <div class="scroll"><table class="scen"><thead><tr><th>Horse</th>{head}</tr></thead><tbody>{body}</tbody></table></div>
      </section>"""

    generated = dt.datetime.now().strftime("%d %b %Y %H:%M")
    track = ", ".join(f"{k.replace('_PLUS', '+').replace('SOFT', 'Soft ').replace('GOOD', 'Good')} {int(100 * v)}%" for k, v in pred.track_distribution.items())

    return f"""<title>Cup Day Best Bets</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&display=swap">
<style>
/* Layout: a racebook page - dark green masthead, white printable body, tier-shaded field grid, A4 print. */
:root {{
  --bg: #fbfaf6; --fg: #1a1f1b; --muted: #5b655c; --rule: #d9ddd5; --panel: #ffffff;
  --green: #0f4d3a; --green-ink: #ffffff; --gold: #c99a2e; --gold-soft: #fff3cf;
  --t1: #e3f1e6; --t2: #f1f6ec; --t3: #f8f8f4; --t4: #ffffff;
  --ew: #e6f0f7; --val: #efe9f7; --rough: #fdebd9; --pos: #157347; --neg: #a23b2a;
  --display: "Barlow Condensed", "Arial Narrow", Impact, sans-serif;
  --body: "Source Sans 3", "Segoe UI", Helvetica, Arial, sans-serif;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #121715; --fg: #e8ece8; --muted: #a2ada4; --rule: #2b332d; --panel: #1a211d; --green: #0b3b2d; --gold-soft: #3a2f12;
  --t1: #1d3a2a; --t2: #1b2d22; --t3: #1a221d; --t4: #171d19; --ew: #16303d; --val: #2a2238; --rough: #3b2616; --pos: #4fcf8a; --neg: #ff8b73; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #121715; --fg: #e8ece8; --muted: #a2ada4; --rule: #2b332d; --panel: #1a211d; --green: #0b3b2d; --gold-soft: #3a2f12;
  --t1: #1d3a2a; --t2: #1b2d22; --t3: #1a221d; --t4: #171d19; --ew: #16303d; --val: #2a2238; --rough: #3b2616; --pos: #4fcf8a; --neg: #ff8b73; color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 13px; line-height: 1.35; margin: 0; }}
.page {{ max-width: 1180px; margin: 0 auto; padding-block: 0 24px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
header.mast {{ background: var(--green); color: var(--green-ink); margin: 0 -16px; padding: 18px max(16px, env(safe-area-inset-left, 0px)) 16px max(16px, env(safe-area-inset-right, 0px)); border-bottom: 5px solid var(--gold); }}
.mast-inner {{ max-width: 1148px; margin: 0 auto; display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px 24px; align-items: flex-end; }}
.mast h1 {{ font-family: var(--display); font-weight: 800; font-size: clamp(30px, 5vw, 46px); letter-spacing: .01em; margin: 0; line-height: .95; text-wrap: balance; }}
.mast .eyebrow {{ font-family: var(--display); font-weight: 600; letter-spacing: .14em; text-transform: uppercase; font-size: 12px; opacity: .85; }}
.mast .facts {{ display: grid; grid-template-columns: auto auto; gap: 2px 14px; font-size: 12.5px; }}
.mast .facts b {{ font-weight: 700; }}
h2 {{ font-family: var(--display); font-weight: 700; font-size: 22px; letter-spacing: .02em; margin: 0 0 6px; text-transform: uppercase; }}
.lede {{ color: var(--muted); margin: 0 0 10px; max-width: 70ch; }}
section.panel {{ margin-top: 18px; }}
.bets {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }}
.bet {{ background: var(--panel); border: 1px solid var(--rule); border-top: 6px solid var(--gold); padding: 12px 14px; break-inside: avoid; }}
.bet.best {{ background: var(--gold-soft); border-top-color: var(--gold); }}
.bet.ew {{ border-top-color: #2d6f9c; background: var(--ew); }}
.bet.val {{ border-top-color: #6a4fa3; background: var(--val); }}
.bet.rough {{ border-top-color: #d0742a; background: var(--rough); }}
.bet-kind {{ font-family: var(--display); text-transform: uppercase; letter-spacing: .12em; font-size: 12px; font-weight: 700; color: var(--muted); }}
.bet-horse {{ font-family: var(--display); font-size: 26px; font-weight: 800; margin: 2px 0 2px; display: flex; align-items: center; gap: 8px; }}
.bet-meta {{ color: var(--muted); font-size: 12px; }}
.bet-nums {{ display: flex; flex-wrap: wrap; gap: 6px 14px; margin: 8px 0 6px; font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }}
.bet-nums b {{ color: var(--fg); font-size: 15px; }}
.bet-why {{ margin: 0; font-size: 13px; }}
.cloth {{ display: inline-grid; place-items: center; min-width: 26px; height: 26px; padding: 0 5px; border-radius: 4px; background: var(--green); color: #fff; font-family: var(--display); font-weight: 800; font-size: 16px; }}
.exotics {{ display: flex; flex-wrap: wrap; gap: 10px 26px; margin-top: 12px; font-size: 13.5px; }}
.exotics b {{ font-family: var(--display); font-size: 18px; letter-spacing: .04em; }}
.map {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }}
.lane {{ background: var(--panel); border: 1px solid var(--rule); padding: 8px 10px; }}
.lane h3 {{ margin: 0 0 6px; font-family: var(--display); font-size: 15px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); }}
.chip {{ display: inline-flex; align-items: center; gap: 5px; margin: 2px 4px 2px 0; padding: 2px 7px 2px 3px; border-radius: 14px; border: 1px solid var(--rule); font-family: var(--display); font-weight: 700; font-size: 14px; }}
.chip small {{ font-family: var(--body); font-weight: 400; font-size: 11.5px; color: var(--muted); }}
.chip.c1 {{ background: var(--rough); }} .chip.c2 {{ background: var(--gold-soft); }} .chip.c3 {{ background: var(--t2); }} .chip.c4 {{ background: var(--ew); }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; background: var(--panel); }}
th {{ font-family: var(--display); text-transform: uppercase; letter-spacing: .08em; font-size: 11.5px; text-align: left; color: var(--muted); padding: 6px 7px; border-bottom: 2px solid var(--green); white-space: nowrap; }}
td {{ padding: 6px 7px; border-bottom: 1px solid var(--rule); vertical-align: top; }}
td small, .horse small {{ display: block; color: var(--muted); font-size: 11px; line-height: 1.25; }}
td.num {{ text-align: right; white-space: nowrap; }}
td.model b {{ font-size: 14px; }}
tr.t1 td {{ background: var(--t1); }} tr.t2 td {{ background: var(--t2); }} tr.t3 td {{ background: var(--t3); }} tr.t4 td {{ background: var(--t4); color: var(--muted); }}
td.going.gp {{ color: var(--pos); font-weight: 600; }} td.going.gn {{ color: var(--neg); font-weight: 600; }}
small.pos {{ color: var(--pos); font-weight: 600; }} small.neg {{ color: var(--neg); }}
td.stars {{ color: var(--gold); letter-spacing: .05em; white-space: nowrap; font-size: 13px; }}
td.verdict {{ min-width: 220px; font-size: 12px; }}
td.done {{ white-space: nowrap; font-size: 11.5px; }}
.scen td, .scen th {{ text-align: right; }} .scen td.horse, .scen th:first-child {{ text-align: left; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 11.5px; color: var(--muted); margin-top: 8px; }}
.legend i {{ display: inline-block; width: 14px; height: 12px; vertical-align: -2px; border: 1px solid var(--rule); margin-right: 4px; }}
footer {{ margin-top: 18px; color: var(--muted); font-size: 11px; border-top: 1px solid var(--rule); padding-top: 8px; }}
@page {{ size: A4 landscape; margin: 9mm; }}
@media print {{
  :root {{ --bg: #ffffff; }}
  body {{ font-size: 10.5px; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  .page {{ max-width: none; padding: 0; }}
  header.mast {{ margin: 0; padding: 10px 12px; }}
  .mast h1 {{ font-size: 30px; }}
  .bets {{ grid-template-columns: repeat(4, 1fr); }}
  .bet-horse {{ font-size: 20px; }}
  section.panel {{ margin-top: 10px; }}
  .field {{ break-before: page; }}
  tr {{ break-inside: avoid; }}
  td, th {{ padding: 3px 5px; }}
  td.verdict {{ min-width: 0; }}
  .scroll {{ overflow: visible; }}
}}
</style>
<div class="page">
<header class="mast"><div class="mast-inner">
  <div><div class="eyebrow">Flemington · Race 7 · 3:00pm AEDT · Tuesday 3 November 2026</div><h1>Lexus Melbourne Cup<br>best bets &amp; form</h1></div>
  <div class="facts">
    <span>Distance</span><b>3200m, Group 1 handicap, 24 runners</b>
    <span>Track</span><b>{e(race['expected_track'])}</b>
    <span>Going model</span><b>{e(track)}</b>
    <span>Pace</span><b>{e(pred.pace[0])} tempo expected</b>
  </div>
</div></header>

<section class="panel">
  <h2>Best bets</h2>
  <p class="lede">Model probabilities combine the market, tempo-adjusted lead-up form, how closely each preparation (and the trainer's method) matches past first-four finishers, going, race shape, weight, barrier and age. Edge = model chance x market odds, minus one.</p>
  <div class="bets">
    {bet_card("Best bet", top, "best")}
    {bet_card("Each-way", each_way, "ew")}
    {bet_card("Best value", value, "val")}
    {bet_card("Roughie", roughie, "rough")}
  </div>
  <div class="exotics">
    <span>First four box: <b>{' · '.join(map(str, first_four))}</b></span>
    <span>Wider first four / trifecta box: <b>{' · '.join(map(str, wide_box))}</b></span>
    <span>Quinella: <b>{first_four[0]} / {first_four[1]}</b></span>
  </div>
</section>

<section class="panel">
  <h2>Speed map</h2>
  <p class="lede">{e(race['pace_outlook'])}</p>
  <div class="map">
    <div class="lane"><h3>Leaders</h3>{chips(sm['leaders'], 'c1')}</div>
    <div class="lane"><h3>On pace</h3>{chips(sm['on_pace'], 'c2')}</div>
    <div class="lane"><h3>Midfield</h3>{chips(sm['midfield'], 'c3')}</div>
    <div class="lane"><h3>Back / swoopers</h3>{chips(sm['back'], 'c4')}</div>
  </div>
</section>
{scen_html}
<section class="panel field">
  <h2>The field, ranked by the model</h2>
  <div class="scroll">
  <table>
    <thead><tr><th>#</th><th>No</th><th>Horse / rider / trainer</th><th>Bar</th><th>Kg</th><th>Style</th><th>Last start</th><th>Going</th><th>Modelled on</th><th>Trainer done it?</th><th>Model</th><th>Odds / edge</th><th>Rating</th><th>Verdict</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  </div>
  <div class="legend">
    <span><i style="background:var(--t1)"></i>Top four</span><span><i style="background:var(--t2)"></i>Next four</span><span><i style="background:var(--t3)"></i>Place hopes</span><span><i style="background:var(--t4)"></i>Outsiders</span>
    <span>Rating: ★★★★★ 12%+ win chance, ★★★★ 8%+, ★★★ 5%+, ★★ 2.5%+</span>
    <span>"Modelled on" = closest past first-four prep; credit is gated by how alike the two trainers' methods are (x0 = different school, x1 = same method).</span>
    <span>"Trainer done it?" = this trainer's own Cup record with this type of preparation (runners / wins / placings); 'first time' uses the method school's record.</span>
  </div>
</section>

<footer>Generated {generated} from the equintel melbourne_cup model (market weight {pred.weights.market}, {pred.sims:,} simulations). Prices are a 31 Oct - 2 Nov consensus; re-run on race morning with the final track rating and prices. Research model, not betting advice. Gamble responsibly.</footer>
</div>
"""
