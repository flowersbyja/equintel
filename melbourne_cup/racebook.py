"""Racebook: Best Bets-style printed form guide. Complete field with drawn silks, barrier in brackets,
last four starts per runner, the race rundown, and colour-coded ratings. Printable A4 portrait."""
from __future__ import annotations

import datetime as dt
import html

import numpy as np

from .going import going_fit
from .model import Prediction

STYLE_LABEL = {"LEADER": "leader", "ON_PACE": "on pace", "MIDFIELD": "midfield", "BACKMARKER": "swooper"}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _stars(win: float) -> int:
    return 5 if win >= 0.12 else 4 if win >= 0.08 else 3 if win >= 0.05 else 2 if win >= 0.025 else 1


def _fig(pos: int, field: int) -> str:
    return str(pos) if pos <= 9 else "0"


def _checks(x, y, w, h, c1, c2, n=4):
    cw, ch = w / n, h / n
    out = []
    for i in range(n):
        for j in range(n):
            out.append(f'<rect x="{x + i * cw:.1f}" y="{y + j * ch:.1f}" width="{cw:.1f}" height="{ch:.1f}" fill="{c1 if (i + j) % 2 == 0 else c2}"/>')
    return "".join(out)


def silks_svg(s: dict) -> str:
    """Silks image. Uses the real silks picture (s['image'], a data URI cropped from the official sweep grid) when
    present; otherwise falls back to a drawn jacket from body / sleeves / cap / pattern."""
    if s.get("image"):
        return f'<img class="silks" src="{s["image"]}" width="44" height="52" alt="{html.escape(s.get("desc", ""))}" style="object-fit:contain">'
    body, sleeves, cap = s.get("body", "#9e9e9e"), s.get("sleeves", "#9e9e9e"), s.get("cap", "#9e9e9e")
    pat, pc = s.get("pattern", "plain"), s.get("pattern_color", "#ffffff")
    sp, sc = s.get("sleeve_pattern", "plain"), s.get("sleeve_color", "#ffffff")
    cp, cc = s.get("cap_pattern", "plain"), s.get("cap_color", "#ffffff")
    ov = ""
    if pat == "hoops":
        ov = "".join(f'<rect x="14" y="{y}" width="20" height="3.4" fill="{pc}"/>' for y in (17, 24, 31, 38))
    elif pat == "band":
        ov = f'<rect x="14" y="23" width="20" height="5" fill="{pc}"/>'
    elif pat == "stripes":
        ov = "".join(f'<rect x="{x}" y="14" width="3" height="26" fill="{pc}"/>' for x in (18, 27))
    elif pat == "sash":
        ov = f'<polygon points="14,14 20,14 34,38 28,38" fill="{pc}"/>'
    elif pat == "dsashes":
        ov = "".join(f'<polygon points="{14 + o},14 {18 + o},14 {34},{30 - o} {34},{34 - o}" fill="{pc}"/>' for o in (0, 8)) + f'<polygon points="14,26 14,30 26,40 30,40" fill="{pc}"/>'
    elif pat == "check_sash":
        ov = f'<polygon points="14,14 21,14 34,34 34,40 30,40" fill="{pc}"/>' + "".join(f'<rect x="{15 + i * 3}" y="{15 + i * 4}" width="2.6" height="2.6" fill="{body}"/>' for i in range(7))
    elif pat == "star":
        ov = f'<polygon points="24,18 26.2,24 32.5,24 27.4,27.7 29.3,33.7 24,30 18.7,33.7 20.6,27.7 15.5,24 21.8,24" fill="{pc}"/>'
    elif pat == "stars":
        ov = "".join(f'<polygon transform="translate({x},{y}) scale(0.45)" points="0,-8 2.4,-2.5 8,-2.5 3.6,1.5 5.2,7 0,3.6 -5.2,7 -3.6,1.5 -8,-2.5 -2.4,-2.5" fill="{pc}"/>' for x, y in ((19, 20), (29, 20), (24, 27), (19, 34), (29, 34)))
    elif pat == "spots":
        ov = "".join(f'<circle cx="{x}" cy="{y}" r="2.4" fill="{pc}"/>' for x, y in ((19, 20), (29, 20), (24, 27), (19, 34), (29, 34)))
    elif pat == "spots3":
        ov = "".join(f'<circle cx="{x}" cy="{y}" r="2.6" fill="{pc}"/>' for x, y in ((19, 22), (29, 22), (24, 30)))
    elif pat == "horseshoe":
        ov = f'<path d="M18 20 a6 6 0 1 0 12 0 v8 a6 6 0 0 1 -12 0 z" fill="none" stroke="{pc}" stroke-width="2.6"/>'
    elif pat == "square":
        ov = f'<rect x="19" y="21" width="10" height="10" fill="none" stroke="{pc}" stroke-width="2.2"/>'
    elif pat == "mandala":
        ov = "".join(f'<circle cx="24" cy="27" r="{r}" fill="none" stroke="{c}" stroke-width="1.6"/>' for r, c in ((9, pc), (6, s.get("pattern_color2", "#f2c200")), (3, pc)))
    elif pat == "logo":
        ov = f'<polygon points="21,22 27,22 24,17" fill="{pc}"/><rect x="21" y="23" width="6" height="2" fill="{pc}"/>'
    # sleeves
    sl = ""
    if sp == "check":
        sl = _checks(2, 14, 12, 16, sleeves, sc, 3) + _checks(34, 14, 12, 16, sleeves, sc, 3)
    elif sp == "hoops":
        sl = "".join(f'<rect x="{x}" y="{y}" width="12" height="3" fill="{sc}"/>' for x in (2, 34) for y in (17, 23, 29))
    elif sp == "armband":
        sl = "".join(f'<rect x="{x}" y="22" width="12" height="3.5" fill="{sc}"/>' for x in (2, 34))
    elif sp == "stars":
        sl = "".join(f'<circle cx="{x}" cy="{y}" r="1.4" fill="{sc}"/>' for x in (8, 40) for y in (18, 24, 30))
    # cap
    capfill = cap
    capov = ""
    if cp == "check":
        capov = f'<clipPath id="c{abs(hash(str(s))) % 10000}"><path d="M15 2 a9 6 0 0 1 18 0 l1 4 h-20 z"/></clipPath><g clip-path="url(#c{abs(hash(str(s))) % 10000})">{_checks(14, 0, 22, 7, cap, cc, 4)}</g>'
    elif cp == "halves":
        capov = f'<path d="M24 0.2 a9 6 0 0 1 9 5.8 l1 0 h-10 z" fill="{cc}"/>'
    elif cp == "stripe":
        capov = f'<rect x="22.5" y="0" width="3" height="6" fill="{cc}"/>'
    grey = ' opacity="0.55"' if s.get("status") == "tbc" else ""
    return f'''<svg class="silks" viewBox="0 0 48 44" width="40" height="40" aria-label="{html.escape(s.get("desc", ""))}"{grey}>
<rect x="2" y="14" width="12" height="16" rx="2" fill="{sleeves}" stroke="#333" stroke-width="1"/>
<rect x="34" y="14" width="12" height="16" rx="2" fill="{sleeves}" stroke="#333" stroke-width="1"/>
{sl}
<rect x="14" y="14" width="20" height="26" rx="2" fill="{body}" stroke="#333" stroke-width="1"/>
{ov}
<path d="M15 2 a9 6 0 0 1 18 0 l1 4 h-20 z" fill="{capfill}" stroke="#333" stroke-width="1"/>
{capov}
<rect x="13" y="6" width="22" height="2.5" fill="#333"/>
</svg>'''


def build_racebook(pred: Prediction, scenarios: dict | None = None) -> str:
    rs = pred.runners
    e = html.escape
    order = list(np.argsort(-pred.win_prob))
    rank_of = {i: k + 1 for k, i in enumerate(order)}
    race = pred.race
    top = order[0]
    edges = sorted(range(len(rs)), key=lambda i: -(pred.win_prob[i] * rs[i]["odds"] - 1))
    value = next(i for i in edges if i != top and rs[i]["odds"] >= 10)
    each_way = next(i for i in sorted(range(len(rs)), key=lambda i: -pred.top4_prob[i]) if i not in (top, value) and rs[i]["odds"] >= 8)
    roughie = next(i for i in sorted(range(len(rs)), key=lambda i: -pred.top4_prob[i]) if rs[i]["odds"] >= 20 and i not in (top, value, each_way))
    sm = pred.speed_map

    def name_bar(r):
        return f"{e(r['horse']).upper()} ({r['barrier']})"

    # ---------- rundown -------------------------------------------------------------
    leaders = ", ".join(f"{rs[i]['horse']} ({rs[i]['barrier']})" for i in sorted(range(len(rs)), key=lambda i: rs[i].get("settle", 12))[:4])
    swoopers = ", ".join(rs[i]["horse"] for i in sorted(range(len(rs)), key=lambda i: -rs[i].get("settle", 12))[:5])
    t1, t2, t3, t4 = (rs[i] for i in order[:4])
    c = pred.components
    pat = pred.pattern

    def why(i):
        r = rs[i]
        notes = []
        tr = pat[r["horse"]]["type_record"]
        notes.append(r["verdict"])
        notes.append(tr["notes"][0].split(" (")[0] + ".")
        return " ".join(notes)

    rundown = f"""
<section class="rundown">
  <h2>The rundown</h2>
  <p><b>The shape.</b> {e(race['pace_outlook'])} The model reads the tempo as <b>{e(pred.pace[0])}</b> ({e(pred.pace[1])}). Expect {e(leaders)} to be in the first half a dozen, with {e(swoopers)} settling at the tail and needing the race run to suit.
  The track is forecast {e(race['expected_track'])}; the model carries a going distribution of {", ".join(f"{k.replace('_PLUS', '+').replace('SOFT', 'Soft ').replace('GOOD', 'Good')} {int(100 * v)}%" for k, v in pred.track_distribution.items())}.</p>
  <p><b>The market.</b> Point Of Law heads the market after his Geelong Cup win; Trustyourinstinct, Hopewell Rock, Shockletz, Juja Kibo and So You Are are next. The model's biggest disagreements with the market are where a trainer is attempting a preparation they have never used before, or where the horse's closest historical match comes from a trainer who works a completely different way.</p>
  <p><b>Selections.</b> <b>1. {name_bar(t1)}</b> {e(why(order[0]))} <b>2. {name_bar(t2)}</b> {e(why(order[1]))} <b>3. {name_bar(t3)}</b> {e(why(order[2]))} <b>4. {name_bar(t4)}</b> {e(why(order[3]))}</p>
  <p><b>Suggested bets.</b> Best bet {e(rs[top]['horse'])} win at ${rs[top]['odds']:.1f} (model {_pct(pred.win_prob[top])}). Each-way {e(rs[each_way]['horse'])} at ${rs[each_way]['odds']:.1f} (top-4 {_pct(pred.top4_prob[each_way])}).
  Value {e(rs[value]['horse'])} at ${rs[value]['odds']:.1f} (model {_pct(pred.win_prob[value])} against a market {_pct(c[value]['market_prob'])}). Roughie {e(rs[roughie]['horse'])} at ${rs[roughie]['odds']:.1f}.
  First four box {' - '.join(str(rs[i]['no']) for i in order[:4])}; wider box {' - '.join(str(rs[i]['no']) for i in order[:6])}; quinella {rs[order[0]]['no']} / {rs[order[1]]['no']}.</p>
  <p><b>If the day changes.</b> On-speed horses winning the early races lifts Point Of Law, Enfjaar and So You Are; swoopers dominating lifts Juja Kibo, Middle Earth, Trustyourinstinct and Shockletz. A Good 4 helps Trustyourinstinct and Zahrann; Soft 7 or worse helps Juja Kibo, Mr Monaco and Hopewell Rock and hurts Trustyourinstinct and Goodie Two Shoes.</p>
</section>"""

    # ---------- runner blocks ---------------------------------------------------------
    blocks = []
    for i in sorted(range(len(rs)), key=lambda i: rs[i]["no"]):
        r = rs[i]
        comp = c[i]
        p = pat[r["horse"]]
        stars = _stars(pred.win_prob[i])
        figs = "-".join(_fig(x["pos"], x["field"]) for x in r.get("last4", []))
        gf = going_fit(r)["fit"]
        going_word = "loves wet" if gf >= 0.7 else "handles wet" if gf > 0.1 else "wants firm" if gf <= -0.7 else "prefers dry" if gf < -0.1 else "wet unknown"
        tr = p["type_record"]
        best = p["best_matches"][0]
        rows = "".join(
            f"<tr><td>{e(x['date'])}</td><td>{e(x['track'])}</td><td>{e(x['race'])}{' <i title=\"race label inferred from form figures\">~</i>' if x.get('approx') else ''}</td><td class='num'>{x['dist']}</td><td class='num'><b>{x['pos']}</b>/{x['field']}</td><td>{e(str(x['margin']))}</td><td>{e(x['going'])}</td></tr>"
            for x in r.get("last4", []))
        edge = pred.win_prob[i] * r["odds"] - 1
        blocks.append(f"""
<article class="runner r{stars}">
  <div class="rhead">
    {silks_svg(r['silks'])}
    <div class="rtitle">
      <div class="rname"><span class="rno">{r['no']}</span> {name_bar(r)} <span class="stars" title="{stars} of 5">{'★' * stars}{'☆' * (5 - stars)}</span></div>
      <div class="rsub">{e(r['jockey'])} · {e(r['trainer'])} · {r['weight']}kg · {r['age']}{r['sex'].lower()} · {e(r['country'])} · {STYLE_LABEL[r['style']]}, settles ~{r.get('settle', '?')} · {going_word} · gear: {e(r.get('gear', 'none'))}</div>
      <div class="rsilks">{e(r['silks'].get('desc', ''))}</div>
    </div>
    <div class="rnums">
      <div class="big">${r['odds']:.1f}</div>
      <div>model <b>{_pct(pred.win_prob[i])}</b> win · <b>{_pct(pred.top4_prob[i])}</b> top-4 · rank {rank_of[i]}</div>
      <div>edge <b class="{'pos' if edge > 0.1 else 'neg' if edge < -0.3 else ''}">{100 * edge:+.0f}%</b> · form {e(figs)}</div>
    </div>
  </div>
  <table class="form"><thead><tr><th>Date</th><th>Track</th><th>Race</th><th>Dist</th><th>Fin</th><th>Margin</th><th>Going</th></tr></thead><tbody>{rows}</tbody></table>
  <p class="comment"><b>Comment:</b> {e(r['verdict'])} <b>Trainer:</b> {e(tr['notes'][0])}. <b>Modelled on:</b> {e(best['horse'])} {best['year']} ({e(best['trainer'])}), credit x{best['trainer_multiplier']:.2f}. <b>Last start:</b> {e(pred.form[r['horse']]['runs'][0]['tempo_note'])}.</p>
</article>""")

    generated = dt.datetime.now().strftime("%d %b %Y %H:%M")
    return f"""<title>Cup Racebook</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&display=swap">
<style>
/* Layout: newspaper racebook - masthead, rundown column, then one block per runner in saddlecloth order with drawn silks; A4 portrait print. */
:root {{
  --bg: #fbfaf6; --fg: #1a1f1b; --muted: #5b655c; --rule: #d9ddd5; --panel: #ffffff;
  --green: #0f4d3a; --gold: #c99a2e; --pos: #157347; --neg: #a23b2a;
  --r5: #d9ecdd; --r4: #e8f2e6; --r3: #f3f5ee; --r2: #f9f8f3; --r1: #ffffff;
  --display: "Barlow Condensed", "Arial Narrow", Impact, sans-serif;
  --body: "Source Sans 3", "Segoe UI", Helvetica, Arial, sans-serif;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #121715; --fg: #e8ece8; --muted: #a2ada4; --rule: #2b332d; --panel: #1a211d; --green: #0b3b2d; --pos: #4fcf8a; --neg: #ff8b73;
  --r5: #1d3a2a; --r4: #1b2d22; --r3: #1a221d; --r2: #171d19; --r1: #151a17; color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --bg: #121715; --fg: #e8ece8; --muted: #a2ada4; --rule: #2b332d; --panel: #1a211d; --green: #0b3b2d; --pos: #4fcf8a; --neg: #ff8b73;
  --r5: #1d3a2a; --r4: #1b2d22; --r3: #1a221d; --r2: #171d19; --r1: #151a17; color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 13px; line-height: 1.35; margin: 0; }}
.page {{ max-width: 1040px; margin: 0 auto; padding-block: 0 24px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
header.mast {{ background: var(--green); color: #fff; margin: 0 -16px; padding: 16px max(16px, env(safe-area-inset-left, 0px)) 14px max(16px, env(safe-area-inset-right, 0px)); border-bottom: 5px solid var(--gold); }}
.mast-inner {{ max-width: 1008px; margin: 0 auto; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-end; gap: 8px 20px; }}
.mast h1 {{ font-family: var(--display); font-weight: 800; font-size: clamp(28px, 5vw, 42px); margin: 0; line-height: .95; text-wrap: balance; }}
.eyebrow {{ font-family: var(--display); letter-spacing: .14em; text-transform: uppercase; font-size: 12px; opacity: .85; }}
.mast .facts {{ font-size: 12.5px; max-width: 46ch; }}
h2 {{ font-family: var(--display); font-weight: 700; font-size: 22px; letter-spacing: .02em; margin: 18px 0 6px; text-transform: uppercase; }}
.rundown p {{ margin: 0 0 8px; max-width: 95ch; }}
.key {{ display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 11.5px; color: var(--muted); margin: 6px 0 10px; }}
.key i {{ display: inline-block; width: 14px; height: 12px; border: 1px solid var(--rule); vertical-align: -2px; margin-right: 4px; }}
.runner {{ background: var(--panel); border: 1px solid var(--rule); border-left: 6px solid var(--rule); margin: 0 0 10px; padding: 8px 10px 6px; break-inside: avoid; }}
.runner.r5 {{ background: var(--r5); border-left-color: var(--gold); }} .runner.r4 {{ background: var(--r4); border-left-color: #6fae7c; }} .runner.r3 {{ background: var(--r3); border-left-color: #a9c3a9; }} .runner.r2 {{ background: var(--r2); }} .runner.r1 {{ background: var(--r1); }}
.rhead {{ display: flex; gap: 10px; align-items: flex-start; flex-wrap: wrap; }}
.silks {{ flex: 0 0 auto; width: 52px; height: 62px; object-fit: contain; }}
.rtitle {{ flex: 1 1 320px; min-width: 0; }}
.rname {{ font-family: var(--display); font-weight: 800; font-size: 22px; letter-spacing: .01em; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
.rno {{ display: inline-grid; place-items: center; min-width: 26px; height: 26px; padding: 0 5px; border-radius: 4px; background: var(--green); color: #fff; font-size: 16px; }}
.stars {{ color: var(--gold); font-size: 15px; letter-spacing: .05em; font-family: var(--body); }}
.rsub {{ color: var(--muted); font-size: 12px; }}
.rsilks {{ font-size: 11px; color: var(--muted); font-style: italic; }}
.rnums {{ text-align: right; font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; flex: 0 0 auto; }}
.rnums .big {{ font-family: var(--display); font-size: 24px; font-weight: 800; color: var(--fg); line-height: 1; }}
.rnums b {{ color: var(--fg); }} b.pos {{ color: var(--pos); }} b.neg {{ color: var(--neg); }}
table.form {{ border-collapse: collapse; width: 100%; margin-top: 6px; font-size: 11.5px; font-variant-numeric: tabular-nums; }}
table.form th {{ font-family: var(--display); text-transform: uppercase; letter-spacing: .08em; font-size: 10.5px; color: var(--muted); text-align: left; padding: 2px 6px; border-bottom: 1px solid var(--rule); }}
table.form td {{ padding: 2px 6px; border-bottom: 1px dotted var(--rule); }}
table.form td.num {{ text-align: right; white-space: nowrap; }}
table.form i {{ color: var(--muted); font-style: normal; }}
.comment {{ margin: 6px 0 0; font-size: 12px; }}
footer {{ margin-top: 16px; color: var(--muted); font-size: 11px; border-top: 1px solid var(--rule); padding-top: 8px; }}
@page {{ size: A4 portrait; margin: 10mm; }}
@media print {{
  :root {{ --bg: #ffffff; }}
  body {{ font-size: 10.5px; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  .page {{ max-width: none; padding: 0; }}
  header.mast {{ margin: 0; padding: 10px 12px; }}
  .runner {{ margin-bottom: 6px; padding: 6px 8px 4px; }}
  .rname {{ font-size: 17px; }}
  .field {{ break-before: page; }}
  table.form {{ font-size: 10px; }}
}}
</style>
<div class="page">
<header class="mast"><div class="mast-inner">
  <div><div class="eyebrow">Flemington · Race 7 · 3:00pm AEDT · Tuesday 3 November 2026 · 3200m · Group 1 handicap</div><h1>Lexus Melbourne Cup racebook</h1></div>
  <div class="facts">Track: {e(race['expected_track'])}. 24 runners. Prices are a 31 Oct - 2 Nov consensus.</div>
</div></header>
{rundown}
<section class="field">
  <h2>The field</h2>
  <div class="key"><span>Barrier in brackets after the name. Form figures oldest to newest, 0 = unplaced (10th or worse).</span>
    <span><i style="background:var(--r5)"></i>★★★★★ 12%+ win chance</span><span><i style="background:var(--r4)"></i>★★★★ 8%+</span><span><i style="background:var(--r3)"></i>★★★ 5%+</span><span><i style="background:var(--r2)"></i>★★ 2.5%+</span><span><i style="background:var(--r1)"></i>★ outsider</span>
    <span>Silks transcribed from the ABC sweep grid. ~ = race label inferred from published form figures.</span></div>
  {''.join(blocks)}
</section>
<footer>Generated {generated} from the equintel melbourne_cup model. Silks drawn from the ABC Melbourne Cup sweep grid. Research model, not betting advice. Gamble responsibly.</footer>
</div>
"""
