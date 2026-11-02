"""Best Bets-format form guide: runners 1-24 in order, silks, barrier, career / track / distance / going / first-up
stats grid, last 12 starts with weight, barrier, jockey, SP, winner, margin, time and in-running positions, then the
race rundown and selection reasoning. Reads melbourne_cup/cup_data/form_feed_2026.json (see form_feed_schema.json);
runners without a feed entry fall back to the last4 lines in the field file and show '-' for missing fields."""
from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path

import numpy as np

from .model import Prediction
from .racebook import silks_svg, _stars, STYLE_LABEL

FEED = Path(__file__).resolve().parent / "cup_data" / "form_feed_2026.json"
ANALYSIS = Path(__file__).resolve().parent / "cup_data" / "trainer_analysis.json"
STAT_KEYS = [("career", "Career"), ("track", "Trk"), ("distance", "Dst"), ("track_distance", "Trk/Dst"), ("firm", "Firm"), ("good", "Good"),
             ("soft", "Soft"), ("heavy", "Heavy"), ("first_up", "1st Up"), ("second_up", "2nd Up"), ("jockey", "Jky"), ("clockwise", "Clockwise")]


def _pct(x):
    return f"{100 * x:.1f}%"


def _v(x):
    return html.escape(str(x)) if x not in (None, "") else "-"


def load_feed() -> dict:
    if FEED.exists():
        return json.loads(FEED.read_text())
    return {}


def load_analysis() -> dict:
    if ANALYSIS.exists():
        return json.loads(ANALYSIS.read_text()).get("trainers", {})
    return {}


def build_formbook(pred: Prediction) -> str:
    rs = pred.runners
    e = html.escape
    feed = load_feed()
    analysis = load_analysis()
    order = list(np.argsort(-pred.win_prob))
    rank_of = {i: k + 1 for k, i in enumerate(order)}
    c = pred.components
    pat = pred.pattern
    t = [rs[i] for i in order[:4]]
    missing = []
    blocks = []
    for i in sorted(range(len(rs)), key=lambda i: rs[i]["no"]):
        r = rs[i]
        f = feed.get(r["horse"], {})
        stats = f.get("stats", {})
        if not f.get("form_last12"):
            missing.append(r["horse"])
        runs = f.get("form_last12") or [
            {"date": x["date"], "track": x["track"], "race": x["race"], "dist": x["dist"], "going": x["going"], "finish": x["pos"], "field": x["field"], "margin": x["margin"]}
            for x in reversed(r.get("last4", []))]
        stars = _stars(pred.win_prob[i])
        stat_cells = "".join(f"<div><span>{lab}</span><b>{_v(f.get('career') if key == 'career' else stats.get(key))}</b></div>" for key, lab in STAT_KEYS)
        run_rows = "".join(
            f"<tr><td>{_v(x.get('date'))}</td><td>{_v(x.get('track'))}</td><td>{_v(x.get('race'))} <small>{_v(x.get('class'))}</small></td><td class='num'>{_v(x.get('dist'))}</td>"
            f"<td>{_v(x.get('going'))}</td><td class='num'><b>{_v(x.get('finish'))}</b>/{_v(x.get('field'))}</td><td>{_v(x.get('margin'))}</td><td class='num'>{_v(x.get('weight'))}</td>"
            f"<td class='num'>{_v(x.get('barrier'))}</td><td>{_v(x.get('jockey'))}</td><td>{_v(x.get('sp'))}</td><td>{_v(x.get('winner'))}</td><td>{_v(x.get('time'))}</td>"
            f"<td class='num'>{_v(x.get('pos_800'))}-{_v(x.get('pos_400'))}</td><td class='small'>{_v(x.get('in_running') or x.get('steward_comment'))}</td></tr>"
            for x in runs)
        tr = pat[r["horse"]]["type_record"]
        best = pat[r["horse"]]["best_matches"][0]
        blocks.append(f"""
<article class="runner r{stars}">
  <div class="rhead">
    {silks_svg(r['silks'])}
    <div class="rtitle">
      <div class="rname"><span class="rno">{r['no']}</span> {e(r['horse']).upper()} ({r['barrier']}) <span class="stars">{'★' * stars}{'☆' * (5 - stars)}</span></div>
      <div class="rsub">{_v(f.get('colour_sex_age') or f"{r['age']}{r['sex'].lower()}")} · {_v(f.get('sire_dam'))} · {e(r['trainer'])} · <b>{e(r['jockey'])}</b> · {r['weight']}kg · {STYLE_LABEL[r['style']]} · gear {e(r.get('gear', 'none'))}</div>
      <div class="rsilks">Colours: {_v(f.get('colours') or r['silks'].get('desc'))}{' · owner ' + e(f['owner']) if f.get('owner') else ''}</div>
    </div>
    <div class="rnums"><div class="big">${r['odds']:.1f}</div><div>model <b>{_pct(pred.win_prob[i])}</b> · top-4 <b>{_pct(pred.top4_prob[i])}</b> · rank {rank_of[i]}</div><div>prizemoney {_v(f.get('prizemoney'))}</div></div>
  </div>
  <div class="stats">{stat_cells}</div>
  <div class="scroll"><table class="form"><thead><tr><th>Date</th><th>Track</th><th>Race</th><th>Dist</th><th>Going</th><th>Fin</th><th>Mgn</th><th>Wt</th><th>Bar</th><th>Jockey</th><th>SP</th><th>Winner</th><th>Time</th><th>800-400</th><th>In running</th></tr></thead><tbody>{run_rows}</tbody></table></div>
  <p class="comment"><b>Comment:</b> {e(r['verdict'])} <b>Modelled on:</b> {e(best['horse'])} {best['year']} ({e(best['trainer'])}, credit x{best['trainer_multiplier']:.2f}).</p>
  <p class="comment tstyle"><b>Trainer style, {e(r['trainer'])}:</b> {e(analysis.get(r['trainer'], {}).get('this_year', {}).get(r['horse'], ''))} <b>Record with this type:</b> {e(tr['notes'][0])}.</p>
</article>""")

    generated = dt.datetime.now().strftime("%d %b %Y %H:%M")
    warn = f'<p class="warn">Form feed missing for {len(missing)} runners ({e(", ".join(missing))}); showing the last four starts from the field file instead. Run scripts/import_form.py once a form export is available.</p>' if missing else ""
    return f"""<title>Cup Form Book</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Source+Sans+3:wght@400;600;700&display=swap">
<style>
/* Layout: full-width form book, one runner per block in saddlecloth order, stats grid then 12-run table; A4 landscape print. */
:root {{ --bg:#fbfaf6; --fg:#1a1f1b; --muted:#5b655c; --rule:#d9ddd5; --panel:#fff; --green:#0f4d3a; --gold:#c99a2e; --warn:#a23b2a;
  --r5:#d9ecdd; --r4:#e8f2e6; --r3:#f3f5ee; --r2:#f9f8f3; --r1:#fff; --display:"Barlow Condensed","Arial Narrow",Impact,sans-serif; --body:"Source Sans 3","Segoe UI",Helvetica,Arial,sans-serif; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#121715; --fg:#e8ece8; --muted:#a2ada4; --rule:#2b332d; --panel:#1a211d; --green:#0b3b2d; --warn:#ff8b73; --r5:#1d3a2a; --r4:#1b2d22; --r3:#1a221d; --r2:#171d19; --r1:#151a17; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#121715; --fg:#e8ece8; --muted:#a2ada4; --rule:#2b332d; --panel:#1a211d; --green:#0b3b2d; --warn:#ff8b73; --r5:#1d3a2a; --r4:#1b2d22; --r3:#1a221d; --r2:#171d19; --r1:#151a17; color-scheme:dark; }}
*{{box-sizing:border-box}} body{{background:var(--bg);color:var(--fg);font-family:var(--body);font-size:12.5px;line-height:1.3;margin:0}}
.page{{max-width:1240px;margin:0 auto;padding-block:0 24px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}}
header.mast{{background:var(--green);color:#fff;margin:0 -16px;padding:14px max(16px,env(safe-area-inset-left,0px)) 12px max(16px,env(safe-area-inset-right,0px));border-bottom:5px solid var(--gold)}}
.mast h1{{font-family:var(--display);font-weight:800;font-size:clamp(26px,4vw,38px);margin:0;line-height:.95}} .eyebrow{{font-family:var(--display);letter-spacing:.14em;text-transform:uppercase;font-size:12px;opacity:.85}}
h2{{font-family:var(--display);font-weight:700;font-size:22px;margin:16px 0 6px;text-transform:uppercase}}
.warn{{color:var(--warn);font-weight:600}}
.runner{{background:var(--panel);border:1px solid var(--rule);border-left:6px solid var(--rule);margin:0 0 10px;padding:8px 10px 6px;break-inside:avoid}}
.runner.r5{{background:var(--r5);border-left-color:var(--gold)}} .runner.r4{{background:var(--r4);border-left-color:#6fae7c}} .runner.r3{{background:var(--r3);border-left-color:#a9c3a9}} .runner.r2{{background:var(--r2)}}
.rhead{{display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap}} .rtitle{{flex:1 1 360px;min-width:0}}
.rname{{font-family:var(--display);font-weight:800;font-size:22px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
.rno{{display:inline-grid;place-items:center;min-width:26px;height:26px;padding:0 5px;border-radius:4px;background:var(--green);color:#fff;font-size:16px}}
.stars{{color:var(--gold);font-family:var(--body);font-size:15px}} .rsub,.rsilks{{color:var(--muted);font-size:12px}} .rsilks{{font-style:italic}}
.rnums{{text-align:right;font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}} .rnums .big{{font-family:var(--display);font-size:24px;font-weight:800;color:var(--fg);line-height:1}} .rnums b{{color:var(--fg)}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(88px,1fr));gap:4px;margin:6px 0;font-size:11.5px}} .stats div{{border:1px solid var(--rule);padding:2px 6px;background:rgba(255,255,255,.35)}} .stats span{{display:block;font-family:var(--display);text-transform:uppercase;letter-spacing:.08em;font-size:10px;color:var(--muted)}}
.scroll{{overflow-x:auto}} table.form{{border-collapse:collapse;width:100%;font-size:11px;font-variant-numeric:tabular-nums}}
table.form th{{font-family:var(--display);text-transform:uppercase;letter-spacing:.06em;font-size:10px;color:var(--muted);text-align:left;padding:2px 5px;border-bottom:1px solid var(--rule);white-space:nowrap}}
table.form td{{padding:2px 5px;border-bottom:1px dotted var(--rule);vertical-align:top}} td.num{{text-align:right;white-space:nowrap}} td.small{{font-size:10.5px;color:var(--muted)}} table.form small{{color:var(--muted)}}
.comment{{margin:6px 0 0;font-size:12px}} .tstyle{{border-top:1px dotted var(--rule);padding-top:4px}}
.tcard{{background:var(--panel);border:1px solid var(--rule);padding:8px 12px;margin:0 0 8px;break-inside:avoid}} .tcard h3{{font-family:var(--display);font-size:18px;margin:0 0 4px;text-transform:uppercase}} .tcard h3 small{{font-family:var(--body);text-transform:none;font-weight:400;color:var(--muted);font-size:12px}} .tcard p{{margin:0 0 4px;max-width:120ch}}
footer{{margin-top:16px;color:var(--muted);font-size:11px;border-top:1px solid var(--rule);padding-top:8px}}
@page{{size:A4 landscape;margin:9mm}} @media print{{:root{{--bg:#fff}} body{{font-size:10px;-webkit-print-color-adjust:exact;print-color-adjust:exact}} .page{{max-width:none;padding:0}} header.mast{{margin:0}} .scroll{{overflow:visible}} table.form{{font-size:9.5px}}}}
</style>
<div class="page">
<header class="mast"><div class="eyebrow">Flemington · Race 7 · 3:00pm AEDT · Tuesday 3 November 2026 · 3200m · Group 1 handicap · {e(pred.race['expected_track'])}</div><h1>Lexus Melbourne Cup form book</h1></header>
{warn}
<section class="bets"><h2>Best bets</h2><p><b>Best bet:</b> {e(t[0]['horse'])} ({t[0]['barrier']}) ${t[0]['odds']:.1f} · <b>Next best:</b> {e(t[1]['horse'])} ({t[1]['barrier']}) ${t[1]['odds']:.1f} · <b>Third:</b> {e(t[2]['horse'])} ({t[2]['barrier']}) · <b>Fourth:</b> {e(t[3]['horse'])} ({t[3]['barrier']}) · <b>First four box:</b> {' - '.join(str(x['no']) for x in t)} · <b>Wider box:</b> {' - '.join(str(rs[i]['no']) for i in order[:6])}</p></section>
<section>{''.join(blocks)}</section>
<section class="trainers">
<h2>Trainer styles: how each stable prepares a Cup horse, and who in this field is doing it the same way</h2>
{''.join(f"""<article class="tcard"><h3>{e(name)} <small>({', '.join(e(h) for h in a.get('runners', []))})</small></h3>
<p><b>Method.</b> {e(a.get('style', ''))}</p>
<p><b>Cup record.</b> {e(a.get('cup_record', ''))}</p>
<p><b>This year.</b> {' '.join(f'<b>{e(h)}:</b> {e(t)}' for h, t in a.get('this_year', {}).items())}</p>
<p><b>Closest method in this field.</b> {e(a.get('like_in_field', ''))}</p></article>""" for name, a in analysis.items())}
</section>
<section>
<h2>The run of the race and how the picks were made</h2>
<p><b>Shape:</b> {e(pred.race['pace_outlook'])} Model pace read: {e(pred.pace[0])} ({e(pred.pace[1])}).</p>
<p><b>Early.</b> Royal Supremacy (21) and Enfjaar (8) are the two who want to lead; Wine Dark Sea (23), Kizlyar (20), So You Are (7) and Onesmoothoperator (13) will be in the first six with Vauban (22) and Point Of Law (9) trailing them, which is the position Ethan Brown wants. Mr Monaco (16) and Sons And Lovers (3) midfield. Juja Kibo (14), Trustyourinstinct (18), Middle Earth (1), Knight's Choice (2), Zakouma (10), Asterix (6) and Shockletz (17) will be in the last eight. With eight on-pace types and several drawn wide, the first 800m should be honest, which is what the swoopers need and what Twilight Payment-style leaders do not get.</p>
<p><b>Middle stages.</b> Expect the tempo to steady down the back as the leaders try to get a breather, then the Flemington straight to be a 400m test of stamina rather than speed. On a Soft 5-6 the horses with wet form (Juja Kibo, Mr Monaco, So You Are, Hopewell Rock, Middle Earth) hold their ground; Trustyourinstinct, Goodie Two Shoes and Zahrann are the ones whose connections have said they want it firm.</p>
<p><b>Selections:</b> 1. {e(t[0]['horse'])} ({t[0]['barrier']}) · 2. {e(t[1]['horse'])} ({t[1]['barrier']}) · 3. {e(t[2]['horse'])} ({t[2]['barrier']}) · 4. {e(t[3]['horse'])} ({t[3]['barrier']}).</p>
<p><b>Why these four.</b> {e(t[0]['horse'])}: {e(t[0]['verdict'])} {e(analysis.get(t[0]['trainer'], {}).get('this_year', {}).get(t[0]['horse'], ''))} {e(t[1]['horse'])}: {e(t[1]['verdict'])} {e(analysis.get(t[1]['trainer'], {}).get('this_year', {}).get(t[1]['horse'], ''))} {e(t[2]['horse'])}: {e(t[2]['verdict'])} {e(analysis.get(t[2]['trainer'], {}).get('this_year', {}).get(t[2]['horse'], ''))} {e(t[3]['horse'])}: {e(t[3]['verdict'])} {e(analysis.get(t[3]['trainer'], {}).get('this_year', {}).get(t[3]['horse'], ''))}</p>
<p><b>Why not the favourite.</b> Point Of Law is a good horse with the right profile (lightly raced, light weight, sustained on-pace Geelong Cup win), but he is being prepared a way his trainer has never prepared a Cup horse before, and the trainers who won this way (Weld, de Royer-Dupre, Delzangles) did it as their standard method. The market has him at 12-13%; the model at about 8%. He is in the first-four box, not the best bet.</p>
<p><b>Method:</b> each runner's price is shrunk towards the model's own view, which adds tempo-adjusted lead-up form, a preparation match against every first-four finisher since 2000 gated by how alike the trainers' methods are, the trainer's own record with this type of preparation, going fit from wet records, race shape and weight / barrier / age. The per-runner comments above show each piece.</p>
</section>
<footer>Generated {generated}. Research model, not betting advice.</footer>
</div>
"""
