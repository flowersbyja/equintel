"""Best Bets-format form guide (newspaper layout), phone-friendly.

Per runner: header line (TAB no, name, blinkers *, tcbdwh flags, starts, prize money, jockey, (barrier), weight, trainer and
location), colour-sex-age-sire-dam-owner line, silks, stats grid (Track, Dist, Tk/Dist, WDR, W&P%, 1st-up, Last 15, Firm,
Good, Soft, Heavy, Synth, 2nd-up, Rating), the last FIVE starts in Best Bets column order, and underneath a TRAINER block
with its own star rating: how the trainer mapped this Cup, which past horse/prep it resembles, whether the trainer normally
trains this way, and whether the horse's running style matches the stable's habit. Ends with colour-coded 1-4 per factor."""
from __future__ import annotations

import datetime as dt
import html
import json
import re
from pathlib import Path

import numpy as np

from .going import going_fit
from .model import Prediction
from .racebook import silks_svg, _stars
from .patterns import campaign_vector, style_similarity, FORWARD_SCORE

DATA = Path(__file__).resolve().parent / "cup_data"
TRACK_CODE = {"Flemington": "FLEM", "Caulfield": "CAUL", "Moonee Valley": "MV", "Geelong": "GEEL", "Bendigo": "BEND", "Randwick": "RAND", "Rosehill": "ROSE",
              "Eagle Farm": "EFM", "Doomben": "DOOM", "Morphettville": "MORP", "Sandown": "SAND", "Newcastle": "NCLE", "Leopardstown": "LEOP", "Curragh": "CURR",
              "Gowran Park": "GOWR", "Killarney": "KILL", "Navan": "NAVN", "Punchestown": "PUNC", "Fairyhouse": "FAIR", "Limerick": "LIMK", "Ascot": "ASCT",
              "Newmarket": "NMKT", "Newbury": "NBRY", "Goodwood": "GDWD", "York": "YORK", "Chester": "CHST", "Carlisle": "CARL", "Yarmouth": "YARM", "Epsom": "EPSM",
              "Longchamp": "LONG", "Saint-Cloud": "STCL", "Meydan": "MEYD", "Riyadh": "RIYD"}
GOING_LETTER = {"firm": "F", "good": "G", "soft": "S", "heavy": "H", "yield": "S", "synth": "SYN"}
TRAINER_LOC = {"Ciaron Maher": "Ballarat", "Joseph O'Brien": "Ireland", "Chris Waller": "Rosehill", "George Boughey": "Newmarket", "Roger Varian": "Newmarket",
               "Harry Charlton": "Beckhampton", "John Symons & Sheila Laxon": "Ballarat", "Bjorn Baker": "Warwick Farm", "Grahame Begg": "Cranbourne",
               "Gregory Hickman": "Warwick Farm", "Tim Fitzsimmons": "Bendigo", "Thomas Carberry": "Ballarat", "Gai Waterhouse & Adrian Bott": "Flemington"}
TRADITIONAL = {"CAULFIELD_CUP", "COX_PLATE", "MV_CUP", "GEELONG_CUP", "LEXUS_HOTHAM", "BART_CUMMINGS", "HERBERT_POWER", "MACKINNON"}


def _e(x):
    return html.escape(str(x)) if x not in (None, "") else "-"


def _going_letter(g):
    g = (g or "").lower()
    for k, v in GOING_LETTER.items():
        if k in g:
            return v + (re.search(r"\d", g).group(0) if re.search(r"\d", g) else "")
    return "-"


def _parse_date(s):
    m = re.match(r"(\d{1,2})([A-Za-z]{3})(\d{2})", s or "")
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(0), "%d%b%y").date()
    except ValueError:
        return None


def last15(runs):
    """oldest -> newest finishing positions with a spell marker where the gap exceeds 60 days."""
    seq = list(reversed(runs))
    out = ""
    prev = None
    for x in seq:
        d = _parse_date(x.get("date"))
        if prev and d and (d - prev).days > 60:
            out += "✦"
        if d:
            prev = d
        f = x.get("finish")
        out += ("0" if isinstance(f, int) and f > 9 else str(f)) if f not in (None, "") else "x"
    return out[-20:] or "-"


def tcbdwh(r, runs):
    flags = ""
    if any(x.get("track") == "Flemington" and x.get("finish") == 1 for x in runs):
        flags += "t"
    if any(x.get("dist") == 3200 and x.get("finish") == 1 for x in runs):
        flags += "d"
    if any(x.get("track") == "Flemington" and x.get("dist") == 3200 and x.get("finish") == 1 for x in runs):
        flags += "c"
    if any("heavy" in (x.get("going") or "").lower() and x.get("finish") == 1 for x in runs):
        flags += "w"
    last = runs[0] if runs else {}
    if last and isinstance(last.get("sp"), str) and last["sp"].lower().endswith("f") and last.get("finish") != 1:
        flags += "b"
    return f"({flags})" if flags else ""


def trainer_block(pred, r, analysis):
    idx = pred.idx
    p = pred.pattern[r["horse"]]
    tr = p["type_record"]
    prof = idx.trainer_profile(r["trainer"])
    camp = campaign_vector(r, idx.dims, prof)
    departure = 1.0 - style_similarity(prof, camp)
    stable_forward = prof.get("forward", 0.5)
    horse_forward = FORWARD_SCORE[r["style"]]
    style_gap = horse_forward - stable_forward
    own_rec = idx.trainer_type_record(r["trainer"], p["archetype"] or "OTHER")
    best = p["best_matches"][0]
    # star rating for the trainer's pattern fit
    stars = 3
    if own_rec and own_rec["runners"] > 0:
        stars += 1 if own_rec["wins"] > 0 else 0
        stars -= 1 if (own_rec["wins"] == 0 and own_rec["placings"] == 0 and own_rec["runners"] >= 4) else 0
    else:
        stars -= 1  # never done this before
    if departure > 0.3:
        stars -= 1
    if abs(style_gap) > 0.4:
        stars -= 1
    if tr["term"] > 0.25:
        stars += 1
    stars = max(1, min(5, stars))
    a = analysis.get(r["trainer"], {})
    normal = "Yes, this is how the stable normally prepares a Cup horse." if departure <= 0.2 else (
        "Partly: some of this campaign departs from the stable's usual build." if departure <= 0.35 else
        "No: this horse has been prepared in a way this trainer does not normally use.")
    if abs(style_gap) > 0.4:
        style_txt = (f"The stable's Cup horses are usually ridden {'forward' if stable_forward >= 0.5 else 'from behind'}, but this horse is a "
                     f"{'swooper' if horse_forward < 0.4 else 'forward runner'}: the trainer is building a horse to race against the stable's habit.")
    else:
        style_txt = f"Running style ({'on the speed' if horse_forward >= 0.6 else 'midfield' if horse_forward >= 0.4 else 'swooper'}) matches how this stable's Cup horses are usually ridden."
    same = "same trainer" if best["same_trainer"] else f"different trainer, method similarity {best['trainer_style_similarity']:.2f}, so credit only x{best['trainer_multiplier']:.2f}"
    return f"""
<div class="tblock">
  <div class="tline"><b>TRAINER {html.escape(r['trainer']).upper()}</b> <span class="tstars">{'★' * stars}{'☆' * (5 - stars)}</span> <small>pattern-fit rating</small></div>
  <p><b>How this Cup was mapped:</b> {html.escape(r.get('campaign', ''))}</p>
  <p><b>Most like:</b> {html.escape(best['horse'])} {best['year']} ({html.escape(best['trainer'])}, finished {best['pos']}) - {same}. <b>Trainer's own record with this type:</b> {html.escape(tr['notes'][0])}.</p>
  <p><b>Does this trainer normally do it this way?</b> {normal} {style_txt} <b>Stable method:</b> {html.escape(a.get('style', ''))}</p>
  <p><b>Verdict on the fit:</b> {html.escape(a.get('this_year', {}).get(r['horse'], ''))}</p>
</div>"""


def factor_rankings(pred, feed):
    rs = pred.runners
    n = len(rs)
    win = pred.win_prob

    def wet_score(r, kind):
        rec = r.get("wet_record") or {}
        runs = feed.get(r["horse"], {}).get("form_last12", [])
        if kind == "heavy":
            s = sum((1.0 if x.get("finish") == 1 else 0.5 if (x.get("finish") or 99) <= 3 else 0.0) for x in runs if "heavy" in (x.get("going") or "").lower())
            return s + 0.2 * (r.get("wet_rating", 0))
        if kind == "soft":
            s = sum((1.0 if x.get("finish") == 1 else 0.5 if (x.get("finish") or 99) <= 3 else 0.0) for x in runs if "soft" in (x.get("going") or "").lower() or "yield" in (x.get("going") or "").lower())
            return s + 0.3 * going_fit(r)["fit"]
        return going_fit(r)["fit"] + 0.5 * (rec.get("wins", 0) + 0.5 * (rec.get("seconds", 0) + rec.get("thirds", 0))) / max(rec.get("starts", 1), 1) if rec else going_fit(r)["fit"]

    def trend_score(r):
        s = 0.0
        s += 1.0 if 4 <= r["age"] <= 6 else 0.3 if r["age"] == 7 else 0.0
        s += 1.0 if 51 <= r["weight"] <= 56.5 else 0.5 if r["weight"] <= 57.5 else 0.0
        s += 1.0 if r["barrier"] <= 19 else 0.0
        s += 0.8 if r["sex"] != "M" else 0.0
        s += 1.0 if r["leadup_group"] in {"CAULFIELD_CUP", "COX_PLATE", "GEELONG_CUP", "EURO_DIRECT"} else 0.4
        s += 0.8 if (r["leadups"][0]["tempo"] in ("MODERATE", "FAST")) else 0.0
        s += 0.6 if r["leadups"][0]["finish"] <= 4 else 0.0
        s += 0.5 if r.get("prior_3200") else 0.0
        return s

    cats = [
        ("model", "Our model (all factors)", "#157347", lambda i: win[i]),
        ("wet", "Wet-track form", "#1f5fbf", lambda i: wet_score(rs[i], "wet")),
        ("heavy", "Heavy-track form", "#5b2d8e", lambda i: wet_score(rs[i], "heavy")),
        ("slow", "Slow (soft) track form", "#0f8a8a", lambda i: wet_score(rs[i], "soft")),
        ("local", "Raced only in Australia / NZ", "#c99a2e", lambda i: win[i] if rs[i]["origin"] == "LOCAL" else -1),
        ("tradition", "Traditional Cup lead-up race", "#c0392b", lambda i: win[i] if rs[i]["leadup_group"] in TRADITIONAL else -1),
        ("trend", "15-year Cup trends (2011-2025)", "#444", lambda i: trend_score(rs[i]) + 0.01 * win[i]),
    ]
    out = []
    for key, label, colour, fn in cats:
        scored = sorted(range(n), key=lambda i: -fn(i))
        top = [i for i in scored if fn(i) > -0.5][:4]
        out.append((key, label, colour, top))
    return out


def build_bestbets_classic(pred: Prediction) -> str:
    rs = pred.runners
    feed = json.loads((DATA / "form_feed_2026.json").read_text()) if (DATA / "form_feed_2026.json").exists() else {}
    analysis = json.loads((DATA / "trainer_analysis.json").read_text()).get("trainers", {}) if (DATA / "trainer_analysis.json").exists() else {}
    order = list(np.argsort(-pred.win_prob))
    e = html.escape
    blocks = []
    for i in sorted(range(len(rs)), key=lambda i: rs[i]["no"]):
        r = rs[i]
        f = feed.get(r["horse"], {})
        runs = f.get("form_last12", [])
        st = f.get("stats", {})
        blink = "*" if "blink" in (r.get("gear") or "").lower() else ""
        stars = _stars(pred.win_prob[i])
        form_rows = ""
        for x in runs[:5]:
            fin = f"{_e(x.get('finish'))}-{_e(x.get('field'))}"
            jwb = " ".join(s for s in [_e(x.get("jockey")) if x.get("jockey") else "", f"{x['weight']}" if x.get("weight") else "", f",{x['barrier']}" if x.get("barrier") else ""] if s)
            form_rows += (f"<tr><td class='fin'>{fin}</td><td>{_e(TRACK_CODE.get(x.get('track'), (x.get('track') or '-')[:4].upper()))}</td><td class='num'>{_e(x.get('dist'))}</td>"
                          f"<td>{_e(x.get('race'))} <small>{_e(x.get('class'))}</small></td><td class='num'>-</td><td>{_e(x.get('date'))}</td><td>{_going_letter(x.get('going'))}</td>"
                          f"<td>{jwb or '-'}</td><td>{_e(x.get('sp'))}</td><td>{_e(x.get('winner'))}</td><td class='num'>{_e(x.get('pos_800'))}/{_e(x.get('pos_400'))}</td>"
                          f"<td>{_e(x.get('time'))}</td><td>{_e(x.get('margin'))}</td><td class='cmt'>{_e(x.get('in_running'))}</td></tr>")
        blocks.append(f"""
<article class="bb r{stars}">
  <div class="hdr"><span class="tab">{r['no']}</span><span class="nm">{e(r['horse']).upper()}{blink}</span> <span class="flags">{tcbdwh(r, runs)}</span>
    <span class="starts">{_e(f.get('career'))}</span> <span class="prize">{_e(f.get('prizemoney'))}</span>
    <span class="jky">{e(r['jockey']).upper()} ({r['barrier']})</span> <span class="wt">{r['weight']}</span>
    <span class="trn">{e(r['trainer']).upper()} ({e(TRAINER_LOC.get(r['trainer'], ''))})</span> <span class="stars">{'★' * stars}{'☆' * (5 - stars)}</span></div>
  <div class="body">
    {silks_svg(r['silks'])}
    <div class="mid">
      <div class="ped">{_e(f.get('colour_sex_age'))} {_e(f.get('sire_dam'))} - {_e(f.get('owner') or r['silks'].get('desc'))}</div>
      <div class="grid">
        <span><b>Track</b> {_e(st.get('track'))}</span><span><b>Dist</b> {_e(st.get('distance'))}</span><span><b>Tk/Dist</b> {_e(st.get('track_distance'))}</span><span><b>WDR</b> {_e(st.get('wdr'))}</span><span><b>W&amp;P%</b> {_e(st.get('wp'))}</span><span><b>1st-up</b> {_e(st.get('first_up'))}</span><span class="wide"><b>Last 15</b> {last15(runs)}</span>
        <span><b>Firm</b> {_e(st.get('firm'))}</span><span><b>Good</b> {_e(st.get('good'))}</span><span><b>Soft</b> {_e(st.get('soft'))}</span><span><b>Heavy</b> {_e(st.get('heavy'))}</span><span><b>Synth</b> {_e(st.get('synthetic'))}</span><span><b>2nd-up</b> {_e(st.get('second_up'))}</span><span class="wide"><b>Model</b> {100 * pred.win_prob[i]:.1f}% win · {100 * pred.top4_prob[i]:.1f}% top-4 · ${r['odds']:.1f}</span>
      </div>
    </div>
  </div>
  <div class="scroll"><table class="form"><thead><tr><th>Plc</th><th>Trk</th><th>Dist</th><th>Race &amp; type</th><th>Lim</th><th>Date</th><th>Cond</th><th>Jky Wt Bar</th><th>Odds</th><th>Winner/2nd</th><th>800/400</th><th>Time</th><th>Mgn</th><th>Comment</th></tr></thead><tbody>{form_rows}</tbody></table></div>
  {trainer_block(pred, r, analysis)}
</article>""")

    # colour-coded 1-4 per factor
    cats = factor_rankings(pred, feed)
    cat_html = ""
    for key, label, colour, top in cats:
        items = "".join(f'<li><span class="pick" style="background:{colour}">{k + 1}</span> <b>{e(rs[i]["horse"])}</b> ({rs[i]["barrier"]}) <small>No.{rs[i]["no"]}</small></li>' for k, i in enumerate(top))
        cat_html += f'<div class="cat" style="border-top-color:{colour}"><h3 style="color:{colour}">{e(label)}</h3><ol>{items}</ol></div>'
    counts = {}
    for _, _, _, top in cats:
        for i in top:
            counts[i] = counts.get(i, 0) + 1
    consensus = sorted(counts.items(), key=lambda kv: (-kv[1], -pred.win_prob[kv[0]]))[:6]
    cons_html = ", ".join(f"<b>{e(rs[i]['horse'])}</b> ({c} of 7)" for i, c in consensus)
    t = [rs[i] for i in order[:4]]
    generated = dt.datetime.now().strftime("%d %b %Y %H:%M")
    return f"""<title>Cup Best Bets Form</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Source+Sans+3:wght@400;600;700&display=swap">
<style>
/* Layout: Best Bets newspaper form, phone-first; cream runner header band, red name, yellow stats band, five form lines, trainer block, colour-coded picks. */
:root{{--bg:#fbfaf6;--fg:#111;--muted:#555;--rule:#d9d4c7;--band:#fdf1c7;--band2:#fff8dc;--red:#c8102e;--green:#0f4d3a;--gold:#c99a2e;--panel:#fff;
--display:"Barlow Condensed","Arial Narrow",Impact,sans-serif;--body:"Source Sans 3","Segoe UI",Helvetica,Arial,sans-serif}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#121412;--fg:#ecece6;--muted:#a9a99f;--rule:#33342e;--band:#3a3214;--band2:#2a2611;--red:#ff6b7a;--green:#0b3b2d;--panel:#1b1d1a;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#121412;--fg:#ecece6;--muted:#a9a99f;--rule:#33342e;--band:#3a3214;--band2:#2a2611;--red:#ff6b7a;--green:#0b3b2d;--panel:#1b1d1a;color-scheme:dark}}
*{{box-sizing:border-box}} body{{background:var(--bg);color:var(--fg);font-family:var(--body);font-size:13px;line-height:1.3;margin:0}}
.page{{max-width:1100px;margin:0 auto;padding-block:0 24px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}}
header.mast{{background:var(--green);color:#fff;margin:0 -16px;padding:14px max(16px,env(safe-area-inset-left,0px)) 12px max(16px,env(safe-area-inset-right,0px));border-bottom:5px solid var(--gold)}}
.mast h1{{font-family:var(--display);font-weight:800;font-size:clamp(26px,5vw,40px);margin:0;line-height:.95}} .eyebrow{{font-family:var(--display);letter-spacing:.14em;text-transform:uppercase;font-size:12px;opacity:.85}}
h2{{font-family:var(--display);font-weight:800;font-size:24px;margin:18px 0 6px;text-transform:uppercase}}
.bb{{border:1px solid var(--rule);margin:0 0 12px;background:var(--panel);break-inside:avoid}}
.hdr{{background:var(--band);padding:6px 10px;display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 12px;font-family:var(--display);font-size:15px}}
.tab{{color:var(--red);font-weight:800;font-size:26px;line-height:1}} .nm{{color:var(--red);font-weight:800;font-size:24px;letter-spacing:.01em}} .flags{{font-family:var(--body);font-size:12px;color:var(--muted)}}
.starts,.prize,.wt{{font-weight:700}} .jky,.trn{{font-weight:700}} .stars{{color:var(--gold);font-family:var(--body);font-size:14px;margin-left:auto}}
.body{{display:flex;gap:10px;padding:6px 10px;align-items:flex-start}} .body .silks{{flex:0 0 auto;width:58px;height:68px;object-fit:contain}} .mid{{flex:1 1 0;min-width:0}}
.ped{{color:var(--red);font-size:11.5px;font-weight:600;margin-bottom:4px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:2px 8px;background:var(--band2);padding:4px 6px;font-size:11.5px;font-variant-numeric:tabular-nums}} .grid span b{{font-family:var(--display);text-transform:uppercase;font-size:11px;margin-right:3px}} .grid .wide{{grid-column:span 2}}
.scroll{{overflow-x:auto}} table.form{{border-collapse:collapse;width:100%;font-size:11.5px;font-variant-numeric:tabular-nums}}
table.form th{{font-family:var(--display);text-transform:uppercase;letter-spacing:.05em;font-size:10.5px;color:var(--muted);text-align:left;padding:3px 5px;border-bottom:1px solid var(--rule);white-space:nowrap}}
table.form td{{padding:3px 5px;border-bottom:1px dotted var(--rule);vertical-align:top;white-space:nowrap}} td.fin{{font-weight:700}} td.num{{text-align:right}} td.cmt{{white-space:normal;min-width:160px;color:var(--muted)}} table.form small{{color:var(--muted)}}
.tblock{{background:var(--band2);border-top:2px solid var(--gold);padding:6px 10px 8px;font-size:12px}} .tline{{font-family:var(--display);font-size:16px;margin-bottom:4px}} .tstars{{color:var(--gold);font-family:var(--body);font-size:14px}} .tline small{{font-family:var(--body);color:var(--muted);font-size:11px}} .tblock p{{margin:0 0 4px}}
.cats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}} .cat{{background:var(--panel);border:1px solid var(--rule);border-top:6px solid;padding:8px 10px}} .cat h3{{font-family:var(--display);text-transform:uppercase;font-size:15px;margin:0 0 4px;letter-spacing:.06em}} .cat ol{{list-style:none;margin:0;padding:0}} .cat li{{margin:3px 0}}
.pick{{display:inline-grid;place-items:center;width:22px;height:22px;border-radius:50%;color:#fff;font-family:var(--display);font-weight:800}}
.key{{font-size:11.5px;color:var(--muted);margin:6px 0 10px;max-width:120ch}}
footer{{margin-top:16px;color:var(--muted);font-size:11px;border-top:1px solid var(--rule);padding-top:8px}}
@page{{size:A4 portrait;margin:9mm}} @media print{{:root{{--bg:#fff}} body{{font-size:10px;-webkit-print-color-adjust:exact;print-color-adjust:exact}} .page{{max-width:none;padding:0}} header.mast{{margin:0}} .scroll{{overflow:visible}} table.form{{font-size:9px}} td.cmt{{min-width:0}}}}
</style>
<div class="page">
<header class="mast"><div class="eyebrow">Flemington · Race 7 · 3:00pm · Tue 3 Nov 2026 · 3200m · Group 1 handicap · {e(pred.race['expected_track'])}</div><h1>Lexus Melbourne Cup best bets form</h1></header>
<p class="key">Reading the form: TAB no., name (* blinkers), (t) won on track, (d) won at distance, (c) won course and distance, (w) won on heavy, (b) beaten favourite last start; starts 1st-2nd-3rd; jockey (barrier) weight; trainer (location). Stats: Track, Dist, Tk/Dist, WDR winning distance range, W&amp;P% win and place, 1st-up, Last 15 (✦ = spell), Firm/Good/Soft/Heavy/Synth, 2nd-up. Form lines are the last five starts: placing-starters, track, distance, race and type, limit weight, date, condition (F/G/S/H), jockey weight barrier, odds (F = favourite), winner or runner-up, position at 800m/400m, time, margin, comment. A dash means the figure was not available from published sources.</p>
{''.join(blocks)}
<h2>Top four, colour-coded by factor</h2>
<p class="key">Each colour is a different way of ranking the field. Green is the full model. Blue, purple and teal are wet, heavy and soft-track form from each horse's record. Gold is the best of the horses that have only ever raced in Australia or New Zealand. Red is the best of those coming through a traditional Cup lead-up (Caulfield Cup, Cox Plate, Moonee Valley Cup, Geelong Cup, Archer, Bart Cummings, Herbert Power). Grey is the 2011-2025 trend profile: age 4-6, 51-56.5kg, barrier 19 or inside, not a mare, a traditional or European direct lead-up, a genuinely run last start, placed last start, has run 3200m.</p>
<div class="cats">{cat_html}</div>
<p class="key"><b>Appears in the most colours:</b> {cons_html}.</p>
<p class="key"><b>Model first four:</b> {' - '.join(str(x['no']) for x in t)} ({', '.join(e(x['horse']) for x in t)}).</p>
<footer>Generated {generated}. Form compiled from published results; silks from the ABC Melbourne Cup sweep grid. Research only. Gamble responsibly.</footer>
</div>
"""
