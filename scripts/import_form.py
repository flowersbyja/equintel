#!/usr/bin/env python3
"""Import a form-provider export (CSV or JSON) into melbourne_cup/cup_data/form_feed_2026.json.

Accepts a CSV with one row per past run and at least the columns
  horse,date,track,race,class,dist,going,finish,field,margin,weight,barrier,jockey,sp,winner,second,time,last600,pos_800,pos_400
(extra columns are kept), or a JSON file already in the form_feed_schema shape. Runner-level fields
(career, stats grids, colours, sire_dam) can be supplied in a second CSV with --runners.

  python scripts/import_form.py runs.csv --runners runners.csv
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "melbourne_cup" / "cup_data" / "form_feed_2026.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", help="CSV of past runs (one row per run) or JSON in schema shape")
    ap.add_argument("--runners", help="CSV of runner-level fields (horse, career, colours, sire_dam, stats_* ...)")
    ap.add_argument("--max-runs", type=int, default=12)
    args = ap.parse_args()

    feed = {}
    if OUT.exists():
        feed = json.loads(OUT.read_text())
    src = Path(args.runs)
    if src.suffix.lower() == ".json":
        feed.update(json.loads(src.read_text()))
    else:
        runs = defaultdict(list)
        with open(src, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                horse = row.pop("horse").strip()
                for k in ("dist", "finish", "field", "barrier", "pos_800", "pos_400"):
                    if row.get(k) not in (None, ""):
                        try:
                            row[k] = int(float(row[k]))
                        except ValueError:
                            pass
                if row.get("weight"):
                    try:
                        row["weight"] = float(row["weight"])
                    except ValueError:
                        pass
                runs[horse].append(row)
        for horse, rs in runs.items():
            feed.setdefault(horse, {})["form_last12"] = rs[: args.max_runs]
    if args.runners:
        with open(args.runners, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                horse = row.pop("horse").strip()
                rec = feed.setdefault(horse, {})
                stats = {k[6:]: v for k, v in row.items() if k.startswith("stats_")}
                rec.update({k: v for k, v in row.items() if not k.startswith("stats_")})
                if stats:
                    rec.setdefault("stats", {}).update(stats)
    OUT.write_text(json.dumps(feed, indent=1, ensure_ascii=False))
    print(f"wrote {OUT} with {len(feed)} runners")


if __name__ == "__main__":
    main()
