#!/usr/bin/env python3
"""Run the Melbourne Cup first-four predictor and write a markdown report.

Examples:
  python scripts/predict_melbourne_cup.py
  python scripts/predict_melbourne_cup.py --market-weight 0.6 --sims 50000 --out docs/MELBOURNE_CUP_2026.md
  python scripts/predict_melbourne_cup.py --json predictions.json
  python scripts/predict_melbourne_cup.py --track soft6 --bias on-pace --form-guide docs/MELBOURNE_CUP_2026_FORM_GUIDE.md
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from melbourne_cup.formguide import build_form_guide  # noqa: E402
from melbourne_cup.model import Weights, predict, scenario_table  # noqa: E402
from melbourne_cup.report import build_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Melbourne Cup 2026 win / first-four predictor")
    ap.add_argument("--market-weight", type=float, default=0.75)
    ap.add_argument("--form-weight", type=float, default=0.30)
    ap.add_argument("--pattern-weight", type=float, default=0.30)
    ap.add_argument("--slow-leadup-penalty", type=float, default=0.30)
    ap.add_argument("--track", default="forecast", choices=["forecast", "good4", "soft5", "soft6", "soft7", "heavy"],
                    help="override the going distribution in the race file")
    ap.add_argument("--bias", default="neutral", choices=["neutral", "on-pace", "closers"],
                    help="Cup Day track bias from the earlier races: are on-speed horses or swoopers winning?")
    ap.add_argument("--bias-strength", type=float, default=0.35)
    ap.add_argument("--going-scale", type=float, default=0.45)
    ap.add_argument("--form-guide", default=None, help="write the multi-variable form guide here")
    ap.add_argument("--no-scenarios", action="store_true", help="skip the going / bias scenario table")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--field", default="field_2026.json", help="field file inside melbourne_cup/cup_data")
    ap.add_argument("--out", default=None, help="write markdown report here")
    ap.add_argument("--json", default=None, help="write machine-readable predictions here")
    args = ap.parse_args()

    w = Weights(market=args.market_weight, form=args.form_weight, pattern=args.pattern_weight,
                slow_leadup_penalty=args.slow_leadup_penalty, going_scale=args.going_scale,
                bias=args.bias, bias_strength=args.bias_strength, track=args.track)
    pred = predict(weights=w, sims=args.sims, seed=args.seed, field_file=args.field)
    scenarios = None if args.no_scenarios else scenario_table(w, sims=4000)
    report = build_report(pred, scenarios)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"report written to {args.out}")
    else:
        print(report)
    if args.form_guide:
        Path(args.form_guide).write_text(build_form_guide(pred), encoding="utf-8")
        print(f"form guide written to {args.form_guide}")
    if args.json:
        payload = {
            "race": pred.race,
            "runners": [
                {"no": r["no"], "horse": r["horse"], "odds": r["odds"], "win_prob": float(pred.win_prob[i]),
                 "place_prob": float(pred.place_prob[i]), "top4_prob": float(pred.top4_prob[i]),
                 "components": pred.components[i]}
                for i, r in enumerate(pred.runners)
            ],
            "first_four_exact": [{"order": list(c), "prob": p} for c, p in pred.first_four_exact],
            "first_four_boxed": [{"set": sorted(c), "prob": p} for c, p in pred.first_four_boxed],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"json written to {args.json}")


if __name__ == "__main__":
    main()
