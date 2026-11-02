"""
Preparation-formula similarity ("does this horse look like a past Cup winner's prep?").

For each 2026 runner we score its similarity to every historical first-four horse (2000-2025) across:
  lead-up race group (with cross-archetype relatedness), days since the last run, number of Australian
  prep runs, origin, age, weight, barrier, running style, final lead-up finish, sex.

Each historical template contributes  conf x result_weight x similarity x trainer_multiplier  where

  * result_weight     : 1st 1.0, 2nd 0.55, 3rd 0.4, 4th 0.3
  * trainer_multiplier: 1.0 when the template is the SAME trainer's own past Cup prep ("own formula"),
                        CROSS_TRAINER_CREDIT (default 0.65) when a different trainer is using the same style.

The output per runner is a pattern score plus the named templates it most resembles, so the report can
say e.g. "Point Of Law: 0.78 similar to Dunaden 2011 (Geelong Cup winner, first Australian run)".
"""
from __future__ import annotations

import json
import math
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "cup_data"

RESULT_WEIGHT = {1: 1.0, 2: 0.55, 3: 0.4, 4: 0.3}
CROSS_TRAINER_CREDIT = 0.65
TOP_N_TEMPLATES = 5

FEATURE_WEIGHTS = {
    "leadup_group": 2.4,
    "days_since": 1.1,
    "aus_prep_runs": 0.8,
    "origin": 1.0,
    "age": 0.9,
    "weight": 1.0,
    "barrier": 0.4,
    "style": 0.9,
    "leadup_finish": 1.0,
    "sex": 0.3,
}

STYLE_ORDER = {"LEADER": 0, "ON_PACE": 1, "MIDFIELD": 2, "BACKMARKER": 3}


def load_json(name: str) -> dict:
    with open(DATA_DIR / name, encoding="utf-8") as fh:
        return json.load(fh)


class ArchetypeIndex:
    def __init__(self, archetypes_doc: dict):
        self.archetypes = archetypes_doc["archetypes"]
        self.group_to_arch = {}
        for arch_id, arch in self.archetypes.items():
            for g in arch["leadup_groups"]:
                self.group_to_arch[g] = arch_id
        self.related = {}
        for a, b, w in archetypes_doc.get("related_archetypes", []):
            self.related[(a, b)] = w
            self.related[(b, a)] = w
        self.aliases = archetypes_doc.get("trainer_aliases", {})

    def archetype_of(self, leadup_group: str) -> str | None:
        return self.group_to_arch.get(leadup_group)

    def group_similarity(self, g1: str, g2: str) -> float:
        if g1 == g2:
            return 1.0
        a1, a2 = self.archetype_of(g1), self.archetype_of(g2)
        if a1 is None or a2 is None:
            return 0.0
        if a1 == a2:
            return 0.75
        return self.related.get((a1, a2), 0.0)

    def canonical_trainer(self, name: str) -> str:
        return self.aliases.get(name, name)


def _gauss(diff: float, scale: float) -> float:
    return math.exp(-0.5 * (diff / scale) ** 2)


def _origin_similarity(o1: str, o2: str) -> float:
    if o1 == o2:
        return 1.0
    flyins = {"FLYIN_DIRECT", "FLYIN_RUN"}
    locals_ = {"LOCAL", "SETTLED_IMPORT"}
    if {o1, o2} <= flyins or {o1, o2} <= locals_:
        return 0.55
    if "SETTLED_IMPORT" in (o1, o2) and ({o1, o2} & flyins):
        return 0.3
    return 0.0


def similarity(runner: dict, template: dict, idx: ArchetypeIndex) -> tuple[float, dict]:
    r_group = runner["leadup_group"]
    parts = {
        "leadup_group": idx.group_similarity(r_group, template["leadup_group"]),
        # days-since: compare on a log scale so 3 vs 6 days differs more than 50 vs 60
        "days_since": _gauss(math.log(runner["days_since"] + 1) - math.log(template["days_since"] + 1), 0.45),
        "aus_prep_runs": _gauss(runner["aus_prep_runs"] - template["aus_prep_runs"], 1.6),
        "origin": _origin_similarity(runner["origin"], template["origin"]),
        "age": _gauss(runner["age"] - template["age"], 1.5),
        "weight": _gauss(runner["weight"] - template["weight"], 2.2),
        "barrier": _gauss(runner["barrier"] - template["barrier"], 7.0),
        "style": _gauss(STYLE_ORDER[runner["style"]] - STYLE_ORDER[template["style"]], 1.1),
        "leadup_finish": _gauss(math.log(runner["leadups"][0]["finish"]) - math.log(template["leadup_finish"]), 0.8),
        "sex": 1.0 if runner["sex"] == template["sex"] else (0.7 if {runner["sex"], template["sex"]} <= {"G", "H"} else 0.3),
    }
    total_w = sum(FEATURE_WEIGHTS.values())
    score = sum(FEATURE_WEIGHTS[k] * v for k, v in parts.items()) / total_w
    return score, parts


def pattern_scores(runners: list[dict], history: list[dict], idx: ArchetypeIndex, top_k: int = 4) -> dict[str, dict]:
    out = {}
    for r in runners:
        r_trainer = idx.canonical_trainer(r["trainer"])
        contributions = []
        for t in history:
            sim, parts = similarity(r, t, idx)
            same_trainer = idx.canonical_trainer(t["trainer"]) == r_trainer
            mult = 1.0 if same_trainer else CROSS_TRAINER_CREDIT
            contrib = t.get("confidence", 0.8) * RESULT_WEIGHT.get(t["pos"], 0.2) * (sim ** 2.2) * mult
            contributions.append({
                "year": t["year"], "horse": t["horse"], "pos": t["pos"], "trainer": t["trainer"],
                "archetype": idx.archetype_of(t["leadup_group"]), "similarity": round(sim, 3),
                "same_trainer": same_trainer, "contribution": contrib,
                "top_feature_matches": sorted(parts.items(), key=lambda kv: -kv[1])[:3],
            })
        contributions.sort(key=lambda c: -c["contribution"])
        # Sum only the best TOP_N matches so a crowded archetype (many Caulfield Cup templates) does not
        # outscore a tight match to a sparse one (a single Bart Cummings template).
        best = contributions[:TOP_N_TEMPLATES]
        total = sum(c["contribution"] for c in best)
        own = sum(c["contribution"] for c in best if c["same_trainer"])
        winners_only = sum(c["contribution"] for c in best if c["pos"] == 1)
        out[r["horse"]] = {
            "pattern_total": total,
            "own_formula_share": (own / total) if total > 0 else 0.0,
            "winner_template_share": (winners_only / total) if total > 0 else 0.0,
            "archetype": idx.archetype_of(r["leadup_group"]),
            "best_matches": contributions[:top_k],
        }
    return out
