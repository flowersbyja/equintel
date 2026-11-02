"""
Preparation-formula similarity ("does this horse look like a past Cup winner's prep?").

For each 2026 runner we score its similarity to every historical first-four horse (2000-2025) across:
  lead-up race group (with cross-archetype relatedness), days since the last run, number of Australian
  prep runs, origin, age, weight, barrier, running style, final lead-up finish, sex.

Each historical template contributes  conf x result_weight x similarity x trainer_multiplier  where

  * result_weight     : 1st 1.0, 2nd 0.55, 3rd 0.4, 4th 0.3
  * trainer_multiplier: 1.0 when the template is the SAME trainer's own past Cup prep ("own formula");
                        otherwise 0.35 + 0.65 x trainer-STYLE similarity, where style is the trainer's methodology
                        profile (cup_data/trainer_styles.json) blended 40/60 with how this particular horse was
                        actually campaigned. So a Rekindling-shaped horse only inherits Rekindling credit in
                        proportion to how O'Brien-like its trainer's method (and this campaign) really is.

The output per runner is a pattern score plus the named templates it most resembles, so the report can
say e.g. "Point Of Law: 0.78 similar to Dunaden 2011 (Geelong Cup winner, first Australian run)".
"""
from __future__ import annotations

import json
import math
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "cup_data"

RESULT_WEIGHT = {1: 1.0, 2: 0.55, 3: 0.4, 4: 0.3}
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
FORWARD_SCORE = {"LEADER": 1.0, "ON_PACE": 0.75, "MIDFIELD": 0.4, "BACKMARKER": 0.1}
TICKET_GROUPS = {"GEELONG_CUP", "MV_CUP", "BART_CUMMINGS", "LEXUS_HOTHAM"}
# How much the cross-trainer credit depends on methodology similarity:
#   multiplier = CROSS_BASE + (1 - CROSS_BASE) * trainer_style_similarity   (same trainer -> 1.0)
CROSS_BASE = 0.35


def _fresh_from_days(days: float) -> float:
    if days <= 10:
        return 0.1
    if days <= 17:
        return 0.3
    if days <= 35:
        return 0.6
    return 0.9


def campaign_vector(h: dict, dims: list[str], trainer_profile: dict) -> dict:
    """Describe how THIS horse was actually campaigned, on the same dimensions as the trainer profiles.
    Works for a 2026 runner (has 'leadups') or a historical template (has leadup_group / aus_prep_runs ...)."""
    origin = h["origin"]
    aus_runs = h["aus_prep_runs"]
    euro_base = 1.0 if origin == "FLYIN_DIRECT" else 0.8 if origin == "FLYIN_RUN" else (0.6 if (origin == "SETTLED_IMPORT" and aus_runs <= 1) else 0.0)
    if "leadups" in h:
        sharpener = 1.0 if any(lu.get("dist", 2400) <= 2100 for lu in h["leadups"][:3]) else 0.0
    else:
        sharpener = 1.0 if h["leadup_group"] in ("COX_PLATE", "TURNBULL", "MACKINNON") else 0.0
    group = h["leadup_group"]
    ticket = 1.0 if (h.get("golden_ticket") or group in TICKET_GROUPS) else 0.0
    young = 0.7 * (1.0 if h["age"] <= 4 else 0.0) + 0.3 * (1.0 if h["weight"] <= 53.5 else 0.0)
    vec = {
        "fresh": _fresh_from_days(h["days_since"]),
        "aus_runs": min(aus_runs, 5) / 5.0,
        "euro_base": euro_base,
        "wfa_sharpener": sharpener,
        "forward": FORWARD_SCORE[h["style"]],
        "young_light": young,
        "ticket_target": ticket,
        "data_driven": trainer_profile.get("data_driven", 0.4),
    }
    return {d: vec[d] for d in dims}


def blended_style(h: dict, idx: "ArchetypeIndex", campaign_weight: float = 0.6) -> dict:
    prof = idx.trainer_profile(h["trainer"])
    camp = campaign_vector(h, idx.dims, prof)
    return {d: (1 - campaign_weight) * prof[d] + campaign_weight * camp[d] for d in idx.dims}


def style_similarity(v1: dict, v2: dict) -> float:
    """1 - mean absolute difference across the methodology dimensions, in [0, 1]."""
    dims = list(v1.keys())
    return 1.0 - sum(abs(v1[d] - v2[d]) for d in dims) / len(dims)


def load_json(name: str) -> dict:
    with open(DATA_DIR / name, encoding="utf-8") as fh:
        return json.load(fh)


class ArchetypeIndex:
    def __init__(self, archetypes_doc: dict, trainer_styles_doc: dict | None = None):
        self.archetypes = archetypes_doc["archetypes"]
        ts = trainer_styles_doc or load_json("trainer_styles.json")
        self.dims = ts["dimensions"]
        self.trainer_styles = ts["trainers"]
        self.default_style = ts["default"]
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

    def trainer_profile(self, name: str) -> dict:
        for key in (name, self.canonical_trainer(name)):
            if key in self.trainer_styles:
                return self.trainer_styles[key]["style"]
        return dict(self.default_style)

    def trainer_record(self, name: str) -> dict:
        for key in (name, self.canonical_trainer(name)):
            if key in self.trainer_styles:
                return self.trainer_styles[key].get("cup_record", {"runners": 0, "wins": 0, "placings": 0})
        return {"runners": 0, "wins": 0, "placings": 0}

    def trainer_method(self, name: str) -> str:
        for key in (name, self.canonical_trainer(name)):
            if key in self.trainer_styles:
                return self.trainer_styles[key].get("method", "")
        return "No methodology profile on file."


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
    template_styles = {id(t): blended_style(t, idx) for t in history}
    for r in runners:
        r_trainer = idx.canonical_trainer(r["trainer"])
        r_style = blended_style(r, idx)
        contributions = []
        for t in history:
            sim, parts = similarity(r, t, idx)
            same_trainer = idx.canonical_trainer(t["trainer"]) == r_trainer
            tsim = style_similarity(r_style, template_styles[id(t)])
            mult = 1.0 if same_trainer else CROSS_BASE + (1.0 - CROSS_BASE) * tsim
            contrib = t.get("confidence", 0.8) * RESULT_WEIGHT.get(t["pos"], 0.2) * (sim ** 2.2) * mult
            contributions.append({
                "year": t["year"], "horse": t["horse"], "pos": t["pos"], "trainer": t["trainer"],
                "archetype": idx.archetype_of(t["leadup_group"]), "similarity": round(sim, 3),
                "trainer_style_similarity": round(tsim, 3), "trainer_multiplier": round(mult, 3),
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
        # how far this horse's actual campaign departs from its trainer's normal method
        prof = idx.trainer_profile(r["trainer"])
        camp = campaign_vector(r, idx.dims, prof)
        departure = 1.0 - style_similarity(prof, camp)
        out[r["horse"]] = {
            "pattern_total": total,
            "campaign_style": r_style,
            "trainer_profile": prof,
            "campaign_departure": departure,
            "own_formula_share": (own / total) if total > 0 else 0.0,
            "winner_template_share": (winners_only / total) if total > 0 else 0.0,
            "archetype": idx.archetype_of(r["leadup_group"]),
            "best_matches": contributions[:top_k],
        }
    return out
