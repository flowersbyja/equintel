import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from melbourne_cup.features import tempo_adjustment, form_rating  # noqa: E402
from melbourne_cup.model import predict, simulate_first_four, derive_leadup_group  # noqa: E402
from melbourne_cup.patterns import ArchetypeIndex, load_json, similarity, style_similarity, blended_style, campaign_vector  # noqa: E402
from melbourne_cup.going import going_adjustment, PRESETS  # noqa: E402
from melbourne_cup.pace import pace_adjustment  # noqa: E402


def test_tempo_rules_direction():
    up, _ = tempo_adjustment("BACKMARKER", "SLOW", finish=8, field=14)
    down, _ = tempo_adjustment("BACKMARKER", "SLOW", finish=1, field=14)
    assert up > 0 > down
    faded, _ = tempo_adjustment("ON_PACE", "FAST", finish=9, field=14)
    assert faded > 0
    soft_lead, _ = tempo_adjustment("LEADER", "SLOW", finish=1, field=14)
    assert soft_lead < 0


def test_sharpener_is_down_weighted():
    r = {"style": "BACKMARKER", "leadups": [
        {"race": "Crystal Mile", "dist": 1600, "finish": 11, "field": 11, "margin": 8.0, "tempo": "SLOW", "class": "G2"},
        {"race": "Herbert Power", "dist": 2400, "finish": 5, "field": 13, "margin": 0.8, "tempo": "SLOW", "class": "G2"},
    ]}
    f = form_rating(r)
    assert f["runs"][0]["weight"] < f["runs"][1]["weight"]


def test_self_similarity_is_one():
    idx = ArchetypeIndex(load_json("prep_archetypes.json"))
    t = load_json("historical_cups.json")["records"][0]
    runner = dict(t)
    runner["leadups"] = [{"finish": t["leadup_finish"]}]
    sim, _ = similarity(runner, t, idx)
    assert math.isclose(sim, 1.0, abs_tol=1e-9)


def test_leadup_groups():
    assert derive_leadup_group({"leadups": [{"race": "Caulfield Cup G1", "venue": "Caulfield"}]}) == "CAULFIELD_CUP"
    assert derive_leadup_group({"leadups": [{"race": "Kilternan Stakes G3", "venue": "Leopardstown"}]}) == "EURO_DIRECT"
    assert derive_leadup_group({"leadups": [{"race": "Tesio Stakes G3", "venue": "Flemington"}]}) == "OTHER_AUS"


def test_prediction_probabilities():
    pred = predict(sims=2000, seed=1)
    assert len(pred.runners) == 24
    assert math.isclose(pred.win_prob.sum(), 1.0, abs_tol=1e-9)
    assert math.isclose(pred.top4_prob.sum(), 4.0, abs_tol=1e-9)
    assert math.isclose(pred.place_prob.sum(), 3.0, abs_tol=1e-9)
    assert pred.win_prob.max() < 0.35


def test_plackett_luce_respects_strength():
    win = np.array([0.5, 0.3, 0.15, 0.05])
    place, top4, exact, boxed = simulate_first_four(win, list("abcd"), sims=3000, seed=0)
    assert place[0] > place[3]
    assert math.isclose(top4.sum(), 4.0, abs_tol=1e-9)


def test_trainer_style_similarity_bounds():
    idx = ArchetypeIndex(load_json("prep_archetypes.json"))
    a = idx.trainer_profile("Joseph O'Brien")
    b = idx.trainer_profile("Chris Waller")
    assert math.isclose(style_similarity(a, a), 1.0)
    assert 0.0 <= style_similarity(a, b) < 0.7  # fly-in method vs race-hardening method are far apart


def test_point_of_law_campaign_is_flyin_shaped():
    """Maher's Point Of Law was handled from Werribee quarantine with one run: its campaign vector should sit
    closer to Joseph O'Brien's method than Maher's general profile does."""
    idx = ArchetypeIndex(load_json("prep_archetypes.json"))
    field = load_json("field_2026.json")["runners"]
    pol = next(r for r in field if r["horse"] == "Point Of Law")
    pol["leadup_group"] = "GEELONG_CUP"
    maher = idx.trainer_profile("Ciaron Maher")
    job = idx.trainer_profile("Joseph O'Brien")
    camp = campaign_vector(pol, idx.dims, maher)
    assert style_similarity(camp, job) > style_similarity(maher, job)


def test_going_adjustment_direction():
    wet_lover = {"wet_rating": 1, "wet_record": {"starts": 6, "wins": 3, "seconds": 1, "thirds": 0}}
    firm_only = {"wet_rating": -1, "wet_record": None}
    a, _ = going_adjustment(wet_lover, PRESETS["soft6"])
    b, _ = going_adjustment(firm_only, PRESETS["soft6"])
    c, _ = going_adjustment(wet_lover, PRESETS["good4"])
    assert a > 0 > b and c == 0


def test_bias_switch_direction():
    leader = {"style": "LEADER"}
    closer = {"style": "BACKMARKER"}
    la, _ = pace_adjustment(leader, "moderate", "on-pace", 0.35, {})
    ca, _ = pace_adjustment(closer, "moderate", "on-pace", 0.35, {})
    assert la > 0 > ca
    lb, _ = pace_adjustment(leader, "moderate", "closers", 0.35, {})
    assert lb < 0
