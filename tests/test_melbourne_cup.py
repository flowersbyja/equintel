import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from melbourne_cup.features import tempo_adjustment, form_rating  # noqa: E402
from melbourne_cup.model import predict, simulate_first_four, derive_leadup_group  # noqa: E402
from melbourne_cup.patterns import ArchetypeIndex, load_json, similarity  # noqa: E402


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
