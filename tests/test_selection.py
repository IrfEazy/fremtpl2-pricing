import json

import numpy as np
import pytest

from fremtpl2_pricing import selection as sel


def L(*d):  # incumbent losses are base, next = base - d
    base = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    return list(base), list(base - np.array(d))


def test_replace_when_mean_exceeds_threshold():
    a, b = L(1.0, 1.1, 0.9, 1.0, 1.0)
    t = sel.ladder({"simple": a, "complex": b})
    st = t["steps"][0]
    assert st["decision"] == "replace" and t["selected"] == "complex"
    assert st["mean_d"] == pytest.approx(1.0)
    assert st["threshold"] == pytest.approx(st["sd_d"] / np.sqrt(5))
    assert t["rule_version"] == sel.RULE_VERSION and t["n_folds"] == 5
    json.dumps(t)


def test_keep_when_mean_inside_threshold_is_inconclusive():
    a, b = L(1.0, -1.0, 1.0, -1.0, 0.5)  # mean 0.1, large sd
    t = sel.ladder({"simple": a, "complex": b})
    st = t["steps"][0]
    assert st["decision"] == "keep" and t["selected"] == "simple" and st["inconclusive"]


def test_keep_when_next_is_worse():
    a, b = L(-1.0, -1.0, -2.0, -1.0, -1.0)
    t = sel.ladder({"simple": a, "complex": b})
    assert t["selected"] == "simple" and not t["steps"][0]["inconclusive"]


def test_exact_tie_keeps_simpler():
    a, _ = L(0, 0, 0, 0, 0)
    t = sel.ladder({"simple": a, "complex": list(a)})
    st = t["steps"][0]
    assert st["decision"] == "keep" and "exact tie" in st["reason"] and not st["inconclusive"]


def test_zero_variance_positive_mean_replaces():
    a, b = L(0.5, 0.5, 0.5, 0.5, 0.5)
    st = sel.ladder({"s": a, "c": b})["steps"][0]
    assert st["sd_d"] == pytest.approx(0, abs=1e-12) and st["decision"] == "replace"


def test_zero_variance_zero_mean_keeps_and_negative_keeps():
    a, b = L(0, 0, 0, 0, 0)
    assert sel.ladder({"s": a, "c": b})["selected"] == "s"
    a, b = L(-0.5, -0.5, -0.5, -0.5, -0.5)
    assert sel.ladder({"s": a, "c": b})["selected"] == "s"


def test_comparisons_are_against_the_incumbent_not_the_best():
    base = [10.0, 10.0, 10.0, 10.0, 10.0]
    # r2 is better than r1 by a noisy 0.3 (kept), r3 is better than r2 but only 0.2 less than r1
    r1 = [9.0, 9.1, 8.9, 9.0, 9.0]  # beats base clearly
    r2 = [8.7, 8.9, 8.6, 8.9, 8.5]  # mean 0.28 better than r1, d = r1 - r2
    r3 = [8.6, 8.8, 8.4, 8.9, 8.4]
    t = sel.ladder({"c": base, "r1": r1, "r2": r2, "r3": r3})
    assert [s["incumbent"] for s in t["steps"]] == [
        "c",
        t["steps"][1]["incumbent"],
        t["steps"][2]["incumbent"],
    ]
    assert t["steps"][0]["decision"] == "replace"
    for s in t["steps"][1:]:
        incumbent = s["incumbent"]
        assert incumbent in ("r1", "r2")
    # each step's d_k is incumbent - next, with the incumbent as of that step
    inc = "c"
    losses = {"c": base, "r1": r1, "r2": r2, "r3": r3}
    for s in t["steps"]:
        assert s["incumbent"] == inc
        assert s["d_k"] == pytest.approx(list(np.array(losses[inc]) - np.array(losses[s["next"]])))
        if s["decision"] == "replace":
            inc = s["next"]
    assert t["selected"] == inc


def test_incumbent_not_best_sequential():
    # the third rung has the lowest mean loss but is only compared with the incumbent (rung 1)
    r1 = [10.0, 10.0, 10.0, 10.0, 10.0]
    r2 = [10.0, 10.0, 10.0, 10.0, 10.0]  # tie -> kept simpler
    r3 = [9.9, 10.1, 9.7, 10.2, 9.8]  # lower mean (9.94) but noisy: inside the threshold
    t = sel.ladder({"a": r1, "b": r2, "c": r3})
    assert t["steps"][1]["incumbent"] == "a" and t["selected"] == "a"
    assert np.mean(r3) < np.mean(r1)


def test_tweedie_wins_clearly():
    fxs = [10.0, 11.0, 12.0, 13.0, 14.0]
    tw = [9.0, 10.1, 10.9, 12.0, 13.0]
    t = sel.freq_x_sev_vs_tweedie(fxs, tw)
    st = t["steps"][0]
    assert t["selected"] == "tweedie" and not st["inconclusive"] and st["mean_d"] < -st["threshold"]


def test_tweedie_lower_inside_heuristic_keeps_freq_x_sev():
    fxs = [10.0, 11.0, 12.0, 13.0, 14.0]
    tw = [9.0, 12.0, 11.0, 13.5, 13.4]
    t = sel.freq_x_sev_vs_tweedie(fxs, tw)
    st = t["steps"][0]
    assert np.mean(tw) < np.mean(fxs) and st["mean_d"] < 0
    assert t["selected"] == "freq_x_sev" and st["inconclusive"]


def test_freq_x_sev_wins_clearly_and_ties():
    fxs = [10.0, 11.0, 12.0, 13.0, 14.0]
    tw = [11.0, 12.1, 12.9, 14.0, 15.0]
    t = sel.freq_x_sev_vs_tweedie(fxs, tw)
    assert t["selected"] == "freq_x_sev" and not t["steps"][0]["inconclusive"]
    t = sel.freq_x_sev_vs_tweedie(fxs, list(fxs))
    assert t["selected"] == "freq_x_sev" and "exact tie" in t["steps"][0]["reason"]


@pytest.mark.parametrize(
    "bad",
    [
        {"a": [1, 2, 3], "b": [1, 2]},  # mismatched fold counts
        {"a": [1, np.nan, 3], "b": [1, 2, 3]},  # non-finite
        {"a": [1, 2, 3], "b": [1, np.inf, 3]},
        {"a": [1.0], "b": [2.0]},  # one fold
        {"a": [], "b": []},  # missing folds
        {"a": [1, 2, 3]},  # single candidate
    ],
)
def test_malformed_ladder_raises(bad):
    with pytest.raises(ValueError):
        sel.ladder(bad)


def test_malformed_step6_raises():
    with pytest.raises(ValueError):
        sel.freq_x_sev_vs_tweedie([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        sel.freq_x_sev_vs_tweedie([1, 2, np.nan], [1, 2, 3])
    with pytest.raises(ValueError):
        sel.freq_x_sev_vs_tweedie([1.0], [2.0])
