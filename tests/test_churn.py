"""Churn diagnosis: information versus model sensitivity.

The tests that matter here are the ones asserting the diagnostic reaches the *right*
conclusion on constructed cases whose answer is known in advance. A diagnostic that
returns a plausible-sounding assessment regardless of input is worse than none, because
it would be believed at exactly the moment it is wrong.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from equity_monitor import churn, model, snapshots
from tests.test_snapshots import BASE_PERCENTILES, row


def snap(tmp_path, stamp, rows):
    snapshots.write(rows, str(tmp_path), on=stamp, as_of=stamp + "T23:00:00Z")
    return snapshots.read(snapshots.snapshot_path(str(tmp_path), stamp))


def rescore(rows):
    """Recompute conviction from each row's percentiles, leaving the ranks alone.

    Deliberately not ``model.score_rows``: that calls ``prepare()``, which recomputes
    percentiles from raw metric fields these fixtures do not carry, so every input would
    collapse to the imputed median and the test would measure nothing. ``score()`` is the
    pure function of percentiles that these cases are actually about.
    """
    out = []
    for r in rows:
        p = {k: v for k, v in r.items() if k.startswith("p_") and v is not None}
        p["drawdown_52w"] = r.get("drawdown_52w")
        out.append({**r, **model.score(p)})
    return out


def universe(n=60, **overrides):
    """A spread-out universe so ranks are well defined and ties are rare."""
    out = []
    for i in range(n):
        p = {k: min(0.98, max(0.02, v + (i - n / 2) * 0.012))
             for k, v in BASE_PERCENTILES.items()}
        p.update(overrides)
        out.append(row(f"S{i:03d}", **p))
    return out


def shift(rows, key, delta):
    """Move one input for every name, leaving the rest untouched."""
    out = []
    for r in rows:
        c = dict(r)
        c[key] = min(1.0, max(0.0, c[key] + delta))
        out.append(c)
    return rescore(out)


def interleaved(n=60):
    """Floored-quality and mid-quality names interleaved in the conviction ranking.

    Group A pins quality at the floor and compensates with strong momentum and risk;
    group B is mid-quality with weak momentum and risk. Both land in the same conviction
    range, which is what lets a small quality move reorder them.
    """
    rows = []
    for i in range(n):
        p = dict(BASE_PERCENTILES)
        step, floored = i // 2, i % 2 == 0
        for k in model.WEIGHTS["quality"]:
            p[k] = min(0.98, max(0.02, (0.02 if floored else 0.65) + step * 0.004))
        for k in ("p_rs", "p_trend", "p_liquidity", "p_value", "p_lowvol"):
            p[k] = min(0.98, max(0.02, (0.90 if floored else 0.25) - step * 0.004))
        rows.append(row(f"S{i:03d}", **p))
    return rescore(rows)


def nudge_quality(rows, delta=0.05):
    """Move only the floored group's quality, by less than the gap to the other group."""
    out = []
    for i, r in enumerate(rows):
        c = dict(r)
        if i % 2 == 0:
            for k in model.WEIGHTS["quality"]:
                c[k] = min(0.98, c[k] + delta)
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# effective weights
# ---------------------------------------------------------------------------
def test_effective_weights_sum_to_one_per_profile():
    """Each pillar is a third of d ln(conviction); the factor shares must close."""
    for profile in model.QUALITY_PROFILES:
        w = churn._effective_weights(profile)
        assert sum(w.values()) == pytest.approx(1.0), profile


def test_effective_weights_follow_the_sector_profile():
    """A bank's weights read ROE, not ROIC — attributing to an unread column is fiction."""
    bank = churn._effective_weights("Financials")
    assert "p_roe" in bank and "p_roic" not in bank
    default = churn._effective_weights("default")
    assert "p_roic" in default and "p_roe" not in default


# ---------------------------------------------------------------------------
# the movement ratio — the headline test
#
# Five constructed cases whose correct answer is known in advance.
# ---------------------------------------------------------------------------
def test_identical_snapshots_are_quiet_and_name_no_driver(tmp_path):
    rows = universe()
    a = snap(tmp_path, "2026-08-06", rows)
    b = snap(tmp_path, "2026-08-07", rows)
    got = churn.diagnose(a, b)
    assert got["input_movement"] == 0
    assert got["output_movement"] == 0
    assert got["assessment"] == "quiet"
    assert got["largest_contributor"] is None, \
        "named a leading contributor from a list of zeros"


def test_a_uniform_input_shift_does_not_read_as_amplification(tmp_path):
    """Moving every name's input by the same amount preserves the ordering.

    The inputs moved and the standings did not, so the ratio is near zero. This is the
    case that trips a diagnostic keyed on raw movement rather than on standing.
    """
    rows = universe()
    a = snap(tmp_path, "2026-08-06", rows)
    b = snap(tmp_path, "2026-08-07", shift(rows, "p_rs", 0.15))
    got = churn.diagnose(a, b)
    assert got["input_movement"] > 0
    assert got["amplification"] < 1.0
    assert got["assessment"] == "tracks-inputs"


def test_inverting_the_momentum_block_is_read_as_information(tmp_path):
    """The case that broke the first design of this module.

    Inverting the price-derived inputs — roughly 47% of the score by weight — reorders
    the board hard, and is unambiguously information. A correlation-based aggregate
    called it amplification, because a weighted mean of *signed* correlations collapses
    when inputs move in opposing directions. The movement ratio gets it right.
    """
    rows = universe()
    a = snap(tmp_path, "2026-08-06", rows)
    scrambled = []
    for r in rows:
        c = dict(r)
        for k in ("p_rs", "p_trend", "p_liquidity"):
            c[k] = min(0.98, max(0.02, 1.0 - c[k]))
        scrambled.append(c)
    b = snap(tmp_path, "2026-08-07", rescore(scrambled))
    got = churn.diagnose(a, b)
    assert got["output_movement"] > 0.2, "the board should have reordered substantially"
    assert got["amplification"] < churn.AMPLIFICATION_RATIO
    assert got["assessment"] == "tracks-inputs"


def test_amplification_is_detected_when_standing_outruns_the_inputs(tmp_path):
    """The case that actually matters, constructed so the answer is known in advance.

    A 0.05 nudge to the floored group's quality leaves every input's ordering untouched
    — that group stays below the other on quality throughout — but Q near its floor
    means the nudge moves their scores enough to leapfrog. Tiny input movement,
    wholesale reordering. If the diagnostic cannot separate this from information, it is
    of no use on the night it is needed.
    """
    rows = interleaved(60)
    a = snap(tmp_path, "2026-08-06", rows)
    b = snap(tmp_path, "2026-08-07", rescore(nudge_quality(rows)))
    got = churn.diagnose(a, b)
    assert got["input_movement"] < 0.02, "the construction is meant to barely move inputs"
    assert got["output_movement"] > 0.15
    assert got["amplification"] >= churn.AMPLIFICATION_RATIO, got
    assert got["assessment"] == "amplified"
    assert "specification rather than by the data" in got["basis"]


def test_the_input_that_drove_the_reordering_is_named(tmp_path):
    """"Which input caused it" is the first question after "was it real"."""
    rows = universe()
    a = snap(tmp_path, "2026-08-06", rows)
    scrambled = []
    for i, r in enumerate(rows):
        c = dict(r)
        c["p_trend"] = (i * 37 % 60) / 60.0      # deterministic reshuffle of one input
        scrambled.append(c)
    b = snap(tmp_path, "2026-08-07", rescore(scrambled))
    got = churn.diagnose(a, b)
    assert got["largest_contributor"]["factor"] == "p_trend"
    assert got["largest_contributor"]["share"] > 0.5


def test_counterfactuals_reproduce_the_published_ordering(tmp_path):
    """The re-scoring path must agree with what the pipeline actually published.

    ``_conviction`` re-evaluates the final line of the model with the stored uplift
    rather than re-deriving it from drawdown. If that reconstruction were wrong, every
    counterfactual built on it would be measuring a different model than the one that
    produced the board.
    """
    rows = interleaved(40)
    a = snap(tmp_path, "2026-08-07", rows)
    for sym, stored in a["rows"].items():
        keys = list(churn._effective_weights(stored.get("profile") or "default"))
        rebuilt = churn._conviction(churn._percentiles(stored, keys),
                                    stored.get("profile"), stored.get("mr_uplift"))
        assert rebuilt == pytest.approx(stored["conviction"], abs=0.51), sym


# ---------------------------------------------------------------------------
# boundary flapping
# ---------------------------------------------------------------------------
def test_marginal_tier_changes_are_separated_from_real_ones():
    prev = {"date": "2026-08-06", "rows": {
        "A": {"conviction": 70}, "B": {"conviction": 71}, "C": {"conviction": 40}}}
    curr = {"date": "2026-08-07", "rows": {
        "A": {"conviction": 69},    # BUY -> HOLD on a single point: an artifact
        "B": {"conviction": 55},    # BUY -> HOLD on sixteen points: a real change
        "C": {"conviction": 40}}}   # unchanged
    got = churn.boundary_flapping(prev, curr, ["A", "B", "C"])
    assert got["tier_changes"] == 2
    assert got["marginal"] == 1
    assert got["marginal_share"] == pytest.approx(0.5)
    assert got["examples"][0]["symbol"] == "A"
    assert got["examples"][0]["from_tier"] == "BUY"
    assert got["examples"][0]["to_tier"] == "HOLD"


def test_boundary_flapping_reports_no_share_when_nothing_moved():
    """A share of zero-out-of-zero is None, not 0% — they mean different things."""
    prev = {"rows": {"A": {"conviction": 70}}}
    curr = {"rows": {"A": {"conviction": 70}}}
    got = churn.boundary_flapping(prev, curr, ["A"])
    assert got["tier_changes"] == 0
    assert got["marginal_share"] is None


# ---------------------------------------------------------------------------
# elasticity
# ---------------------------------------------------------------------------
def test_the_amplifiers_flagged_are_the_floored_names(tmp_path):
    """Detecting amplification is only half of it — the mechanism has to be nameable."""
    rows = interleaved(40)
    a = snap(tmp_path, "2026-08-06", rows)
    b = snap(tmp_path, "2026-08-07", rescore(nudge_quality(rows)))
    got = churn.elasticity(a, b, sorted(set(a["rows"]) & set(b["rows"])))
    flagged = {x["symbol"] for x in got["amplifiers"]}
    assert flagged, "no amplifiers reported on a board built entirely out of them"
    # Every flagged name is one of the floored (even-indexed) ones, and Q is named as the
    # weakest pillar — the geometric mean's floor is the mechanism.
    assert all(int(s[1:]) % 2 == 0 for s in flagged), flagged
    assert all(x["weakest_pillar"] == "Q" for x in got["amplifiers"])


def test_elasticity_is_undefined_rather_than_infinite_when_inputs_held(tmp_path):
    """Dividing a move by zero input movement must not produce a headline number."""
    rows = universe(20)
    a = snap(tmp_path, "2026-08-06", rows)
    b = snap(tmp_path, "2026-08-07", rows)
    got = churn.elasticity(a, b, sorted(a["rows"]))
    assert got["names"] == 0
    assert got["median"] is None
    assert got["amplifiers"] == []


# ---------------------------------------------------------------------------
# scope and honesty
# ---------------------------------------------------------------------------
def test_diagnose_needs_two_snapshots(tmp_path):
    rows = universe(20)
    a = snap(tmp_path, "2026-08-07", rows)
    assert churn.diagnose(None, a) is None
    assert churn.diagnose(a, None) is None


def test_diagnose_declines_on_too_small_an_overlap(tmp_path):
    """Five names cannot support a reordering measure worth acting on."""
    rows = universe(5)
    a = snap(tmp_path, "2026-08-06", rows)
    b = snap(tmp_path, "2026-08-07", rows)
    assert churn.diagnose(a, b) is None


def test_diagnose_never_recommends_a_specification_change(tmp_path):
    """It reports evidence. Deciding to change the model is a human call on more data."""
    rows = universe()
    a = snap(tmp_path, "2026-08-06", rows)
    b = snap(tmp_path, "2026-08-07", shift(rows, "p_rs", 0.1))
    got = churn.diagnose(a, b)
    # "recommendation" appears in the scope note, which is the disclaimer itself, so the
    # banned list is prescriptions rather than the word.
    flat = str({k: v for k, v in got.items() if k != "scope"}).lower()
    for banned in ("should change", "we recommend", "you must", "fix the model",
                   "increase the weight", "reduce the weight", "change the spec",
                   "retune", "adjust the floor"):
        assert banned not in flat, f"the diagnostic prescribes: {banned}"
    assert "not a recommendation" in got["scope"].lower()
    assert "one night is not a finding" in got["scope"].lower()


# ---------------------------------------------------------------------------
# the terminal
# ---------------------------------------------------------------------------
TERMINAL = Path(__file__).resolve().parents[1] / "web" / "terminal.html"


def test_terminal_renders_the_churn_diagnosis():
    html = TERMINAL.read_text(encoding="utf-8")
    assert "renderChurn" in html
    for field in ("input_movement", "output_movement", "amplification", "assessment",
                  "largest_contributor", "contributions", "boundary", "elasticity",
                  "amplifiers", "marginal"):
        assert field in html, f"{field!r} is diagnosed nightly but never displayed"


def test_terminal_states_that_one_night_is_not_a_finding():
    html = TERMINAL.read_text(encoding="utf-8")
    view = html.split('id="view-monitor"', 1)[1].split('id="view-method"', 1)[0]
    body = html.split("function renderChurn(", 1)[1].split("\nfunction ", 1)[0]
    assert "one night" in (view + body).lower()
