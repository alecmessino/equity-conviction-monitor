"""Structural-yield pillars — properties that must hold, and the traps that must not.

The tests named ``test_trap_*`` each encode a way this class of strategy reports a
signal that is not there. They are the point of this file: every one of them was a
real result during the feasibility measurement, not a hypothetical.
"""
from __future__ import annotations

import math
import random

import pytest

from research import structural_yield as sy


# ---------------------------------------------------------------------------
# pillar 1: cointegration
# ---------------------------------------------------------------------------
def _random_walk(n, seed, drift=0.0, vol=0.01, start=100.0):
    rnd = random.Random(seed)
    out, p = [start], start
    for _ in range(n - 1):
        p *= math.exp(drift + vol * rnd.gauss(0, 1))
        out.append(p)
    return out


def test_cointegrated_pair_is_detected():
    """A constructed stationary spread must clear the 5% Engle-Granger value."""
    base = _random_walk(500, seed=1)
    rnd = random.Random(2)
    # b tracks a with a mean-reverting (AR(1), phi=0.85) log deviation.
    dev, b = 0.0, []
    for p in base:
        dev = 0.85 * dev + 0.01 * rnd.gauss(0, 1)
        b.append(p * math.exp(dev))
    r = sy.engle_granger(b, base)
    assert r["adf"] < sy.EG_CRITICAL[0.05]
    assert r["half_life"] is not None and 1 < r["half_life"] < 20


def test_trap_two_independent_random_walks_do_not_cointegrate():
    """The base rate this pillar has to beat. Independent walks must mostly fail."""
    passes = 0
    for s in range(40):
        r = sy.engle_granger(_random_walk(400, seed=100 + s),
                             _random_walk(400, seed=900 + s))
        if r and r["adf"] is not None and r["adf"] < sy.EG_CRITICAL[0.05]:
            passes += 1
    # Nominally 5% of 40 = 2. Allow headroom for sampling; anything near 40 means
    # the test statistic is broken and every pair would look cointegrated.
    assert passes <= 6, f"{passes}/40 independent walks 'cointegrated'"


def test_trap_multiple_testing_tightens_the_cutoff():
    """Searching many pairs must demand a stronger statistic, not the same one."""
    one = sy.benjamini_hochberg([-3.4], alpha=0.05)
    many = sy.benjamini_hochberg([-3.4] + [-1.0] * 199, alpha=0.05)
    assert many < one, "BH cutoff did not tighten as the search widened"
    assert many < sy.EG_CRITICAL[0.05]


def test_trap_negative_hedge_ratio_is_not_a_spread():
    """A negative beta is a double-sided directional bet, however good its ADF."""
    r = {"adf": -9.0, "hedge_ratio": -0.8, "half_life": 10.0}
    assert not sy.tradeable(r, adf_cutoff=-3.34)
    assert sy.tradeable({**r, "hedge_ratio": 0.8}, adf_cutoff=-3.34)


@pytest.mark.parametrize("hl,ok", [(0.5, False), (10.0, True), (200.0, False)])
def test_half_life_must_be_actionable(hl, ok):
    r = {"adf": -5.0, "hedge_ratio": 1.0, "half_life": hl}
    assert sy.tradeable(r, adf_cutoff=-3.34) is ok


def test_divergent_spread_has_no_half_life():
    assert sy.half_life([1.0 * (1.05 ** i) for i in range(80)]) is None


# ---------------------------------------------------------------------------
# pillar 2: volatility risk premium
# ---------------------------------------------------------------------------
def test_trap_one_event_day_dominates_trailing_realised_vol():
    """The measured MSFT case: a single earnings gap doubled the trailing RV.

    A 25-day option cannot contain a jump whose next occurrence is a quarter out, so
    a VRP computed against the contaminated window reports a premium that is not
    there. Ex-the-event vol must be materially lower.
    """
    rnd = random.Random(7)
    closes = [100.0]
    for _ in range(21):
        closes.append(closes[-1] * math.exp(0.008 * rnd.gauss(0, 1)))
    calm = sy.realized_vol(closes, 21)
    shocked = closes[:-1] + [closes[-1] * 1.14]        # a +14% earnings day
    assert sy.realized_vol(shocked, 21) > calm * 1.8
    assert sy.realized_vol(shocked, 21, drop_extremes=1) < sy.realized_vol(shocked, 21)


def test_forward_vrp_uses_only_subsequent_returns():
    """A constant-vol series priced at a known premium recovers that premium."""
    rnd = random.Random(11)
    dates = [f"d{i:04d}" for i in range(200)]
    closes = [100.0]
    for _ in range(199):
        closes.append(closes[-1] * math.exp(0.01 * rnd.gauss(0, 1)))
    true_vol = 0.01 * math.sqrt(252)
    implied = [(d, true_vol + 0.05) for d in dates[:150]]
    r = sy.forward_vrp(implied, closes, dates, window=21)
    assert r["n"] > 100
    assert 0.02 < r["mean"] < 0.08          # recovers ~5 vol points
    assert r["pct_positive"] > 0.8


def test_forward_vrp_reports_the_tail_not_just_the_mean():
    """Selling insurance is defined by its worst case; the summary must carry it."""
    rnd = random.Random(13)
    dates = [f"d{i:04d}" for i in range(200)]
    closes = [100.0]
    for i in range(199):
        # Calm for most of the sample, then a genuinely volatile crash — a smooth
        # decline would have near-zero variance and no tail to find.
        vol = 0.004 if i < 150 else 0.05
        closes.append(closes[-1] * math.exp(vol * rnd.gauss(0, 1) - (0.01 if i >= 150 else 0)))
    r = sy.forward_vrp([(d, 0.10) for d in dates[:170]], closes, dates, window=21)
    assert r is not None
    assert r["worst"] < r["mean"], "tail not distinguished from the average"
    assert r["worst"] < 0, "a vol explosion must produce a losing insurance sale"
    assert r["p05"] <= r["median"]


# ---------------------------------------------------------------------------
# pillar 3: volatility harvesting
# ---------------------------------------------------------------------------
def test_excess_growth_is_non_negative_and_zero_when_identical():
    """The theorem's two boundary properties."""
    cov_same = [[0.04, 0.04], [0.04, 0.04]]            # perfectly correlated, equal var
    assert sy.excess_growth(cov_same, [0.5, 0.5]) == pytest.approx(0.0, abs=1e-12)
    cov_ind = [[0.04, 0.0], [0.0, 0.04]]
    assert sy.excess_growth(cov_ind, [0.5, 0.5]) > 0


def test_theorem_identity_rebalanced_beats_geometric_by_excess_growth():
    """The load-bearing test: gamma* must equal the *measured* rebalancing gain
    over the weighted-average-log-growth portfolio.

    Measured live at 1.0573 against a predicted 1.0571 on a 10-asset crypto basket.
    If this identity ever breaks, either the covariance or the backtest is wrong.
    """
    series = [_random_walk(760, seed=20 + i, vol=0.02) for i in range(6)]
    w = [1 / 6] * 6
    bt = sy.rebalance_backtest(series, w, every=1)
    years = (len(series[0]) - 1) / 252
    predicted = math.exp(sy.excess_growth(sy.cov_matrix(series), w) * years)
    assert bt["rebalanced"] / bt["geometric"] == pytest.approx(predicted, rel=0.02)


def test_trap_harvest_does_not_promise_beating_buy_and_hold():
    """A dispersed basket with one runaway winner: gamma* > 0 yet hold wins.

    This is the crypto measurement in miniature and the reason ``excess_growth``
    carries the warning it does.
    """
    n = 400
    winner = [100.0 * (1.01 ** i) for i in range(n)]
    flat = [_random_walk(n, seed=50 + i, vol=0.02) for i in range(3)]
    series = [winner] + flat
    w = [0.25] * 4
    assert sy.excess_growth(sy.cov_matrix(series), w) > 0
    bt = sy.rebalance_backtest(series, w, every=1)
    assert bt["vs_geometric"] > 0          # theorem holds
    assert bt["vs_hold"] < 0               # and hold still wins


def test_costs_reduce_the_harvest_and_turnover_is_reported():
    series = [_random_walk(500, seed=70 + i, vol=0.03) for i in range(5)]
    w = [0.2] * 5
    free = sy.rebalance_backtest(series, w, every=1, cost_bps=0)
    paid = sy.rebalance_backtest(series, w, every=1, cost_bps=50)
    assert paid["rebalanced"] < free["rebalanced"]
    assert free["turnover"] > 0


# ---------------------------------------------------------------------------
# pillar 4 and the regime gate
# ---------------------------------------------------------------------------
def test_sigma_event_scales_with_the_move():
    closes = _random_walk(120, seed=3, vol=0.01)
    calm = sy.sigma_event(closes)
    assert abs(calm["sigma"]) < 3
    shocked = sy.sigma_event(closes[:-1] + [closes[-2] * 0.90])
    assert shocked["sigma"] < -3


def test_trend_strength_separates_a_line_from_chop():
    assert sy.trend_strength([100 + i for i in range(80)]) == pytest.approx(1.0)
    chop = [100 + (1 if i % 2 else -1) for i in range(80)]
    assert abs(sy.trend_strength(chop)) < 0.1


def test_regime_gate_blocks_thin_premium_and_strong_trends():
    assert sy.regime_ok(0.05, 0.05)["ok"]
    assert not sy.regime_ok(0.00, 0.05)["ok"]                 # premium too thin
    assert not sy.regime_ok(0.05, 0.80)["ok"]                 # trending
    assert not sy.regime_ok(None, None)["ok"]                 # unobservable != fine
    assert "vrp_thin" in " ".join(sy.regime_ok(0.0, 0.0)["reasons"])
