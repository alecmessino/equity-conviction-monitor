"""Swing layer: signal math, and the structural traps the layer exists to avoid.

The tests named ``test_trap_*`` each encode a way the strategy is usually written down
that does not survive contact with the arithmetic. They are the point of this file.
"""
from __future__ import annotations

import math

import pytest

from equity_monitor import swing


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------
def test_rsi_matches_the_definition_on_wilders_series():
    """Wilder's series from New Concepts in Technical Trading Systems.

    Checked against the arithmetic rather than against a quoted figure. Fifteen closes
    give fourteen changes: gains sum to 3.34 and losses to 1.40, so the seed averages are
    3.34/14 and 1.40/14, RS is 2.3857 and RSI is 70.4641.

    The 70.53 widely quoted for this series is the value *after* further Wilder-smoothed
    bars, not the seed. Asserting it here would have been asserting a misread constant,
    and loosening the tolerance until it passed would have hidden a real disagreement
    about which bar is being reported.
    """
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
              45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
    assert swing.rsi(closes) == pytest.approx(70.4641, abs=0.001)


def test_rsi_smoothing_advances_the_value_on_a_new_bar():
    """The recursive branch runs at all — a seed-only RSI would be a 14-day snapshot."""
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
              45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
    seeded = swing.rsi(closes)
    advanced = swing.rsi(closes + [46.00])
    assert advanced != seeded
    assert advanced < seeded          # a down bar must lower it


def test_rsi_is_none_when_the_window_cannot_be_seeded():
    assert swing.rsi([100.0] * 10) is None


def test_rsi_saturates_rather_than_dividing_by_zero():
    assert swing.rsi([100 + i for i in range(30)]) == 100.0
    assert swing.rsi([100 - i for i in range(30)]) == pytest.approx(0.0, abs=1e-9)


def test_rsi_flat_series_is_neutral_not_a_crash():
    assert swing.rsi([100.0] * 30) == 50.0


# ---------------------------------------------------------------------------
# the selloff leg
# ---------------------------------------------------------------------------
def test_trap_leg_low_is_measured_after_the_high_not_the_52w_low():
    """The 52-week low can predate the 52-week high.

    A name that bottomed, rallied to a high, then eased back has its lowest low *before*
    the peak. Anchoring the retracement to that low describes a decline that never
    happened and every Fibonacci level derived from it is a number about nothing.
    """
    # bottoms at 50 early, peaks at 120, drifts to 100 — the real leg is 120 -> 95.
    highs = [60] * 5 + [55] * 5 + [120] + [110, 105, 100, 98]
    lows = [50] * 5 + [45] * 5 + [115] + [104, 99, 95, 96]
    leg = swing.selloff_leg(highs, lows)
    assert leg["leg_high"] == 120
    assert leg["leg_low"] == 95          # not 45, which is the window's actual minimum
    assert leg["days_since_high"] == 4
    assert leg["days_since_low"] == 1


def test_leg_depth_is_measured_from_the_peak():
    leg = swing.selloff_leg([100, 100, 100], [100, 80, 70])
    assert leg["leg_depth"] == pytest.approx(0.30)


def test_leg_is_none_when_price_never_fell():
    assert swing.selloff_leg([100, 101, 102], [100, 101, 102]) is None


# ---------------------------------------------------------------------------
# targets, stop, reward-to-risk
# ---------------------------------------------------------------------------
def test_retracement_levels_are_fractions_of_the_leg():
    leg = {"leg_high": 200.0, "leg_low": 100.0, "leg_depth": 0.5,
           "days_since_high": 30, "days_since_low": 1}
    t = swing.targets(leg, price=100.0, atr=2.0)
    assert t["target_1"] == pytest.approx(138.2)
    assert t["target_2"] == pytest.approx(150.0)
    assert t["upside_1"] == pytest.approx(0.382)
    assert t["rebound_progress"] == pytest.approx(0.0)


def test_reward_risk_is_measured_against_target_one():
    """Target 1 trims half the book; target 2 is reached less often.

    Sizing against the exit you reach least often is how a strategy with a real edge
    still loses money, so the ratio is computed against the nearer level.
    """
    leg = {"leg_high": 200.0, "leg_low": 100.0, "leg_depth": 0.5,
           "days_since_high": 30, "days_since_low": 1}
    t = swing.targets(leg, price=100.0, atr=5.0)
    assert t["stop"] == pytest.approx(85.0)              # 3 x ATR below entry
    assert t["reward_risk"] == pytest.approx(38.2 / 15.0)


def test_rebound_progress_tracks_position_within_the_leg():
    leg = {"leg_high": 200.0, "leg_low": 100.0, "leg_depth": 0.5,
           "days_since_high": 30, "days_since_low": 5}
    assert swing.targets(leg, 150.0, 2.0)["rebound_progress"] == pytest.approx(0.5)
    assert swing.targets(leg, 200.0, 2.0)["rebound_progress"] == pytest.approx(1.0)


def test_trap_shallow_drawdowns_fail_the_reward_risk_floor():
    """A -12% dip on a 3%-ATR name pays less than it risks.

    This is the structural reason a fixed "-15% off the high" entry loses: volatility
    scales with the fall, so a 3xATR stop widens in step with it, while a fixed-fraction
    retracement target grows faster than linearly. Measured on the 1016-name board the
    median name 10-15% off its high offered +5.3% against -9.8%. The floor is what
    turns the entry depth into an output instead of a guess.
    """
    price = 88.0
    leg = {"leg_high": 100.0, "leg_low": 88.0, "leg_depth": 0.12,
           "days_since_high": 20, "days_since_low": 1}
    shallow = swing.targets(leg, price, atr=price * 0.03)
    assert shallow["reward_risk"] < 1.0
    assert shallow["reward_risk"] < swing.MIN_REWARD_RISK

    deep_price = 65.0
    deep_leg = {"leg_high": 100.0, "leg_low": 65.0, "leg_depth": 0.35,
                "days_since_high": 60, "days_since_low": 1}
    deep = swing.targets(deep_leg, deep_price, atr=deep_price * 0.03)
    assert deep["reward_risk"] > shallow["reward_risk"]


def test_targets_degrade_without_an_atr_rather_than_inventing_a_stop():
    leg = {"leg_high": 200.0, "leg_low": 100.0, "leg_depth": 0.5,
           "days_since_high": 30, "days_since_low": 1}
    t = swing.targets(leg, 100.0, atr=None)
    assert t["target_1"] == pytest.approx(138.2)
    assert t["stop"] is None and t["reward_risk"] is None


# ---------------------------------------------------------------------------
# relative drawdown
# ---------------------------------------------------------------------------
def test_trap_a_name_falling_with_the_market_has_no_relative_drawdown():
    """25% off the high while the index is also 25% off is beta, not a mispricing."""
    bench = [100.0] * 60 + [100 - i for i in range(40)]
    stock = [50.0] * 60 + [50 * (100 - i) / 100 for i in range(40)]
    assert swing.relative_drawdown(stock, bench) == pytest.approx(0.0, abs=1e-9)


def test_relative_drawdown_isolates_the_idiosyncratic_fall():
    bench = [100.0] * 100
    stock = [50.0] * 80 + [50 * (1 - 0.003 * i) for i in range(20)]
    rel = swing.relative_drawdown(stock, bench)
    assert rel == pytest.approx(0.057, abs=0.005)


def test_relative_drawdown_needs_no_beta_estimate():
    """Short histories return a number or None, never a silently degraded regression."""
    assert swing.relative_drawdown([1.0] * 10, [1.0] * 10) is None


# ---------------------------------------------------------------------------
# z-score and stretch
# ---------------------------------------------------------------------------
def test_zscore_is_none_without_a_dispersion_estimate():
    assert swing.zscore(100.0, 110.0, None) is None
    assert swing.zscore(100.0, 110.0, 0.0) is None
    assert swing.zscore(90.0, 100.0, 5.0) == pytest.approx(-2.0)


def test_stdev_requires_a_usable_window():
    assert swing.stdev([1.0, 2.0, 3.0], 50) is None
    assert swing.stdev([float(i) for i in range(50)], 50) == pytest.approx(14.577, abs=0.01)


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def _series(n=300, start=100.0, end=70.0):
    """A monotone decline, then two up days so the name reads as stabilising."""
    closes = [start + (end - start) * i / (n - 3) for i in range(n - 2)]
    closes += [closes[-1] * 1.01, closes[-1] * 1.02]
    highs = [c * 1.01 for c in closes]
    lows = [c * 0.99 for c in closes]
    return closes, highs, lows


def test_evaluate_names_the_gate_that_blocked_a_candidate():
    closes, highs, lows = _series()
    row = {"price": closes[-1], "ma50": closes[-1] * 1.15,
           "atr14": closes[-1] * 0.02, "drawdown_52w": 0.30}
    out = swing.evaluate(row, closes, highs, lows, quality_pctile=0.20)
    assert out["swing_signal"] == "NONE"
    assert out["swing_block"] == "quality below gate"


def test_evaluate_blocks_on_reward_risk_before_looking_at_oversold():
    closes, highs, lows = _series(end=95.0)
    row = {"price": closes[-1], "ma50": closes[-1] * 1.05,
           "atr14": closes[-1] * 0.05, "drawdown_52w": 0.12}
    out = swing.evaluate(row, closes, highs, lows, quality_pctile=0.90)
    assert out["swing_signal"] == "NONE"
    assert out["swing_block"] == "reward:risk below floor"


def test_evaluate_holds_a_still_falling_name_at_watch():
    n = 300
    closes = [100.0 - 30.0 * i / (n - 1) for i in range(n)]   # new low every day
    highs = [c * 1.01 for c in closes]
    lows = [c * 0.99 for c in closes]
    row = {"price": closes[-1], "ma50": closes[-1] * 1.20,
           "atr14": closes[-1] * 0.015, "drawdown_52w": 0.30}
    out = swing.evaluate(row, closes, highs, lows, quality_pctile=0.90)
    assert out["stabilizing"] is False
    assert out["swing_signal"] == "WATCH"
    assert out["swing_block"] == "still making lows"


def test_evaluate_promotes_a_stabilised_oversold_quality_name():
    closes, highs, lows = _series()
    row = {"price": closes[-1], "ma50": closes[-1] * 1.20,
           "atr14": closes[-1] * 0.015, "drawdown_52w": 0.30}
    out = swing.evaluate(row, closes, highs, lows, quality_pctile=0.90)
    assert out["swing_signal"] in {"SETUP", "PRIME"}
    assert out["swing_block"] is None
    assert out["reward_risk"] >= swing.MIN_REWARD_RISK


def test_evaluate_returns_a_full_key_set_even_when_it_bails_early():
    """Every consumer reads the same keys whether or not a signal fired."""
    full = swing.evaluate({"price": 100.0, "ma50": 110.0, "atr14": 2.0,
                           "drawdown_52w": 0.3}, *_series(), quality_pctile=0.9)
    empty = swing.evaluate({"price": None}, [], [], [], quality_pctile=0.9)
    assert set(empty) == set(full)


def test_evaluate_never_reads_a_future_bar():
    """Truncating the series must not change the signal computed at that point.

    The single property that decides whether the research harness is measuring the
    strategy or measuring a leak.
    """
    closes, highs, lows = _series(n=320)
    row = {"price": closes[-21], "ma50": closes[-21] * 1.2,
           "atr14": closes[-21] * 0.015, "drawdown_52w": 0.28}
    as_of = swing.evaluate(row, closes[:-20], highs[:-20], lows[:-20], quality_pctile=0.9)
    with_future = swing.evaluate(row, closes, highs, lows, quality_pctile=0.9)
    assert as_of["rsi14"] != with_future["rsi14"]      # the inputs genuinely differ
    again = swing.evaluate(row, closes[:-20], highs[:-20], lows[:-20], quality_pctile=0.9)
    assert as_of == again                              # and the truncated view is stable
