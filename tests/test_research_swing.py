"""Event-study harness: the defects that made its output wrong, pinned.

Every test named ``test_defect_*`` encodes a way the study reported a number that could
not be traded. The harness had no test coverage at all while it was being trusted to
decide whether a strategy works, which is the same shape of failure as the v2 dashboard.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_spec = importlib.util.spec_from_file_location(
    "research_swing",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "research_swing.py"))
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)


# ---------------------------------------------------------------------------
# benchmark alignment
# ---------------------------------------------------------------------------
def test_defect_unknown_benchmark_date_drops_the_event():
    """It used to fall back to the stock's positional index.

    The two series need not be aligned, so the fallback could hand the relative-drawdown
    calculation a benchmark slice ending *after* the event date — a look-ahead leak that
    produced a plausible number instead of an error.
    """
    index = {"2026-01-02": 0, "2026-01-03": 1}
    dates = ["2026-01-02", "2026-01-03", "2026-01-06"]
    assert rs._bidx(index, dates, 0) == 1
    assert rs._bidx(index, dates, 1) == 2
    assert rs._bidx(index, dates, 2) is None        # unknown date -> drop, never guess
    assert rs._bidx(index, dates, 99) is None


def test_bench_return_is_aligned_by_date_not_by_offset():
    bench = [100.0, 110.0, 121.0]
    index = {"a": 0, "b": 1, "c": 2}
    dates = ["a", "b", "c"]
    assert rs.bench_return(bench, index, dates, 0, 1) == pytest.approx(0.10)
    assert rs.bench_return(bench, index, dates, 0, 2) == pytest.approx(0.21)


# ---------------------------------------------------------------------------
# fills
# ---------------------------------------------------------------------------
def _flat(n, v=100.0):
    return [v] * n


def test_defect_a_gap_through_the_stop_fills_at_the_open():
    """Filling at the stop price pretends a resting order caught a price that never
    traded. Gapping down through a stop is the primary failure mode of a dip-buy."""
    highs = [100.0, 92.0]
    lows = [100.0, 88.0]
    opens = [100.0, 90.0]          # gapped straight through a 95 stop
    closes = [100.0, 91.0]
    out = rs.simulate(highs, lows, opens, closes, 0, 100.0, 95.0, 120.0, 5)
    assert out["exit"] == "stop"
    assert out["ret"] == pytest.approx(-0.10)       # filled at the 90 open, not at 95


def test_stop_fills_at_the_stop_when_there_is_no_gap():
    highs = [100.0, 99.0]
    lows = [100.0, 94.0]
    opens = [100.0, 98.0]
    closes = [100.0, 96.0]
    out = rs.simulate(highs, lows, opens, closes, 0, 100.0, 95.0, 120.0, 5)
    assert out["ret"] == pytest.approx(-0.05)


def test_defect_an_ambiguous_bar_resolves_against_the_strategy():
    """One bar touching both stop and target has no intrabar sequence in daily data."""
    highs = [100.0, 130.0]
    lows = [100.0, 90.0]
    opens = [100.0, 100.0]
    closes = [100.0, 120.0]
    out = rs.simulate(highs, lows, opens, closes, 0, 100.0, 95.0, 120.0, 5)
    assert out["exit"] == "stop"


def test_defect_excursions_stop_at_the_exit():
    """Recorded past the fill they describe a position that was no longer open."""
    highs = [100.0, 101.0, 100.0, 140.0]
    lows = [100.0, 99.0, 80.0, 130.0]
    opens = [100.0, 100.0, 95.0, 135.0]
    closes = [100.0, 100.0, 85.0, 138.0]
    out = rs.simulate(highs, lows, opens, closes, 0, 100.0, 95.0, 200.0, 5)
    assert out["exit"] == "stop"
    assert out["mfe"] < 0.05        # never credited with the +40% it was closed before
    assert out["mae"] >= -0.06      # and the adverse excursion stops at the fill


def test_time_exit_uses_the_close_at_the_horizon():
    closes = _flat(30)
    closes[10] = 110.0
    out = rs.simulate(_flat(30, 200.0), _flat(30, 1.0), _flat(30), closes, 0, 100.0,
                      None, None, 5)
    assert out["exit"] == "time" and out["days"] == 5


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
def _ev(date, v):
    return {"date": date, "excess": v}


def test_defect_the_bootstrap_refuses_an_interval_it_cannot_support():
    """Resampling a handful of blocks estimates its own width badly.

    A confident-looking interval on a two-episode sample is worse than no interval, and
    the previous version always produced one.
    """
    few = [_ev(d, 0.01) for d in _spread_dates(4)]
    assert rs.block_bootstrap(few) is None


def _spread_dates(n_blocks: int, per_block: int = 3):
    """Event dates spanning `n_blocks` distinct calendar blocks.

    Blocks are calendar spans, so consecutive days collapse into one. Generating a run of
    consecutive dates — which an earlier version of these tests did — produces far fewer
    blocks than dates and tests the wrong thing.
    """
    import datetime as dt
    base = dt.date(2016, 1, 4)
    span = int(rs.swing.MAX_HOLD_DAYS * 1.45) + 2
    return [(base + dt.timedelta(days=b * span + i)).isoformat()
            for b in range(n_blocks) for i in range(per_block)]


def test_bootstrap_returns_an_interval_once_there_are_enough_blocks():
    events = [_ev(d, 0.01 * (i % 7 - 3))
              for i, d in enumerate(_spread_dates(rs.MIN_BLOCKS + 6))]
    ci = rs.block_bootstrap(events)
    assert ci is not None
    lo, hi = ci
    assert lo < hi


def test_bootstrap_is_insensitive_to_the_order_events_arrive_in():
    """Blocks are a property of the calendar, not of insertion order."""
    forward = [_ev(d, 0.01) for d in _spread_dates(rs.MIN_BLOCKS + 6)]
    shuffled = list(reversed(forward))
    assert rs.block_bootstrap(forward) == rs.block_bootstrap(shuffled)


def test_events_without_a_date_never_enter_the_interval():
    """An undated event cannot be placed in a block, so it cannot be resampled."""
    dated = [_ev(d, 0.0) for d in _spread_dates(rs.MIN_BLOCKS + 6)]
    ci = rs.block_bootstrap([_ev("", 5.0)] * 500 + dated)
    assert ci is not None
    assert abs(ci[0]) < 0.01 and abs(ci[1]) < 0.01     # the undated 5.0s are excluded
