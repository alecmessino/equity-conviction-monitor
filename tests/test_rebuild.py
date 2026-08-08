"""Historical board reconstruction.

The test that matters most in this file is the refusal: a reconstruction must never be
able to reach ``ledger/snapshots/``. A reconstructed snapshot is structurally identical
to a recorded one, so contamination is undetectable after the fact, and an Information
Coefficient computed over a mixed series would be measuring the look-ahead rather than
the model. Every other test here protects a smaller thing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from equity_monitor import model, rebuild, snapshots, universe as uni
from tests.test_snapshots import row


# ---------------------------------------------------------------------------
# fixtures: a miniature ledger with real-shaped cached bars
# ---------------------------------------------------------------------------
def bars(symbol, n=300, start=100.0, drift=0.001):
    """A deterministic ascending series — enough bars for every window the model uses."""
    dates, close = [], []
    price = start
    for i in range(n):
        # A synthetic but valid trading calendar: weekdays only, from a fixed origin.
        day = 1 + i
        dates.append(f"2026-{1 + (day // 28) % 12:02d}-{1 + day % 28:02d}")
        price *= (1.0 + drift)
        close.append(round(price, 4))
    return {
        "symbol": symbol, "source": "test", "dates": sorted(set(dates))[:len(close)],
        "open": close[:], "high": [c * 1.01 for c in close],
        "low": [c * 0.99 for c in close], "close": close,
        "volume": [1_000_000] * len(close),
    }


@pytest.fixture
def ledger(tmp_path):
    """A ledger directory with cached history and a recorded snapshot."""
    history = tmp_path / "history"
    history.mkdir()
    syms = [uni.BENCHMARK, "AAPL", "MSFT"]
    for i, sym in enumerate(syms):
        node = bars(sym, drift=0.001 + i * 0.0004)
        (history / f"{sym}.json").write_text(json.dumps(node))

    rows = [row("AAPL"), row("MSFT", p_roic=0.4)]
    (tmp_path / "snapshots").mkdir()
    snapshots.write(rows, str(tmp_path), on="2026-08-07", as_of="2026-08-07T23:00:00Z")

    (tmp_path / "index.json").write_text(json.dumps({
        "as_of": "2026-08-07T23:00:00Z", "model_version": model.MODEL_VERSION,
        "all": [{"symbol": "AAPL", "name": "Apple Inc.", "sector": "Information Technology",
                 "roic": 0.31, "fcf_yield": 0.04, "market_cap": 4.5e12},
                {"symbol": "MSFT", "name": "Microsoft Corp.", "sector": "Information Technology",
                 "roic": 0.28, "fcf_yield": 0.03, "market_cap": 3.1e12}],
    }))
    return tmp_path


# ---------------------------------------------------------------------------
# the refusal
# ---------------------------------------------------------------------------
def test_a_reconstruction_cannot_be_written_into_the_recorded_history():
    """The single most important behaviour in this module.

    A reconstructed snapshot is byte-shaped exactly like a recorded one, so a mixed
    series cannot be un-mixed and the contamination is silent. The guard is a path check
    rather than a convention, because a convention is exactly what fails at 2am.
    """
    with pytest.raises(rebuild.RefusedError) as exc:
        rebuild.run(5, rebuild.LEDGER)
    assert "look-ahead" in str(exc.value)


def test_the_refusal_covers_paths_nested_inside_the_history():
    """`ledger/snapshots/scratch` is still the recorded series' directory."""
    nested = os.path.join(rebuild.LEDGER, "snapshots", "anything", "deeper")
    with pytest.raises(rebuild.RefusedError):
        rebuild.run(5, nested)


def test_the_refusal_survives_a_symlink_or_relative_path(tmp_path):
    """`realpath` rather than string comparison — `ledger/../ledger/snapshots` is the same place."""
    sneaky = os.path.join(rebuild.LEDGER, "..", "ledger", "snapshots")
    with pytest.raises(rebuild.RefusedError):
        rebuild.run(5, sneaky)


def test_an_ordinary_output_directory_is_allowed(tmp_path):
    rebuild._assert_not_the_real_history(str(tmp_path))          # must not raise
    rebuild._assert_not_the_real_history(os.path.join(rebuild.LEDGER, "..", "build"))


def test_the_cli_exits_two_on_refusal(monkeypatch, capsys):
    """A distinct exit code, so a script cannot mistake a refusal for a build failure."""
    monkeypatch.setattr("sys.argv", ["rebuild", "--out", rebuild.LEDGER])
    assert rebuild.main() == 2
    assert "REFUSED" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the calendar
# ---------------------------------------------------------------------------
def test_the_calendar_comes_from_the_benchmark(ledger):
    """One calendar for the universe.

    A per-symbol "last N bars" rule would compare names as of different dates whenever
    one of them did not trade — a silent misalignment that looks like signal.
    """
    loaded = rebuild.load_bars(str(ledger / "history"))
    cal = rebuild.sessions(loaded)
    assert cal == loaded[uni.BENCHMARK]["dates"]


def test_a_missing_benchmark_is_a_hard_failure(ledger):
    os.remove(ledger / "history" / f"{uni.BENCHMARK}.json")
    loaded = rebuild.load_bars(str(ledger / "history"))
    with pytest.raises(RuntimeError, match="no cached history"):
        rebuild.sessions(loaded)


def test_back_must_land_inside_the_cached_calendar(ledger):
    for bad in (0, -1, 10_000):
        with pytest.raises(ValueError, match="--back must be between"):
            rebuild.run(bad, str(ledger / "out"), str(ledger))


def test_a_ledger_with_no_cached_bars_fails_clearly(tmp_path):
    (tmp_path / "index.json").write_text(json.dumps({"all": []}))
    (tmp_path / "history").mkdir()
    with pytest.raises(RuntimeError, match="no cached price history"):
        rebuild.run(5, str(tmp_path / "out"), str(tmp_path))


# ---------------------------------------------------------------------------
# truncation
# ---------------------------------------------------------------------------
def test_truncation_keeps_only_bars_at_or_before_the_cutoff(ledger):
    node = json.loads((ledger / "history" / "AAPL.json").read_text())
    cutoff = node["dates"][200]
    cut = rebuild.truncate(node, cutoff)
    assert cut.dates[-1] == cutoff
    assert all(d <= cutoff for d in cut.dates)
    assert len(cut.dates) == len(cut.close) == len(cut.volume)


def test_a_series_with_too_little_history_is_dropped_not_scored_short(ledger):
    """The nightly drops these too. Scoring a 12-month excess return off 30 bars is fiction."""
    node = json.loads((ledger / "history" / "AAPL.json").read_text())
    assert rebuild.truncate(node, node["dates"][rebuild.MIN_BARS - 2]) is None
    assert rebuild.truncate(node, node["dates"][rebuild.MIN_BARS + 5]) is not None


# ---------------------------------------------------------------------------
# the look-ahead boundary
# ---------------------------------------------------------------------------
def test_price_inputs_are_never_lifted_from_the_published_ledger():
    """The whole point: price features are reconstructed, fundamentals are not.

    If a price-derived field leaked into the overlay list, the reconstruction would be
    comparing today's board against itself and every diff would read as quiet.
    """
    for leaked in ("rs_blend", "trend", "adv_usd", "vol_1y", "chg_1d", "ret_12m",
                   "drawdown_52w", "price"):
        assert leaked not in rebuild.FUNDAMENTAL_FIELDS, \
            f"{leaked} is price-derived and must be recomputed, not overlaid"


def test_every_fundamental_the_model_ranks_is_overlaid():
    """A fundamental left un-overlaid would be None and impute to the sector median,
    quietly flattening the quality pillar across the whole reconstructed board."""
    ranked = {metric for _, metric, _ in model.RANK_SPEC}
    price_side = {"rs_blend", "trend", "adv_usd", "vol_1y"}
    for metric in ranked - price_side:
        assert metric in rebuild.FUNDAMENTAL_FIELDS, metric


def test_the_disclosure_names_the_bias_rather_than_gesturing_at_it():
    text = rebuild.DISCLOSURE.lower()
    assert "look-ahead" in text
    assert "quality" in text
    assert "not part of the recorded history" in text


# ---------------------------------------------------------------------------
# end to end, against the real repo ledger
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.isdir(os.path.join(rebuild.LEDGER, "history")),
                    reason="needs the cached price history")
def test_end_to_end_against_the_real_ledger(tmp_path):
    out = str(tmp_path / "rebuild")
    # Read before, compare after. Pinning the expected list to a literal date asserted
    # "the history is one day long" rather than "the reconstruction did not touch it",
    # so the test broke the first night a second snapshot was recorded — which is the
    # one thing this suite must never do, since it would train someone to ignore it.
    before = snapshots.available(rebuild.LEDGER)
    got = rebuild.run(5, out)

    assert got["names"] > 500
    assert got["reconstructed"] is True
    assert got["cutoff"] < got["against"]

    # Both halves of the comparison are on disk, and the recorded one was copied in
    # rather than reconstructed.
    stamps = snapshots.available(out)
    assert stamps == sorted([got["cutoff"], got["against"]])

    # Every artifact carries the disclosure — it must not be possible to screenshot one
    # of these panels and mistake it for the recorded board.
    assert got["watchlist"]["reconstructed"] is True
    assert got["churn"]["reconstructed"] is True
    assert "look-ahead" in got["watchlist"]["disclosure"]

    # And the real history is untouched — the same stamps, neither added to nor removed.
    assert snapshots.available(rebuild.LEDGER) == before


@pytest.mark.skipif(not os.path.isdir(os.path.join(rebuild.LEDGER, "history")),
                    reason="needs the cached price history")
def test_a_longer_lookback_moves_the_board_further(tmp_path):
    """A sanity check on the reconstruction actually reconstructing.

    If truncation were a no-op, both runs would report near-identical stability. More
    sessions of real price movement must reorder the board more.
    """
    near = rebuild.run(2, str(tmp_path / "a"))
    far = rebuild.run(20, str(tmp_path / "b"))
    assert far["stability"]["rank_correlation"] < near["stability"]["rank_correlation"]
    assert far["stability"]["mean_abs_move"] > near["stability"]["mean_abs_move"]
