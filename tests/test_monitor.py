"""Model monitoring: stability, coverage, regime observation, health checks."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from equity_monitor import model, monitor, snapshots
from tests.test_snapshots import BASE_PERCENTILES, row


def snap(tmp_path, stamp, rows):
    snapshots.write(rows, str(tmp_path), on=stamp, as_of=stamp + "T23:00:00Z")


def universe(n=40, seed=0.0):
    out = []
    for i in range(n):
        p = {k: min(1.0, max(0.0, v + seed + (i % 7) * 0.03))
             for k, v in BASE_PERCENTILES.items()}
        out.append(row(f"S{i:03d}", **p))
    return out


# ---------------------------------------------------------------------------
# rank correlation
# ---------------------------------------------------------------------------
def test_spearman_matches_known_values():
    assert monitor._spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert monitor._spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    assert monitor._spearman([1, 2], [2, 1]) is None, "too few points to rank"


def test_spearman_averages_ties():
    """Conviction is an integer over ~1,000 names, so ties are the common case.
    Breaking them arbitrarily would understate agreement between two runs."""
    assert monitor._spearman([1, 1, 1, 2], [1, 1, 1, 2]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# stability
# ---------------------------------------------------------------------------
def test_stability_reports_an_unchanged_board_as_unchanged(tmp_path):
    rows = universe()
    snap(tmp_path, "2026-08-06", rows)
    snap(tmp_path, "2026-08-07", rows)
    got = monitor.stability(
        snapshots.read(snapshots.snapshot_path(str(tmp_path), "2026-08-06")),
        snapshots.read(snapshots.snapshot_path(str(tmp_path), "2026-08-07")))
    assert got["rank_correlation"] == pytest.approx(1.0)
    assert got["mean_abs_move"] == 0
    assert got["tier_changes"] == 0
    assert got["unchanged"] == got["names_compared"]


def test_stability_detects_a_reshuffled_board(tmp_path):
    """A ranking that reorders overnight is fitting noise however good each night looks."""
    before = universe()
    after = [row(r["symbol"], **{k: 1.0 - r[k] for k in BASE_PERCENTILES}) for r in before]
    snap(tmp_path, "2026-08-06", before)
    snap(tmp_path, "2026-08-07", after)
    got = monitor.stability(
        snapshots.read(snapshots.snapshot_path(str(tmp_path), "2026-08-06")),
        snapshots.read(snapshots.snapshot_path(str(tmp_path), "2026-08-07")))
    assert got["rank_correlation"] < 0, f"expected inversion, got {got['rank_correlation']}"
    assert got["mean_abs_move"] > 5


def test_stability_tracks_entries_and_exits(tmp_path):
    snap(tmp_path, "2026-08-06", universe(10))
    snap(tmp_path, "2026-08-07", universe(12))
    got = monitor.stability(
        snapshots.read(snapshots.snapshot_path(str(tmp_path), "2026-08-06")),
        snapshots.read(snapshots.snapshot_path(str(tmp_path), "2026-08-07")))
    assert got["entered"] == ["S010", "S011"]
    assert got["left"] == []
    assert got["names_compared"] == 10


def test_stability_flags_a_specification_change(tmp_path):
    """Comparing across a spec change compares two different models."""
    snap(tmp_path, "2026-08-06", universe(10))
    original = model.MODEL_VERSION
    model.MODEL_VERSION = "v9.9.9-test"
    try:
        snap(tmp_path, "2026-08-07", universe(10))
    finally:
        model.MODEL_VERSION = original
    got = monitor.stability(
        snapshots.read(snapshots.snapshot_path(str(tmp_path), "2026-08-06")),
        snapshots.read(snapshots.snapshot_path(str(tmp_path), "2026-08-07")))
    assert got["spec_changed"] is True


def test_stability_needs_two_snapshots(tmp_path):
    assert monitor.stability(None, None) is None


def test_tier_migration_totals_the_compared_names(tmp_path):
    snap(tmp_path, "2026-08-06", universe(30))
    snap(tmp_path, "2026-08-07", universe(30, seed=0.05))
    got = monitor.stability(
        snapshots.read(snapshots.snapshot_path(str(tmp_path), "2026-08-06")),
        snapshots.read(snapshots.snapshot_path(str(tmp_path), "2026-08-07")))
    total = sum(v for r in got["tier_migration"].values() for v in r.values())
    assert total == got["names_compared"]


# ---------------------------------------------------------------------------
# coverage and specification history
# ---------------------------------------------------------------------------
def test_coverage_trend_reports_one_entry_per_date(tmp_path):
    for stamp in ["2026-08-05", "2026-08-06", "2026-08-07"]:
        snap(tmp_path, stamp, universe(10))
    got = monitor.coverage_trend(str(tmp_path))
    assert got["observations"] == 3
    assert [s["date"] for s in got["series"]] == ["2026-08-05", "2026-08-06", "2026-08-07"]
    assert all(0.0 <= s["mean_confidence"] <= 1.0 for s in got["series"])


def test_spec_consistency_segments_history_at_a_model_change(tmp_path):
    """A series spanning two specs cannot be regressed as one dataset. Surfacing the
    boundary now means Phase 3 starts from a segmented series rather than discovering
    the discontinuity in its results."""
    snap(tmp_path, "2026-08-05", universe(5))
    snap(tmp_path, "2026-08-06", universe(5))
    original = model.MODEL_VERSION
    model.MODEL_VERSION = "v9.9.9-test"
    try:
        snap(tmp_path, "2026-08-07", universe(5))
    finally:
        model.MODEL_VERSION = original

    got = monitor.spec_consistency(str(tmp_path))
    assert len(got["segments"]) == 2
    assert got["segments"][0]["days"] == 2
    assert got["segments"][0]["from"] == "2026-08-05"
    assert got["segments"][1]["from"] == "2026-08-07"


# ---------------------------------------------------------------------------
# regime
# ---------------------------------------------------------------------------
def macro(ten=4.63, ten_chg=0.08, curve=0.44, vix=15.8, credit=2.75):
    return {
        "DGS10": {"value": ten, "change_1m": ten_chg},
        "T10Y2Y": {"value": curve, "change_1m": 0.0},
        "VIXCLS": {"value": vix, "change_1m": 0.0},
        "BAMLH0A0HYM2": {"value": credit, "change_1m": 0.0},
    }


def test_regime_reads_the_four_observable_states():
    got = monitor.regime(macro())
    assert got["available"] is True
    assert got["states"] == {"rates": "stable", "curve": "flat",
                             "volatility": "normal", "credit": "tight"}
    # VIX 15.8 against 2.75% high-yield spreads is an accommodating tape; credit is
    # the cleaner appetite gauge and carries the composite.
    assert got["risk_appetite"] == "risk-seeking"
    assert monitor.regime(macro(vix=12, credit=2.5))["states"]["volatility"] == "calm"


def test_regime_detects_stress():
    got = monitor.regime(macro(vix=32, credit=6.5, curve=-0.4, ten_chg=-0.4))
    assert got["states"]["volatility"] == "stressed"
    assert got["states"]["credit"] == "wide"
    assert got["states"]["curve"] == "inverted"
    assert got["states"]["rates"] == "falling"
    assert got["risk_appetite"] == "risk-averse"


def test_regime_is_unavailable_without_macro():
    assert monitor.regime(None)["available"] is False
    assert monitor.regime({})["available"] is False


def test_regime_emits_no_cycle_label_and_no_forecast():
    """A single "late-cycle" label reads as a forecast and would be wrong often enough
    to matter. Four measurements are honest; one interpretation is not."""
    got = monitor.regime(macro())
    # Scan the claims, not the disclaimer — the note is where "forecast" is allowed
    # to appear, and only in the negative.
    claims = str({k: v for k, v in got.items() if k != "note"}).lower()
    for banned in ("late-cycle", "early-cycle", "recession", "expect", "forecast",
                   "bullish", "bearish"):
        assert banned not in claims, f"regime output claims too much: {banned}"
    assert "not a forecast" in got["note"].lower()


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------
def ledger(rows, as_of=None, failures=()):
    stamp = as_of or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"as_of": stamp, "all": rows, "universe": len(rows),
            "price_failures": list(failures)}


def test_health_passes_on_a_sound_board():
    rows = [dict(r, signal=model.signal(r["conviction"])) for r in universe(60)]
    # spread the tiers so the board is genuinely discriminating
    for i, r in enumerate(rows):
        r["conviction"] = 20 + i
        r["signal"] = model.signal(r["conviction"])
    checks = monitor.health(ledger(rows), None, {"spec_segments": {"segments": []}})
    by = {c["name"]: c for c in checks}
    assert by["Data freshness"]["status"] == "pass"
    assert by["Score dispersion"]["status"] == "pass"
    assert by["Signal tiers populated"]["status"] == "pass"


def test_health_fails_a_degenerate_board():
    """The exact shape of the failure this project started from: every name identical."""
    rows = [dict(row(f"S{i}"), conviction=0, signal="AVOID") for i in range(50)]
    checks = monitor.health(ledger(rows), None, {"spec_segments": {"segments": []}})
    by = {c["name"]: c for c in checks}
    assert by["Score dispersion"]["status"] == "fail"
    assert by["Signal tiers populated"]["status"] == "fail"


def test_health_fails_a_stale_build():
    old = (datetime.now(timezone.utc) - timedelta(hours=90)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [dict(row(f"S{i}"), conviction=20 + i, signal=model.signal(20 + i))
            for i in range(40)]
    checks = monitor.health(ledger(rows, as_of=old), None, {"spec_segments": {"segments": []}})
    fresh = next(c for c in checks if c["name"] == "Data freshness")
    assert fresh["status"] == "fail"
    assert "missed" in fresh["detail"]


def test_health_reports_stability_as_pending_before_it_can_be_computed():
    rows = [dict(row(f"S{i}"), conviction=20 + i, signal=model.signal(20 + i))
            for i in range(40)]
    checks = monitor.health(ledger(rows), None, {"spec_segments": {"segments": []}})
    stab = next(c for c in checks if c["name"] == "Ranking stability")
    assert stab["status"] == "pending"
    assert "second snapshot" in stab["detail"]


def test_health_warns_when_the_ranking_reshuffles():
    rows = [dict(row(f"S{i}"), conviction=20 + i, signal=model.signal(20 + i))
            for i in range(40)]
    stab = {"rank_correlation": 0.40, "from": "2026-08-06", "mean_abs_move": 12.0,
            "tier_changes": 30, "spec_changed": False}
    checks = monitor.health(ledger(rows), stab, {"spec_segments": {"segments": []}})
    by = {c["name"]: c for c in checks}
    assert by["Ranking stability"]["status"] == "warn"
    assert by["Overnight move size"]["status"] == "warn"


def test_health_warns_when_history_spans_two_specifications():
    rows = [dict(row(f"S{i}"), conviction=20 + i, signal=model.signal(20 + i))
            for i in range(40)]
    cov = {"spec_segments": {"segments": [{"days": 5}, {"days": 3}]}}
    checks = monitor.health(ledger(rows), None, cov)
    seg = next(c for c in checks if c["name"] == "Specification history")
    assert seg["status"] == "warn"
    assert "segment" in seg["detail"]


def test_health_never_claims_predictive_accuracy():
    """A green panel says the machinery works, not that the signal has value.
    Conflating the two would be the most damaging thing this module could do."""
    rows = [dict(row(f"S{i}"), conviction=20 + i, signal=model.signal(20 + i))
            for i in range(40)]
    checks = monitor.health(ledger(rows), None, {"spec_segments": {"segments": []}})
    flat = str(checks).lower()
    for banned in ("accurate", "predictive", "alpha", "outperform", "information coefficient"):
        assert banned not in flat, f"health output claims predictive power: {banned}"


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def test_build_is_usable_on_the_very_first_run(tmp_path):
    rows = [dict(row(f"S{i}"), conviction=20 + i, signal=model.signal(20 + i))
            for i in range(40)]
    snap(tmp_path, "2026-08-07", rows)
    got = monitor.build(str(tmp_path), ledger(rows), macro())
    assert got["observations"] == 1
    assert got["stability"] is None
    assert got["regime"]["available"] is True
    assert any(c["status"] == "pending" for c in got["health"])
    assert "predict" in got["scope"].lower()


def test_build_computes_stability_once_two_snapshots_exist(tmp_path):
    rows = [dict(row(f"S{i}"), conviction=20 + i, signal=model.signal(20 + i))
            for i in range(40)]
    snap(tmp_path, "2026-08-06", rows)
    snap(tmp_path, "2026-08-07", rows)
    got = monitor.build(str(tmp_path), ledger(rows), macro())
    assert got["observations"] == 2
    assert got["stability"]["rank_correlation"] == pytest.approx(1.0)
    assert not any(c["status"] == "pending" for c in got["health"])


# ---------------------------------------------------------------------------
# the terminal's Monitor view
#
# monitor.build() is only useful if the terminal actually renders it. These tests read
# web/terminal.html the way test_parity.py does, so a metric added to the report but
# never wired into a panel fails here rather than shipping invisibly.
# ---------------------------------------------------------------------------
TERMINAL = Path(__file__).resolve().parents[1] / "web" / "terminal.html"


def test_terminal_has_a_monitor_view_wired_into_the_nav_and_boot():
    html = TERMINAL.read_text(encoding="utf-8")
    assert 'data-view="monitor"' in html
    assert 'id="view-monitor"' in html
    assert "monitor.json" in html, "the view never fetches the report"
    assert "renderMonitor()" in html, "the view is never rendered"


def test_terminal_renders_every_section_of_the_report(tmp_path):
    """Each top-level key monitor.build() emits must be consumed by the terminal."""
    rows = [dict(row(f"S{i}"), conviction=20 + i, signal=model.signal(20 + i))
            for i in range(40)]
    snap(tmp_path, "2026-08-06", rows)
    snap(tmp_path, "2026-08-07", rows)
    report = monitor.build(str(tmp_path), ledger(rows), macro())

    html = TERMINAL.read_text(encoding="utf-8")
    for key in ("health", "stability", "coverage", "regime", "observations", "scope"):
        assert key in report
        assert f".{key}" in html or f"'{key}'" in html or f'"{key}"' in html, \
            f"the report carries {key!r} but the terminal never reads it"

    for field in ("rank_correlation", "mean_abs_move", "tier_migration", "tier_changes",
                  "mean_confidence", "fully_measured", "by_sector", "spec_segments",
                  "risk_appetite", "first_observation"):
        assert field in html, f"{field!r} is computed nightly but never displayed"


def test_monitor_view_states_its_scope_and_claims_no_predictive_power():
    """The one failure mode of a green health panel is being read as 'the model works'."""
    html = TERMINAL.read_text(encoding="utf-8")
    view = html.split('id="view-monitor"', 1)[1].split('id="view-method"', 1)[0]
    lowered = view.lower()
    assert "predict" in lowered, "the view must state what it does not measure"
    for banned in ("accuracy", "outperform", "backtest", "forecast the", "expected return"):
        assert banned not in lowered, f"the monitor view claims {banned!r}"


def test_history_dependent_panels_declare_what_they_are_waiting_for():
    """A panel with no data yet must say so, not render a default that looks real."""
    html = TERMINAL.read_text(encoding="utf-8")
    for fn in ("renderStability", "renderMigration", "renderCovTrend"):
        body = html.split(f"function {fn}(", 1)[1].split("\nfunction ", 1)[0]
        assert "pending" in body, f"{fn} has no empty state"
        assert "snapshot" in body or "runs to be a trend" in body, \
            f"{fn} does not say what it is waiting for"
