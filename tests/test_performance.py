"""Paper return chained across recorded snapshots.

The tests that matter here are the ones about *when* the curve refuses to exist and
*which* weights it uses. A return series is the easiest artifact in this project to
make flattering by accident: use tonight's weights against tonight's prices and it
prints alpha every single day, forever, and looks entirely plausible while doing it.
"""
from __future__ import annotations

import json

import pytest

from equity_monitor import performance, snapshots


def snap(tmp_path, date, rows, bench=None):
    """Write a snapshot with explicit prices and weights."""
    payload = [dict(symbol=s, conviction=60, sector="Information Technology",
                    profile="default", price=p, weight=w)
               for s, p, w in rows]
    return snapshots.write(payload, str(tmp_path), on=date, as_of=date + "T23:00:00Z",
                           benchmark=bench)


@pytest.fixture
def ledger(tmp_path):
    (tmp_path / "snapshots").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# the look-ahead boundary
# ---------------------------------------------------------------------------
def test_a_leg_uses_the_earlier_nights_weights(ledger):
    """The whole point.

    A holds 100% of RISER, B holds 100% of FALLER. If the leg used the *later* night's
    weights it would report the faller's loss; using the published weights it reports
    the riser's gain, which is what a book following last night's list would have
    earned.
    """
    snap(ledger, "2026-01-01", [("RISER", 100.0, 100.0), ("FALLER", 100.0, 0.0)])
    snap(ledger, "2026-01-02", [("RISER", 110.0, 0.0), ("FALLER", 90.0, 100.0)])
    ls = performance.legs(str(ledger))
    assert len(ls) == 1
    assert ls[0]["book"] == pytest.approx(0.10)


def test_a_name_that_becomes_unpriceable_renormalises_rather_than_scoring_zero(ledger):
    """Dropping out of the price feed is not a 100% loss, and must not be booked as one."""
    snap(ledger, "2026-01-01", [("A", 100.0, 50.0), ("GONE", 100.0, 50.0)])
    snap(ledger, "2026-01-02", [("A", 110.0, 100.0)])
    leg = performance.legs(str(ledger))[0]
    assert leg["book"] == pytest.approx(0.10)      # not 0.05
    assert leg["weight_lost"] == pytest.approx(0.5)


def test_a_leg_losing_most_of_the_book_is_dropped_not_reported(ledger):
    """A data outage produces a return. It must not be mistaken for one."""
    snap(ledger, "2026-01-01", [("A", 100.0, 20.0), ("GONE", 100.0, 80.0)])
    snap(ledger, "2026-01-02", [("A", 110.0, 100.0)])
    assert performance.legs(str(ledger))[0]["usable"] is False
    assert performance.build(str(ledger))["legs"] == 0
    assert performance.build(str(ledger))["legs_dropped"] == 1


# ---------------------------------------------------------------------------
# the benchmark
# ---------------------------------------------------------------------------
def test_the_benchmark_comes_from_the_snapshot(ledger):
    snap(ledger, "2026-01-01", [("A", 100.0, 100.0)], bench={"symbol": "SPY", "price": 500.0})
    snap(ledger, "2026-01-02", [("A", 110.0, 100.0)], bench={"symbol": "SPY", "price": 505.0})
    leg = performance.legs(str(ledger))[0]
    assert leg["benchmark"] == pytest.approx(0.01)


def test_a_missing_benchmark_is_a_gap_not_a_flat_line(ledger):
    """Carrying the last value forward would draw a horizontal segment, which reads as
    'the index did not move' when the truth is 'the index was not recorded'."""
    snap(ledger, "2026-01-01", [("A", 100.0, 100.0)])
    snap(ledger, "2026-01-02", [("A", 110.0, 100.0)])
    out = performance.build(str(ledger))
    assert out["series"][1]["benchmark"] is None
    assert out["benchmark_available"] is False
    assert out["benchmark_total"] is None
    assert out["series"][1]["book"] == pytest.approx(10.0)
    # Including the origin. A zero there is a dot at break-even that nothing measured.
    assert out["series"][0]["benchmark"] is None
    assert out["series"][0]["book"] == 0.0


def test_the_equal_weight_control_is_computed_over_every_scored_name(ledger):
    """Including names the book does not hold — that is what makes it a control."""
    snap(ledger, "2026-01-01", [("HELD", 100.0, 100.0), ("UNHELD", 100.0, 0.0)])
    snap(ledger, "2026-01-02", [("HELD", 110.0, 100.0), ("UNHELD", 130.0, 0.0)])
    leg = performance.legs(str(ledger))[0]
    assert leg["book"] == pytest.approx(0.10)
    assert leg["equal_weight"] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# chaining and the render gate
# ---------------------------------------------------------------------------
def test_returns_compound_rather_than_summing(ledger):
    for i, (p, nxt) in enumerate([(100.0, 110.0), (110.0, 121.0)]):
        snap(ledger, f"2026-01-0{i+1}", [("A", p, 100.0)])
    snap(ledger, "2026-01-03", [("A", 121.0, 100.0)])
    out = performance.build(str(ledger))
    assert out["book_total"] == pytest.approx(21.0)     # not 20


def test_the_curve_refuses_to_render_below_the_threshold(ledger):
    """Two points joined by a line read as a trend. They are one observation."""
    snap(ledger, "2026-01-01", [("A", 100.0, 100.0)])
    snap(ledger, "2026-01-02", [("A", 110.0, 100.0)])
    out = performance.build(str(ledger))
    assert out["renderable"] is False
    assert out["days"] == 2 and out["min_days"] == performance.MIN_DAYS


def test_the_curve_renders_once_the_threshold_is_reached(ledger):
    prices = [100.0, 101.0, 102.0, 103.0, 104.0]
    for i, p in enumerate(prices):
        snap(ledger, f"2026-01-0{i+1}", [("A", p, 100.0)])
    out = performance.build(str(ledger))
    assert out["days"] == performance.MIN_DAYS
    assert out["renderable"] is True
    assert len(out["series"]) == performance.MIN_DAYS


def test_the_gate_counts_measured_legs_not_calendar_dates(ledger):
    """Five snapshots with four unusable legs is not five days of measurement."""
    snap(ledger, "2026-01-01", [("A", 100.0, 5.0), ("GONE", 100.0, 95.0)])
    for i, p in enumerate([101.0, 102.0, 103.0, 104.0]):
        snap(ledger, f"2026-01-0{i+2}", [("A", p, 100.0)])
    out = performance.build(str(ledger))
    assert out["days"] == 5
    assert out["legs_dropped"] == 1
    assert out["renderable"] is False


# ---------------------------------------------------------------------------
# the specification boundary
# ---------------------------------------------------------------------------
def test_a_specification_change_inside_the_window_is_reported(ledger, monkeypatch):
    snap(ledger, "2026-01-01", [("A", 100.0, 100.0)])
    monkeypatch.setattr("equity_monitor.model.spec_hash", lambda: "deadbeefcafe")
    snap(ledger, "2026-01-02", [("A", 110.0, 100.0)])
    out = performance.build(str(ledger))
    assert out["spec_stable"] is False
    assert len(out["spec_hashes"]) == 2        # both sides of the boundary
    assert performance.legs(str(ledger))[0]["spec_changed"] is True


# ---------------------------------------------------------------------------
# no history at all
# ---------------------------------------------------------------------------
def test_an_empty_ledger_reports_nothing_rather_than_zero(ledger):
    out = performance.build(str(ledger))
    assert out["days"] == 0 and out["legs"] == 0
    assert out["book_total"] is None
    assert out["renderable"] is False
    assert out["series"] == []


def test_write_round_trips(ledger):
    snap(ledger, "2026-01-01", [("A", 100.0, 100.0)])
    snap(ledger, "2026-01-02", [("A", 110.0, 100.0)])
    written = performance.write(str(ledger))
    on_disk = json.loads((ledger / "performance.json").read_text())
    assert on_disk == written


# ---------------------------------------------------------------------------
# the deploy gate
# ---------------------------------------------------------------------------
def test_the_validator_rejects_a_curve_that_claims_to_be_renderable_early(ledger):
    """The gate exists because the flag is one refactor away from lying, and this is
    the only artifact in the ledger that states a result rather than a condition."""
    import scripts.validate_ledger as v
    snap(ledger, "2026-01-01", [("A", 100.0, 100.0)])
    snap(ledger, "2026-01-02", [("A", 110.0, 100.0)])
    performance.write(str(ledger))
    assert v.check_performance(str(ledger)) == []

    doctored = json.loads((ledger / "performance.json").read_text())
    doctored["renderable"] = True
    (ledger / "performance.json").write_text(json.dumps(doctored))
    assert any("renderable" in p for p in v.check_performance(str(ledger)))


def test_the_validator_rejects_benchmark_points_on_an_unavailable_benchmark(ledger):
    import scripts.validate_ledger as v
    snap(ledger, "2026-01-01", [("A", 100.0, 100.0)])
    snap(ledger, "2026-01-02", [("A", 110.0, 100.0)])
    performance.write(str(ledger))
    doctored = json.loads((ledger / "performance.json").read_text())
    doctored["series"][0]["benchmark"] = 0.0
    (ledger / "performance.json").write_text(json.dumps(doctored))
    assert any("benchmark" in p for p in v.check_performance(str(ledger)))


def test_a_missing_performance_file_is_not_a_failure(tmp_path):
    import scripts.validate_ledger as v
    assert v.check_performance(str(tmp_path)) == []


def test_the_origin_is_the_first_measured_night_not_the_first_recorded_one(ledger):
    """Latent here until a leg is dropped, and immediate on the crypto ledger, whose
    first three legs lose too much of the book to be returns. Anchoring the series at
    the first recorded date draws a segment across the gap and attributes one night's
    move to all of it."""
    snap(ledger, "2026-01-01", [("A", 100.0, 20.0), ("GONE", 100.0, 80.0)])
    for i, p in enumerate([101.0, 102.0, 103.0, 104.0]):
        snap(ledger, f"2026-01-0{i+2}", [("A", p, 100.0)])
    out = performance.build(str(ledger))
    assert out["legs_dropped"] == 1
    assert out["from"] == "2026-01-02"            # not 2026-01-01
    assert out["recorded_from"] == "2026-01-01"
    assert out["series"][0]["date"] == "2026-01-02"
    assert out["series"][0]["book"] == 0.0


# ---------------------------------------------------------------------------
# days on which nothing traded
# ---------------------------------------------------------------------------
def board(n, price):
    """n names at one price, equally weighted — enough to clear MIN_NAMES_FOR_STALE."""
    return [(f"A{i}", price, 100.0 / n) for i in range(n)]


def test_a_snapshot_pair_with_no_price_movement_at_all_is_not_a_holding_period(ledger):
    """The real cause: a push-triggered workflow re-ran the build on a Saturday, so
    2026-08-08 and 2026-08-09 both carried Friday's closes. Chained into the curve that
    is a 0.00% leg on a day the market never opened — it dilutes the return series and
    inflates the day count the render gate reads."""
    snap(ledger, "2026-08-07", board(10, 100.0))
    snap(ledger, "2026-08-08", board(10, 101.0))
    snap(ledger, "2026-08-09", board(10, 101.0))     # Sunday: Friday's closes again
    ls = performance.legs(str(ledger))
    assert [l["stale"] for l in ls] == [False, True]
    assert [l["usable"] for l in ls] == [True, False]

    out = performance.build(str(ledger))
    assert out["legs"] == 1
    assert out["legs_dropped_stale"] == 1
    assert out["to"] == "2026-08-08"          # the curve stops where measurement stops


def test_one_unchanged_name_on_a_real_day_is_not_staleness(ledger):
    """A quiet name is ordinary. Only the whole board frozen to the cent is a calendar
    artifact, and the difference has to survive a single flat ticker."""
    a = board(10, 100.0)
    b = [(s, 101.0, w) for s, _, w in a]
    b[0] = (b[0][0], 100.0, b[0][2])          # one name genuinely unchanged
    snap(ledger, "2026-08-07", a)
    snap(ledger, "2026-08-08", b)
    ls = performance.legs(str(ledger))
    assert ls[0]["stale"] is False and ls[0]["usable"] is True


def test_too_few_shared_names_cannot_establish_staleness(ledger):
    """Below the threshold an identical pair is as likely to be a thin board as a shut
    market, and guessing wrong here silently deletes a real leg."""
    snap(ledger, "2026-08-07", board(2, 100.0))
    snap(ledger, "2026-08-08", board(2, 100.0))
    assert performance.legs(str(ledger))[0]["stale"] is False


def test_staleness_is_detected_from_prices_not_a_trading_calendar(ledger):
    """A calendar would have to be maintained, knows nothing about other venues, and
    would not have caught this bug anyway — the snapshots were mis-dated relative to
    the closes they carried, so the dates looked entirely plausible."""
    snap(ledger, "2026-08-11", board(10, 100.0))     # a Tuesday
    snap(ledger, "2026-08-12", board(10, 100.0))     # a Wednesday, identical prices
    assert performance.legs(str(ledger))[0]["stale"] is True
