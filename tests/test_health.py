"""Cohort stickiness, persistence and tier flips — from snapshots already recorded.

No schema extension was needed for any of this. Every snapshot already carries
per-symbol conviction, price and weight on a dated file. Storing a derived "cohort"
field would have persisted a value that can be recomputed exactly, and stored values
drift from their definitions the moment the definition changes.

Two things pinned hardest, both learned the expensive way in this project:

* **The axis must be uniform.** Session dates and filenames cannot be mixed — the
  2026-08-11 snapshot records session 2026-08-10, which collides with the 2026-08-10
  snapshot's filename, and five boards on disk became four with nothing reporting it.

* **Ties must not decide the cohort.** Conviction is a rounded integer over ~1,000
  names with only eleven at 80 or above, so a top-ten cohort sits on a wall of ties and
  would churn on sort order rather than on anything the model did.
"""
import pytest

from equity_monitor import health, snapshots


def board(n, conv):
    return [dict(symbol=f"A{i:04d}", conviction=conv(i), sector="Information Technology",
                 profile="default", price=100.0, weight=100.0 / n) for i in range(n)]


@pytest.fixture
def ledger(tmp_path):
    (tmp_path / "snapshots").mkdir()
    return tmp_path


def snap(ledger, date, rows, session=None):
    return snapshots.write(rows, str(ledger), on=date, as_of=date + "T23:00:00Z",
                           session=None if session is None else
                           {"session": session, "coverage": 1.0,
                            "symbols": len(rows), "spread": {}})


# ---------------------------------------------------------------------------
# the axis
# ---------------------------------------------------------------------------
def test_a_session_date_never_collides_with_a_filename(ledger):
    """The real bug: 2026-08-11's session is 2026-08-10, which is also a filename. Keyed
    naively, one board overwrites the other and the count silently drops."""
    snap(ledger, "2026-08-10", board(60, lambda i: 90 - i), session=None)
    snap(ledger, "2026-08-11", board(60, lambda i: 90 - i), session="2026-08-10")
    _, axis, dated = health._by_session(str(ledger))
    assert len(axis) == 2, "a board was lost to a key collision"
    assert dated == 1


def test_the_axis_is_sessions_only_when_every_board_has_one(ledger):
    snap(ledger, "2026-03-02", board(60, lambda i: 90 - i), session="2026-03-01")
    snap(ledger, "2026-03-03", board(60, lambda i: 90 - i), session="2026-03-02")
    _, axis, dated = health._by_session(str(ledger))
    assert axis == ["2026-03-01", "2026-03-02"] and dated == 2
    assert health.stickiness(str(ledger))["dates_are_sessions"] is True


def test_a_mixed_ledger_falls_back_to_filenames_uniformly(ledger):
    """An axis uniformly a day late is worse than a session axis and better than one
    that is half each — and the flag says which it is."""
    snap(ledger, "2026-03-02", board(60, lambda i: 90 - i), session=None)
    snap(ledger, "2026-03-03", board(60, lambda i: 90 - i), session="2026-03-02")
    _, axis, _ = health._by_session(str(ledger))
    assert axis == ["2026-03-02", "2026-03-03"]
    assert health.stickiness(str(ledger))["dates_are_sessions"] is False


# ---------------------------------------------------------------------------
# stickiness
# ---------------------------------------------------------------------------
def test_an_unchanged_leading_cohort_is_fully_sticky(ledger):
    snap(ledger, "2026-03-02", board(300, lambda i: 900 - i), session="2026-03-02")
    snap(ledger, "2026-03-03", board(300, lambda i: 900 - i), session="2026-03-03")
    st = health.stickiness(str(ledger))
    assert st["stickiness"] == 1.0 and st["pairs"] == 1


def test_ties_do_not_decide_the_cohort(ledger):
    """Every name scores identically, so membership is entirely a tie-break. It must be
    reproducible rather than flickering on dict order — otherwise the metric measures
    the sort, not the model."""
    snap(ledger, "2026-03-02", board(300, lambda i: 70), session="2026-03-02")
    snap(ledger, "2026-03-03", board(300, lambda i: 70), session="2026-03-03")
    assert health.stickiness(str(ledger))["stickiness"] == 1.0


def test_the_cohort_is_a_decile_not_a_fixed_ten(ledger):
    """With ~1,000 names and conviction rounded to integers, a ten-name cohort sits on a
    wall of ties. A decile dilutes that to a few names at its boundary."""
    snap(ledger, "2026-03-02", board(300, lambda i: 900 - i), session="2026-03-02")
    snap(ledger, "2026-03-03", board(300, lambda i: 900 - i), session="2026-03-03")
    st = health.stickiness(str(ledger))
    assert st["latest"]["of"] == 30           # 10% of 300
    assert st["cohort_share"] == health.COHORT_SHARE


def test_a_reordered_cohort_is_not_sticky(ledger):
    snap(ledger, "2026-03-02", board(300, lambda i: 900 - i), session="2026-03-02")
    snap(ledger, "2026-03-03", board(300, lambda i: i), session="2026-03-03")
    assert health.stickiness(str(ledger))["stickiness"] == 0.0


def test_a_stale_pair_is_not_counted_as_perfectly_sticky(ledger):
    """A weekend pair is the same board twice. Scoring it 100% would flatter the number
    for a session on which the market never opened."""
    snap(ledger, "2026-03-02", board(300, lambda i: 900 - i), session="2026-03-02")
    snap(ledger, "2026-03-03", board(300, lambda i: 900 - i), session="2026-03-02")
    assert health.stickiness(str(ledger))["pairs"] == 0


def test_one_session_cannot_produce_a_stickiness(ledger):
    snap(ledger, "2026-03-02", board(300, lambda i: 900 - i), session="2026-03-02")
    st = health.stickiness(str(ledger))
    assert st["stickiness"] is None and st["latest"] is None


def test_the_threshold_is_a_convention_not_a_historical_norm(ledger):
    snap(ledger, "2026-03-02", board(300, lambda i: 900 - i), session="2026-03-02")
    assert "historical norm" in health.stickiness(str(ledger))["basis"]


# ---------------------------------------------------------------------------
# flips
# ---------------------------------------------------------------------------
def test_crossing_the_backed_line_is_counted_in_both_directions(ledger):
    snap(ledger, "2026-03-02", board(60, lambda i: 75 if i < 5 else 40),
         session="2026-03-02")
    snap(ledger, "2026-03-03", board(60, lambda i: 40 if i < 5 else (75 if i < 9 else 40)),
         session="2026-03-03")
    f = health.flips(str(ledger))
    assert f["n_out"] == 5 and f["n_into"] == 4
    assert f["out_of_backed"][0] == "A0000"


def test_symbols_travel_with_the_counts(ledger):
    """A count you cannot act on is a decoration."""
    snap(ledger, "2026-03-02", board(60, lambda i: 40), session="2026-03-02")
    snap(ledger, "2026-03-03", board(60, lambda i: 75 if i < 3 else 40),
         session="2026-03-03")
    assert health.flips(str(ledger))["into_backed"] == ["A0000", "A0001", "A0002"]


def test_a_single_session_reports_pending_rather_than_zero(ledger):
    """Zero flips and no comparison available are different states."""
    snap(ledger, "2026-03-02", board(60, lambda i: 75), session="2026-03-02")
    assert health.flips(str(ledger))["pending"] is True


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def test_a_name_seen_once_cannot_outrank_one_backed_across_the_window(ledger):
    for i, d in enumerate(["2026-03-02", "2026-03-03", "2026-03-04"]):
        rows = board(60, lambda j: 75 if j < 3 else 40)
        if i == 2:
            rows.append(dict(symbol="ONCE", conviction=95, sector="Information Technology",
                             profile="default", price=100.0, weight=0.0))
        snap(ledger, d, rows, session=d)
    p = health.persistence(str(ledger))
    once = next(r for r in p["rows"] if r["symbol"] == "ONCE")
    steady = next(r for r in p["rows"] if r["symbol"] == "A0000")
    assert once["share_above"] == 1.0 and steady["share_above"] == 1.0
    assert steady["persistence"] > once["persistence"]
    assert p["rows"][0]["symbol"] != "ONCE"


def test_absence_is_a_distinct_cell_from_a_low_score(ledger):
    """A board small enough that MAX_NAMES cannot truncate the departed name out of the
    result — otherwise the test passes or fails on the row limit rather than on the
    thing it is checking."""
    snap(ledger, "2026-03-02", board(25, lambda i: 75), session="2026-03-02")
    snap(ledger, "2026-03-03", board(25, lambda i: 75)[:-1], session="2026-03-03")
    p = health.persistence(str(ledger))
    assert len(p["rows"]) <= health.MAX_NAMES
    gone = next(r for r in p["rows"] if r["symbol"] == "A0024")
    assert gone["cells"] == [75.0, None]
    present = next(r for r in p["rows"] if r["symbol"] == "A0000")
    assert present["cells"] == [75.0, 75.0]


def test_the_window_travels_with_the_result(ledger):
    """Nine of eleven and nine of ninety are different claims."""
    for d in ["2026-03-02", "2026-03-03"]:
        snap(ledger, d, board(60, lambda i: 75), session=d)
    assert health.persistence(str(ledger))["window"] == 2


# ---------------------------------------------------------------------------
# live data
# ---------------------------------------------------------------------------
def test_the_live_ledger_produces_a_usable_ribbon():
    h = health.build()
    assert h["persistence"]["rows"], "persistence is empty on the real ledger"
    assert h["stickiness"]["sessions"] >= len(snapshots.available(health.LEDGER)) - 1


def test_an_empty_ledger_is_not_an_error(tmp_path):
    (tmp_path / "snapshots").mkdir()
    h = health.build(str(tmp_path))
    assert h["stickiness"]["stickiness"] is None
    assert h["persistence"]["rows"] == []
    assert h["flips"]["pending"] is True
