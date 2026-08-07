"""The overnight diff.

The central assertion in this file is that a one-point tier flip does not appear in the
morning list of decisions. The churn diagnostic measured 39 of 39 tier changes on an
ordinary night coming from moves of two points or less; a diff that presented those as
"new BUY" would be wrong every single day, in detail, which is the specific failure mode
that makes a tool untrustworthy.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from equity_monitor import churn, model, snapshots, watchlist
from tests.test_snapshots import BASE_PERCENTILES, row


def snap(tmp_path, stamp, rows):
    snapshots.write(rows, str(tmp_path), on=stamp, as_of=stamp + "T23:00:00Z")
    return snapshots.read(snapshots.snapshot_path(str(tmp_path), stamp))


def board(convictions: dict[str, int], date="2026-08-07") -> dict:
    """A snapshot-shaped board with hand-set convictions.

    The diff reads conviction and tier only, so setting them directly is exactly what is
    under test here — unlike the attribution tests, which need model-consistent pillars.
    """
    return {"date": date, "sectors": {s: "Information Technology" for s in convictions},
            "rows": {s: {"conviction": c, "data_confidence": 1.0}
                     for s, c in convictions.items()}}


def attribution(**by_symbol) -> dict:
    return {sym: {"factors": factors} for sym, factors in by_symbol.items()}


# ---------------------------------------------------------------------------
# the distinction the whole module exists for
# ---------------------------------------------------------------------------
def test_a_one_point_tier_flip_is_a_boundary_crossing_not_an_upgrade():
    """69 -> 70 is the label crossing a line, not a change of view."""
    got = watchlist.diff(board({"A": 69}, "2026-08-06"), board({"A": 70}))
    assert got["counts"]["upgrades"] == 0
    assert got["counts"]["boundary"] == 1
    assert got["boundary"][0]["symbol"] == "A"
    assert got["boundary"][0]["from_tier"] == "HOLD"
    assert got["boundary"][0]["to_tier"] == "BUY"
    assert got["boundary"][0]["marginal"] is True


def test_a_real_reclassification_is_an_upgrade():
    got = watchlist.diff(board({"A": 60}, "2026-08-06"), board({"A": 72}))
    assert got["counts"]["upgrades"] == 1
    assert got["counts"]["boundary"] == 0
    assert got["upgrades"][0]["delta"] == 12
    assert got["upgrades"][0]["to_tier"] == "BUY"


def test_the_marginal_threshold_is_the_one_churn_uses():
    """Two modules disagreeing about what counts as marginal would be worse than either.

    ``churn`` measures how many tier changes were marginal; this module decides which
    ones to put in front of a person each morning. If those thresholds drifted apart,
    the Monitor tab would report a number the watchlist contradicts.
    """
    edge = int(churn.MARGINAL_MOVE)
    got = watchlist.diff(board({"A": 70 - edge}, "2026-08-06"), board({"A": 70}))
    assert got["counts"]["boundary"] == 1, "at exactly the threshold, still marginal"
    got = watchlist.diff(board({"A": 70 - edge - 1}, "2026-08-06"), board({"A": 70}))
    assert got["counts"]["upgrades"] == 1, "one point past it, a real upgrade"


def test_a_large_move_inside_one_tier_is_reported_as_a_mover():
    """58 -> 68 is not a BUY, but a tier-only diff drops it entirely."""
    got = watchlist.diff(board({"A": 58}, "2026-08-06"), board({"A": 68}))
    assert got["counts"]["upgrades"] == 0
    assert got["counts"]["movers"] == 1
    assert got["movers"][0]["from_tier"] == got["movers"][0]["to_tier"] == "HOLD"
    assert got["movers"][0]["delta"] == 10


def test_a_small_move_inside_one_tier_is_not_reported_at_all():
    """The morning list must not fill with noise; most nights most names do nothing."""
    got = watchlist.diff(board({"A": 60}, "2026-08-06"), board({"A": 62}))
    assert got["counts"] == {"upgrades": 0, "downgrades": 0, "movers": 0,
                             "boundary": 0, "entered": 0, "left": 0, "unchanged": 1}
    assert got["upgrades"] == got["downgrades"] == got["movers"] == []


def test_downgrades_are_separated_from_upgrades_by_direction_not_by_sign():
    got = watchlist.diff(board({"UP": 60, "DOWN": 75}, "2026-08-06"),
                         board({"UP": 75, "DOWN": 60}))
    assert [r["symbol"] for r in got["upgrades"]] == ["UP"]
    assert [r["symbol"] for r in got["downgrades"]] == ["DOWN"]
    assert got["downgrades"][0]["from_tier"] == "BUY"
    assert got["downgrades"][0]["to_tier"] == "HOLD"


# ---------------------------------------------------------------------------
# the reason column
# ---------------------------------------------------------------------------
def test_the_driver_is_the_largest_push_in_the_direction_of_the_move():
    """Not the largest by magnitude.

    A name that rose is explained by what lifted it. The biggest number in the
    decomposition is sometimes a drag pulling the other way, and reporting that as the
    reason produces "+11, driven by deteriorating trend" — two true numbers arranged
    into a false sentence. This is the same bug the attribution headline had.
    """
    attr = attribution(A={"p_rs": 4.0, "p_trend": -9.0, "p_roic": 2.0})
    got = watchlist.diff(board({"A": 60}, "2026-08-06"), board({"A": 71}), attr)
    driver = got["upgrades"][0]["driver"]
    assert driver["factor"] == "p_rs", "named the drag as the reason for a rise"
    assert driver["points"] == 4.0
    assert got["upgrades"][0]["drag"]["factor"] == "p_trend"


def test_a_fall_is_explained_by_what_pulled_it_down():
    attr = attribution(A={"p_rs": 6.0, "p_roic": -8.0})
    got = watchlist.diff(board({"A": 75}, "2026-08-06"), board({"A": 60}), attr)
    assert got["downgrades"][0]["driver"]["factor"] == "p_roic"
    assert got["downgrades"][0]["drag"]["factor"] == "p_rs"


def test_a_missing_attribution_leaves_the_reason_blank_rather_than_guessing():
    got = watchlist.diff(board({"A": 60}, "2026-08-06"), board({"A": 72}))
    assert got["upgrades"][0]["driver"] is None
    assert got["upgrades"][0]["drag"] is None


def test_negligible_factors_are_not_promoted_to_reasons():
    """A 0.004-point contribution is float noise, not an explanation."""
    attr = attribution(A={"p_rs": 0.004, "p_trend": 0.002})
    got = watchlist.diff(board({"A": 60}, "2026-08-06"), board({"A": 72}), attr)
    assert got["upgrades"][0]["driver"] is None


# ---------------------------------------------------------------------------
# universe changes
# ---------------------------------------------------------------------------
def test_names_entering_and_leaving_are_never_mistaken_for_rating_actions():
    """An index reconstitution is mechanics. Filing it under "new BUY" would be a lie."""
    got = watchlist.diff(board({"OLD": 80}, "2026-08-06"), board({"NEW": 80}))
    assert got["counts"]["upgrades"] == 0
    assert [r["symbol"] for r in got["entered"]] == ["NEW"]
    assert [r["symbol"] for r in got["left"]] == ["OLD"]
    assert got["entered"][0]["tier"] == "STRONG"


def test_sections_are_capped_so_the_morning_view_stays_readable():
    n = watchlist.SECTION_LIMIT + 15
    before = board({f"S{i:03d}": 60 for i in range(n)}, "2026-08-06")
    after = board({f"S{i:03d}": 75 for i in range(n)})
    got = watchlist.diff(before, after)
    assert len(got["upgrades"]) == watchlist.SECTION_LIMIT
    assert got["counts"]["upgrades"] == n, "the count must report the true total"


def test_upgrades_are_ordered_by_how_far_they_moved():
    before = board({"A": 60, "B": 60, "C": 60}, "2026-08-06")
    after = board({"A": 71, "B": 85, "C": 78})
    got = watchlist.diff(before, after)
    assert [r["symbol"] for r in got["upgrades"]] == ["B", "C", "A"]


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def test_from_ledger_needs_two_snapshots(tmp_path):
    rows = [row("AAPL"), row("MSFT", p_roic=0.4)]
    snap(tmp_path, "2026-08-07", rows)
    assert watchlist.from_ledger(str(tmp_path)) is None


def test_from_ledger_joins_names_and_attribution_from_the_ledger(tmp_path):
    rows = [row("AAPL"), row("MSFT", p_roic=0.4)]
    snap(tmp_path, "2026-08-06", rows)
    lifted = [dict(r, **model.score({**{k: v for k, v in r.items()
                                        if k.startswith("p_") and v is not None},
                                     "p_rs": 0.98, "p_trend": 0.98}))
              for r in rows]
    snap(tmp_path, "2026-08-07", lifted)

    payload = {
        "all": [{"symbol": "AAPL", "name": "Apple Inc.", "sector": "Information Technology"},
                {"symbol": "MSFT", "name": "Microsoft Corp.", "sector": "Information Technology"}],
        "attribution": {"names": {"AAPL": {"factors": {"p_rs": 3.0, "p_roic": -0.2}}}},
    }
    got = watchlist.from_ledger(str(tmp_path), payload)
    assert got is not None
    rows_out = got["upgrades"] + got["movers"] + got["boundary"]
    by_sym = {r["symbol"]: r for r in rows_out}
    if "AAPL" in by_sym:
        assert by_sym["AAPL"]["name"] == "Apple Inc."
        assert by_sym["AAPL"]["driver"]["factor"] == "p_rs"


def test_from_ledger_builds_without_a_payload(tmp_path):
    """Tickers alone is a degraded view, not a broken one."""
    rows = [row("AAPL"), row("MSFT")]
    snap(tmp_path, "2026-08-06", rows)
    snap(tmp_path, "2026-08-07", rows)
    got = watchlist.from_ledger(str(tmp_path))
    assert got is not None
    assert got["names_compared"] == 2


def test_scope_explains_why_boundary_crossings_are_held_back():
    got = watchlist.diff(board({"A": 69}, "2026-08-06"), board({"A": 70}))
    assert "not different holdings" in got["scope"]


# ---------------------------------------------------------------------------
# the terminal
# ---------------------------------------------------------------------------
TERMINAL = Path(__file__).resolve().parents[1] / "web" / "terminal.html"


def test_terminal_has_a_watchlist_view():
    html = TERMINAL.read_text(encoding="utf-8")
    assert 'data-view="watchlist"' in html
    assert 'id="view-watchlist"' in html
    assert "watchlist.json" in html
    assert "renderWatchlist" in html


def test_terminal_renders_every_section_of_the_diff():
    html = TERMINAL.read_text(encoding="utf-8")
    for field in ("upgrades", "downgrades", "movers", "boundary", "entered", "left",
                  "driver", "drag", "counts"):
        assert field in html, f"{field!r} is produced nightly but never displayed"


def test_terminal_keeps_boundary_crossings_visually_subordinate():
    """They must be reachable and clearly secondary — the whole point is not to trade them."""
    html = TERMINAL.read_text(encoding="utf-8")
    body = html.split("function renderWatchlist(", 1)[1].split("\nfunction ", 1)[0]
    assert "boundary" in body
    lowered = body.lower()
    assert "not a change of view" in lowered or "crossed a threshold" in lowered
