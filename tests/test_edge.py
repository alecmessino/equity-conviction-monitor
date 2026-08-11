"""Whether conviction predicts the next session, and reporting a null as a null.

Ported from the crypto sibling, where the question first mattered. That basket is
losing to its equal-weight control by ~283bp, which reads as proof the selection
subtracts value; the information coefficient over the same legs is +0.006 with an
interval spanning zero. A concentrated book with no *measurable* edge underperforms an
equal-weight control as a matter of course, because concentration adds variance without
adding expected return.

Equity has one usable leg, so nothing is measurable here yet and the honest output says
so. These tests pin the behaviour now, while the answer is "nothing", precisely because
that is when a panel is most likely to quietly start claiming something.
"""
import pytest

from equity_monitor import edge, performance, snapshots


def board(n, price, conv=lambda i: 100 - i):
    return [dict(symbol=f"A{i:02d}", conviction=conv(i),
                 sector="Information Technology", profile="default",
                 price=price(i) if callable(price) else price, weight=100.0 / n)
            for i in range(n)]


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
# the measurement
# ---------------------------------------------------------------------------
def test_a_perfectly_predictive_score_scores_ic_one(ledger):
    """Highest conviction earns the best return, monotonically."""
    snap(ledger, "2026-03-02", board(30, 100.0), session="2026-03-02")
    snap(ledger, "2026-03-03", board(30, lambda i: 100.0 * (1 + (30 - i) / 100.0)),
         session="2026-03-03")
    ls = edge.legs(str(ledger))
    assert ls and ls[0]["ic"] == pytest.approx(1.0)
    assert ls[0]["spread_bp"] > 0


def test_an_inverted_score_scores_ic_minus_one(ledger):
    """An inverted ranking is a usable signal read backwards. It must never be reported
    as 'no relationship' — the remedies are opposite."""
    snap(ledger, "2026-03-02", board(30, 100.0), session="2026-03-02")
    snap(ledger, "2026-03-03", board(30, lambda i: 100.0 * (1 + i / 100.0)),
         session="2026-03-03")
    assert edge.legs(str(ledger))[0]["ic"] == pytest.approx(-1.0)


def test_a_thin_board_is_skipped_rather_than_measured(ledger):
    """A rank correlation over a handful of names is noise with a decimal point."""
    snap(ledger, "2026-03-02", board(5, 100.0), session="2026-03-02")
    snap(ledger, "2026-03-03", board(5, 110.0), session="2026-03-03")
    assert edge.legs(str(ledger)) == []


def test_it_reuses_the_curve_definition_of_a_usable_leg(ledger):
    """A second definition of 'which snapshot pairs are a holding period' is a second
    thing to keep in step, and it would drift silently because both would still produce
    plausible numbers. A stale pair is excluded here for the same reason it is there."""
    snap(ledger, "2026-03-02", board(30, 100.0), session="2026-03-02")
    snap(ledger, "2026-03-03", board(30, 100.0), session="2026-03-02")   # same session
    assert [l["stale"] for l in performance.legs(str(ledger))] == [True]
    assert edge.legs(str(ledger)) == []


def test_legs_are_session_dated(ledger):
    snap(ledger, "2026-03-03", board(30, 100.0), session="2026-03-02")
    snap(ledger, "2026-03-04", board(30, 101.0), session="2026-03-03")
    leg = edge.legs(str(ledger))[0]
    assert (leg["from"], leg["to"]) == ("2026-03-02", "2026-03-03")
    assert leg["session_dated"] is True


# ---------------------------------------------------------------------------
# reporting a null as a null
# ---------------------------------------------------------------------------
def test_one_leg_measures_nothing_and_says_so(ledger):
    """The equity ledger's actual state. An interval cannot be formed from a single
    observation, and inventing one would be the whole failure this module guards."""
    snap(ledger, "2026-03-02", board(30, 100.0), session="2026-03-02")
    snap(ledger, "2026-03-03", board(30, 101.0), session="2026-03-03")
    out = edge.build(str(ledger))
    assert out["legs"] == 1
    assert out["measurable"] is False
    assert out["mean_ic"] is None and out["ci"] is None
    assert "Not enough" in out["verdict"]


def test_a_handful_of_legs_is_not_an_edge(ledger):
    """Five legs of noise produce a mean IC that is not zero. The interval is what
    stops it being read as a finding."""
    import random
    rng = random.Random(11)
    for i in range(6):
        d = f"2026-03-{i+2:02d}"
        snap(ledger, d, board(30, lambda j: 100.0 * (1 + rng.uniform(-.03, .03))),
             session=d)
    out = edge.build(str(ledger))
    assert out["legs"] >= 2
    assert out["measurable"] is False, "a few legs of noise must never read as an edge"
    assert out["ci"][0] < out["mean_ic"] < out["ci"][1]
    assert "neither evidence" in out["verdict"]


def test_the_sample_size_still_needed_is_stated(ledger):
    """Without it, 'not measurable yet' is indistinguishable from 'never will be'."""
    for i in range(4):
        d = f"2026-03-{i+2:02d}"
        snap(ledger, d, board(30, lambda j: 100.0 + i + j * 0.1), session=d)
    need = edge.build(str(ledger))["legs_needed"]
    assert set(need) == {"0.02", "0.03", "0.05"}
    assert need["0.02"] > need["0.03"] > need["0.05"]


def test_the_realised_gap_travels_with_the_null(ledger):
    """Shown apart, each number invites the wrong conclusion: the gap looks like proof
    of a bad model, the null looks like the gap does not matter."""
    snap(ledger, "2026-03-02", board(30, 100.0), session="2026-03-02")
    snap(ledger, "2026-03-03", board(30, 101.0), session="2026-03-03")
    out = edge.build(str(ledger))
    assert "book_total" in out and "equal_weight_total" in out


def test_an_empty_ledger_is_not_an_error(tmp_path):
    (tmp_path / "snapshots").mkdir()
    out = edge.build(str(tmp_path))
    assert out["legs"] == 0 and out["measurable"] is False


def test_the_live_ledger_reports_no_measurable_edge():
    """The regression that matters: if this ever claims a measurable edge on the real
    ledger's handful of legs, the interval logic has broken."""
    out = edge.build()
    assert out["measurable"] is False
