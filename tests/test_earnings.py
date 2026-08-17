"""The earnings calendar: what it derives, and what it refuses to claim.

Every test here corresponds to a way this module was wrong before it shipped. Two of
them were silent — the calendar rendered, the numbers looked like numbers, and both were
measuring something other than what the label said.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from equity_monitor import earnings

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = ROOT / "web" / "terminal.html"
WORKFLOW = ROOT / ".github" / "workflows" / "pages.yml"


# ---------------------------------------------------------------------------
# index parsing — the two formats EDGAR actually publishes
# ---------------------------------------------------------------------------
QUARTERLY_IDX = """Description:           Master Index of EDGAR Dissemination Feed

Form Type   Company Name                                                  CIK         Date Filed  File Name
---------------------------------------------------------------------------------------------------------
10-Q             1 800 FLOWERS COM INC                                         1084869     2026-05-07  edgar/data/1084869/0001084869-26-000019.txt
8-K              APPLE INC                                                     320183      2026-05-01  edgar/data/320183/0000320193-26-000052.txt
4                SOME INSIDER                                                  999999      2026-05-02  edgar/data/999999/0000999999-26-000001.txt
"""

# The daily file writes the date without dashes. A reader that looks for an ISO date
# finds none here and matches inside the accession number in the path instead.
DAILY_IDX = """Form Type   Company Name                                                  CIK
      Date Filed  File Name
---------------------------------------------------------------------------------------------------------
10-K             WESTERN DIGITAL CORP                                          106040      20260814    edgar/data/106040/0000106040-26-000073.txt
8-K              H&R BLOCK INC                                                 12659       20260814    edgar/data/12659/0000012659-26-000041.txt
"""


def test_parses_the_quarterly_format():
    rows = earnings._parse_idx(QUARTERLY_IDX)
    assert ("10-Q", 1084869, "2026-05-07") in rows
    assert ("8-K", 320183, "2026-05-01") in rows


def test_parses_the_daily_format_and_normalises_its_date():
    """The regression that made the calendar stop at the last completed quarter.

    The daily index writes 20260814 where the quarterly writes 2026-05-07. A date regex
    matches neither the daily date nor nothing — it matches *inside the accession
    number*, yielding a four-digit CIK and a date like 1371-26-00. Nothing raised: the
    rows were simply wrong, and the only visible symptom was a calendar with no filing
    newer than the previous quarter.
    """
    rows = earnings._parse_idx(DAILY_IDX)
    assert ("10-K", 106040, "2026-08-14") in rows
    assert ("8-K", 12659, "2026-08-14") in rows
    assert all(len(f) == 10 and f[4] == "-" for _, _, f in rows)


def test_forms_outside_the_keep_set_are_dropped():
    rows = earnings._parse_idx(QUARTERLY_IDX)
    assert all(form in earnings.KEEP for form, _, _ in rows)
    assert not any(cik == 999999 for _, cik, _ in rows)


# ---------------------------------------------------------------------------
# period end — the mode, not the maximum
# ---------------------------------------------------------------------------
def test_period_end_ignores_the_dei_cover_page_fact():
    """`max(end)` picks the cover date, which is days before the filing, not months.

    NetApp files sixteen facts dated 2026-04-24 and one — the dei shares-outstanding
    concept — dated 2026-05-28, the day the 10-Q was prepared. Taking the maximum and
    pairing it with the filing measured cover-date-to-filing and reported an 8-day
    median reporting lag for a form that takes about a month. Every observed lag came
    out at 8: a distribution with no spread is the signature of measuring the wrong
    quantity, not of a very punctual universe.
    """
    row = {"as_of": {
        "revenue": {"end": "2026-04-24", "tag": "Revenues"},
        "net_income": {"end": "2026-04-24", "tag": "NetIncomeLoss"},
        "assets": {"end": "2026-04-24", "tag": "Assets"},
        "shares": {"end": "2026-05-28", "tag": "EntityCommonStockSharesOutstanding"},
    }}
    assert earnings._period_end(row) == date(2026, 4, 24)


def test_period_end_is_the_mode_so_one_odd_fact_cannot_carry_it():
    row = {"as_of": {
        "a": {"end": "2026-03-31", "tag": "Revenues"},
        "b": {"end": "2026-03-31", "tag": "Assets"},
        "c": {"end": "2026-06-30", "tag": "SomethingRestated"},
    }}
    assert earnings._period_end(row) == date(2026, 3, 31)


def test_period_end_is_none_without_facts():
    assert earnings._period_end({}) is None
    assert earnings._period_end({"as_of": {}}) is None


# ---------------------------------------------------------------------------
# lag measurement
# ---------------------------------------------------------------------------
def test_lag_pairs_the_period_end_with_the_filing_that_reported_it():
    """The *first* filing after the period end, not the most recent one.

    The ledger's fundamentals come from SEC's frames API, which trails the filings by up
    to two quarters. Pairing a period end with the newest filing measures that staleness
    rather than the filer's reporting lag.
    """
    anns = [(date(2026, 8, 5), "8-K"), (date(2026, 5, 6), "8-K"), (date(2026, 2, 4), "8-K")]
    assert earnings._measure_lag(date(2026, 3, 31), anns) == 36


def test_lag_is_refused_when_the_period_predates_the_index_window():
    """A November filer's report sits before the window opens, so the earliest filing we
    can see is the *next* quarter's — a quarter plus a lag, which passed the sanity band
    at 110 and 117 days and dragged the projection with it."""
    anns = [(date(2026, 5, 20), "8-K"), (date(2026, 2, 18), "8-K")]
    pe = date(2025, 10, 31)
    assert earnings._measure_lag(pe, anns) == 110          # unguarded, and wrong
    assert earnings._measure_lag(pe, anns, date(2026, 1, 1)) is None


def test_lag_outside_the_sanity_band_is_refused():
    anns = [(date(2026, 4, 2), "8-K")]
    assert earnings._measure_lag(date(2026, 3, 31), anns) is None      # 2 days
    anns = [(date(2026, 9, 30), "8-K")]
    assert earnings._measure_lag(date(2026, 3, 31), anns) is None      # 183 days


def test_the_8k_release_wins_over_the_10q_that_trails_it():
    """Companies put the numbers out on an 8-K and file the 10-Q days later. The 8-K is
    the date a desk schedules around."""
    filings = [("2026-05-08", "10-Q"), ("2026-05-01", "8-K"), ("2026-02-01", "10-Q")]
    anns = earnings._announcements(filings)
    assert anns[0] == (date(2026, 5, 1), "8-K")


def test_an_unrelated_8k_does_not_masquerade_as_the_release():
    """An 8-K a month away from the periodic filing is a governance event, not results."""
    filings = [("2026-05-08", "10-Q"), ("2026-04-02", "8-K")]
    anns = earnings._announcements(filings)
    assert anns[0] == (date(2026, 5, 8), "10-Q")


# ---------------------------------------------------------------------------
# the payload
# ---------------------------------------------------------------------------
def _row(sym, pe="2026-03-31", **kw):
    row = {"symbol": sym, "name": f"{sym} Inc", "sector": "Industrials",
           "conviction": 70, "signal": "BUY",
           "as_of": {"revenue": {"end": pe, "tag": "Revenues"}}}
    row.update(kw)
    return row


def test_build_separates_confirmed_from_estimated(monkeypatch):
    """The distinction the whole module exists to preserve."""
    today = date(2026, 8, 17)
    monkeypatch.setattr(earnings, "cik_map", lambda: {"AAA": 1, "BBB": 2})
    monkeypatch.setattr(earnings, "filing_index", lambda *a, **k: {
        # filed three days ago — a fact
        1: [("2026-08-14", "8-K"), ("2026-05-06", "8-K")],
        # last reported in May — the next one is a projection
        2: [("2026-05-06", "8-K"), ("2026-02-04", "8-K")],
    })
    got = earnings.build([_row("AAA"), _row("BBB")], today=today)
    by = {e["symbol"]: e for e in got["entries"]}
    assert by["AAA"]["state"] == "confirmed"
    assert by["AAA"]["date"] == "2026-08-14"
    assert by["AAA"]["band_days"] == 0
    assert by["BBB"]["state"] == "estimated"
    assert by["BBB"]["band_days"] > 0
    assert by["BBB"]["date"] > today.isoformat(), "a projection must be in the future"


def test_every_projection_lands_in_the_future(monkeypatch):
    today = date(2026, 8, 17)
    monkeypatch.setattr(earnings, "cik_map", lambda: {f"S{i}": i for i in range(30)})
    monkeypatch.setattr(earnings, "filing_index", lambda *a, **k: {
        i: [("2026-05-06", "8-K"), ("2026-02-04", "8-K")] for i in range(30)})
    rows = [_row(f"S{i}", pe="2025-06-30") for i in range(30)]   # deliberately stale
    got = earnings.build(rows, today=today)
    for e in got["entries"]:
        if e["state"] == "estimated":
            assert e["date"] > today.isoformat(), f"{e['symbol']} projected into the past"


def test_a_name_with_no_cik_is_reported_not_guessed(monkeypatch):
    monkeypatch.setattr(earnings, "cik_map", lambda: {})
    monkeypatch.setattr(earnings, "filing_index", lambda *a, **k: {})
    got = earnings.build([_row("ZZZ")], today=date(2026, 8, 17))
    e = got["entries"][0]
    assert e["date"] is None and e["state"] == "unknown"
    assert "no CIK" in e["basis"]
    assert got["unresolved_n"] == 1


def test_a_name_with_no_filings_is_reported_not_guessed(monkeypatch):
    monkeypatch.setattr(earnings, "cik_map", lambda: {"AAA": 1})
    monkeypatch.setattr(earnings, "filing_index", lambda *a, **k: {})
    got = earnings.build([_row("AAA")], today=date(2026, 8, 17))
    e = got["entries"][0]
    assert e["state"] == "unknown" and e["date"] is None
    assert "too little history" in e["basis"]


def test_the_book_is_flagged_so_the_calendar_can_be_scoped_to_it(monkeypatch):
    monkeypatch.setattr(earnings, "cik_map", lambda: {"AAA": 1, "BBB": 2})
    monkeypatch.setattr(earnings, "filing_index", lambda *a, **k: {
        1: [("2026-05-06", "8-K")], 2: [("2026-05-06", "8-K")]})
    got = earnings.build([_row("AAA", weight=1.4), _row("BBB")], today=date(2026, 8, 17))
    by = {e["symbol"]: e for e in got["entries"]}
    assert by["AAA"]["in_book"] and by["AAA"]["weight"] == 1.4
    assert not by["BBB"]["in_book"]
    assert got["in_book"] == 1


def test_the_basis_states_that_estimates_are_estimates(monkeypatch):
    monkeypatch.setattr(earnings, "cik_map", lambda: {"AAA": 1})
    monkeypatch.setattr(earnings, "filing_index", lambda *a, **k: {
        1: [("2026-05-06", "8-K"), ("2026-02-04", "8-K")]})
    got = earnings.build([_row("AAA")], today=date(2026, 8, 17))
    low = got["basis"].lower()
    assert "estimated" in low and "confirmed" in low
    assert "vendor" in low, "the basis must say where this does not come from"
    for banned in ("guaranteed", "accurate to", "will report on"):
        assert banned not in low


def test_unscored_names_are_not_on_the_calendar(monkeypatch):
    """Benchmarks and ETFs have no conviction and no earnings — scoring one is a
    category error and so is scheduling one."""
    monkeypatch.setattr(earnings, "cik_map", lambda: {"AAA": 1, "SPY": 9})
    monkeypatch.setattr(earnings, "filing_index", lambda *a, **k: {
        1: [("2026-05-06", "8-K")], 9: [("2026-05-06", "8-K")]})
    rows = [_row("AAA"), dict(_row("SPY"), conviction=None)]
    got = earnings.build(rows, today=date(2026, 8, 17))
    assert [e["symbol"] for e in got["entries"]] == ["AAA"]
    assert got["universe"] == 1


# ---------------------------------------------------------------------------
# the coupling that has broken this project before
# ---------------------------------------------------------------------------
def test_the_terminal_has_an_earnings_view_wired_into_the_nav_and_boot():
    html = TERMINAL.read_text(encoding="utf-8")
    assert 'data-view="earnings"' in html
    assert 'id="view-earnings"' in html
    assert "earnings.json" in html, "the view never fetches the calendar"
    assert "renderEarnings" in html, "the view is never rendered"


def test_the_terminal_distinguishes_confirmed_from_estimated():
    html = TERMINAL.read_text(encoding="utf-8")
    body = html.split("function renderEarnings(", 1)[1].split("\nfunction ", 1)[0]
    low = body.lower()
    assert "confirmed" in low and "estimated" in low
    assert "band_days" in body or "band" in low, "an estimate must show its band"


@pytest.mark.parametrize("field", ["state", "days_away", "in_book", "last_reported",
                                   "lag_days", "median_lag_days", "band_days"])
def test_every_field_the_calendar_emits_is_displayed(field):
    html = TERMINAL.read_text(encoding="utf-8")
    assert field in html, f"{field!r} is derived nightly but never displayed"


def test_the_calendar_is_committed_and_copied_into_the_artifact():
    """`ledger/performance.json` was written by the nightly, committed by nobody, and
    404'd in silence on the deployed site for weeks. The coupling is pinned, not trusted."""
    wf = WORKFLOW.read_text(encoding="utf-8")
    add = __import__("re").search(r"git add (.+?)\n\s*if git diff", wf, __import__("re").S)
    assert add and "ledger/earnings.json" in add.group(1)
    assert __import__("re").search(r"cp\s+ledger/earnings\.json\s+site/ledger/", wf)


def test_a_results_release_confirms_before_its_10q_lands(monkeypatch):
    """The case the 8-K watch exists for.

    A company reports on Monday and files the 10-Q the following week. Anchoring only on
    periodic filings left the row showing a projected date for those days — through
    exactly the window where the date had stopped being a projection.
    """
    monkeypatch.setattr(earnings, "cik_map", lambda: {"AAA": 1})
    monkeypatch.setattr(earnings, "filing_index", lambda *a, **k: {
        1: [("2026-08-14", "8-K"),                       # results, no 10-Q yet
            ("2026-05-08", "10-Q"), ("2026-05-01", "8-K")]})
    got = earnings.build([_row("AAA")], today=date(2026, 8, 17))
    e = got["entries"][0]
    assert e["state"] == "confirmed"
    assert e["date"] == "2026-08-14" and e["last_form"] == "8-K"


def test_a_mid_quarter_8k_is_not_mistaken_for_results(monkeypatch):
    """An officer change three weeks after the 10-Q must not confirm a date."""
    monkeypatch.setattr(earnings, "cik_map", lambda: {"AAA": 1})
    monkeypatch.setattr(earnings, "filing_index", lambda *a, **k: {
        1: [("2026-08-12", "8-K"),                       # 26 days after the 10-Q
            ("2026-07-17", "10-Q"), ("2026-07-10", "8-K")]})
    got = earnings.build([_row("AAA", pe="2026-06-30")], today=date(2026, 8, 17))
    e = got["entries"][0]
    assert e["state"] == "estimated", "a governance 8-K confirmed an earnings date"
    assert e["last_reported"] == "2026-07-10"
