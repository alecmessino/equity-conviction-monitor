"""Whether a published file that did not refresh can still pass for a current one.

Once, it could. The earnings step raised a TypeError its own `except` swallowed, the run
reported success, and `earnings.json` kept the previous night's dates while every other
artifact moved on. Nothing on the page or in the log said so; the only tell was that one
filename was missing from the refresh commit. Every test here is a way that failure, or
one shaped like it, gets caught now.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from equity_monitor import earnings, freshness

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = ROOT / "web" / "terminal.html"
NIGHTLY = ROOT / "equity_monitor" / "nightly.py"
LEDGER = ROOT / "ledger"

BOARD = "2026-08-21T23:17:56Z"


def _healthy(tmp_path: Path, **overrides) -> Path:
    """A ledger directory where every audited artifact refreshed with the board."""
    d = tmp_path / "ledger"
    d.mkdir(exist_ok=True)
    for name, key, _ in freshness.ARTIFACTS:
        (d / name).write_text(json.dumps(overrides.get(name, {key: BOARD})))
    return d


# ---------------------------------------------------------------------------
# stamps
# ---------------------------------------------------------------------------
def test_a_full_timestamp_and_a_date_are_both_readable():
    assert freshness.read_stamp({"as_of": BOARD}, "as_of") == datetime(
        2026, 8, 21, 23, 17, 56, tzinfo=timezone.utc)
    got = freshness.read_stamp({"as_of": "2026-08-21"}, "as_of")
    assert got.date() == datetime(2026, 8, 21).date()


def test_a_date_only_stamp_anchors_to_the_end_of_its_day():
    """The calendar is built during the 23:00 run and publishes a date, not a time.
    Anchoring '2026-08-21' to midnight would invent 23 hours of age on a file that
    refreshed forty minutes ago, and a lag threshold would then fire every night."""
    day = freshness.read_stamp({"as_of": "2026-08-21"}, "as_of")
    board = freshness.read_stamp({"as_of": BOARD}, "as_of")
    assert abs((board - day).total_seconds()) < 3600


def test_an_unparseable_or_absent_stamp_is_none_rather_than_a_guess():
    assert freshness.read_stamp({}, "as_of") is None
    assert freshness.read_stamp({"as_of": ""}, "as_of") is None
    assert freshness.read_stamp({"as_of": "last Tuesday"}, "as_of") is None
    assert freshness.read_stamp({"as_of": 20260821}, "as_of") is None


def test_stamping_records_the_run_and_clears_a_previous_failure(tmp_path):
    p = tmp_path / "watchlist.json"
    p.write_text(json.dumps({"counts": {"upgrades": 3}, "stale": True,
                             "stale_reason": "TypeError: boom",
                             "stale_since": "2026-08-20T23:00:00Z"}))
    assert freshness.stamp(str(p), BOARD)
    got = json.loads(p.read_text())
    assert got["as_of"] == BOARD
    assert "stale" not in got, "a file that rebuilt tonight must stop warning by itself"
    assert "stale_reason" not in got
    assert got["counts"] == {"upgrades": 3}, "the payload is annotated, never replaced"


def test_stamping_an_absent_file_is_not_an_error(tmp_path):
    assert freshness.stamp(str(tmp_path / "nope.json"), BOARD) is False


# ---------------------------------------------------------------------------
# stale marking
# ---------------------------------------------------------------------------
def test_marking_stale_preserves_the_payload_and_the_date_it_was_built(tmp_path):
    p = tmp_path / "earnings.json"
    p.write_text(json.dumps({"as_of": "2026-08-14", "confirmed": 3, "entries": [1, 2]}))
    assert freshness.mark_stale(str(p), reason="TypeError: boom", as_of=BOARD)
    got = json.loads(p.read_text())
    assert got["stale"] is True
    assert "TypeError" in got["stale_reason"]
    assert got["stale_since"] == BOARD
    assert got["built_as_of"] == "2026-08-14"
    assert got["entries"] == [1, 2], "deleting the file reads as 'nothing to report'"


def test_marking_stale_twice_keeps_the_date_it_was_actually_built(tmp_path):
    p = tmp_path / "earnings.json"
    p.write_text(json.dumps({"as_of": "2026-08-14"}))
    freshness.mark_stale(str(p), reason="first", as_of="2026-08-20T23:00:00Z")
    freshness.mark_stale(str(p), reason="second", as_of=BOARD)
    got = json.loads(p.read_text())
    assert got["built_as_of"] == "2026-08-14", \
        "a second failure must not re-date the file to its own stale copy"
    assert got["stale_reason"] == "second"


def test_marking_an_absent_or_unreadable_file_never_raises(tmp_path):
    assert freshness.mark_stale(str(tmp_path / "nope.json"), reason="x", as_of=BOARD) is False
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert freshness.mark_stale(str(bad), reason="x", as_of=BOARD) is False


def test_the_calendars_own_name_for_the_operation_still_works(tmp_path):
    """nightly monkeypatches `earnings.mark_stale` by name in the test suite, and the
    terminal reads the keys it writes. Moving the behaviour must not move the name."""
    p = tmp_path / "earnings.json"
    p.write_text(json.dumps({"as_of": "2026-08-14"}))
    assert earnings.mark_stale(str(p), reason="TypeError: boom", as_of=BOARD)
    assert json.loads(p.read_text())["stale"] is True


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------
def test_a_healthy_ledger_reports_ok(tmp_path):
    d = _healthy(tmp_path)
    rep = freshness.audit(str(d))
    assert rep["ok"] is True
    assert rep["stale"] == [] and rep["lagging"] == [] and rep["missing"] == []
    assert rep["max_lag_hours"] == pytest.approx(0.0, abs=0.01)


def test_a_file_left_behind_by_a_failed_step_is_caught_by_its_age(tmp_path):
    """The exact 2026-08-17 failure: the board moved, the calendar did not, and nothing
    said so. Age against the board catches it even with no stale flag written."""
    d = _healthy(tmp_path, **{"earnings.json": {"as_of": "2026-08-17"}})

    rep = freshness.audit(str(d))
    assert rep["ok"] is False
    assert "earnings.json" in rep["lagging"]
    assert rep["max_lag_hours"] > freshness.LAG_WARN_HOURS
    assert rep["worst"] == "earnings.json"


def test_a_flagged_file_is_reported_with_its_reason(tmp_path):
    d = _healthy(tmp_path)
    freshness.mark_stale(str(d / "monitor.json"),
                         reason="TypeError: progress() missing 1 required positional argument",
                         as_of=BOARD)
    rep = freshness.audit(str(d))
    assert rep["ok"] is False
    assert "monitor.json" in rep["stale"]
    row = next(f for f in rep["files"] if f["file"] == "monitor.json")
    assert "TypeError" in row["stale_reason"]


def test_a_missing_artifact_is_reported_as_missing_not_as_current(tmp_path):
    d = tmp_path / "ledger"
    d.mkdir()
    for name, key, _ in freshness.ARTIFACTS:
        if name == "edge.json":
            continue
        (d / name).write_text(json.dumps({key: BOARD}))
    rep = freshness.audit(str(d))
    assert "edge.json" in rep["missing"]
    assert rep["ok"] is False


def test_an_unstamped_artifact_is_reported_rather_than_assumed_fresh(tmp_path):
    d = _healthy(tmp_path, **{"trends.json": {"dates": [], "series": {}}})
    rep = freshness.audit(str(d))
    assert "trends.json" in rep["unstamped"]
    row = next(f for f in rep["files"] if f["file"] == "trends.json")
    assert row["lag_hours"] is None, "no stamp means unknown age, not zero age"


def test_a_file_written_after_the_board_is_not_called_lagging(tmp_path):
    """monitor, health and the rest are written minutes AFTER index.json in the same
    run, so their stamps are slightly newer. A one-sided comparison would flag every
    healthy night."""
    d = tmp_path / "ledger"
    d.mkdir()
    for name, key, _ in freshness.ARTIFACTS:
        (d / name).write_text(json.dumps({key: "2026-08-21T23:19:00Z"}))
    (d / "index.json").write_text(json.dumps({"as_of": BOARD}))
    rep = freshness.audit(str(d))
    assert rep["ok"] is True
    assert rep["lagging"] == []


def test_the_audit_is_recorded_inside_the_board_the_terminal_already_has(tmp_path):
    d = _healthy(tmp_path, **{"index.json": {"as_of": BOARD, "all": []}})
    freshness.write_audit(str(d), BOARD)
    payload = json.loads((d / "index.json").read_text())
    assert payload["data_health"]["ok"] is True
    assert payload["all"] == [], "the board must survive being annotated"


def test_the_audit_explains_why_age_is_measured_against_the_board(tmp_path):
    d = tmp_path / "ledger"
    d.mkdir()
    (d / "index.json").write_text(json.dumps({"as_of": BOARD}))
    basis = freshness.audit(str(d))["basis"]
    assert "weekday" in basis.lower()
    assert "symbol" in basis, "the files that cannot be audited must be named"


def test_symbol_keyed_files_are_deliberately_not_audited():
    """history.json and macro.json are keyed by symbol/series with nowhere to put a
    stamp. Listing them would report 'unstamped' every night forever, which trains the
    reader to ignore the panel."""
    names = {n for n, _, _ in freshness.ARTIFACTS}
    assert "history.json" not in names
    assert "macro.json" not in names


# ---------------------------------------------------------------------------
# the seam: nightly, the shipped ledger, and the terminal
# ---------------------------------------------------------------------------
def test_every_audited_artifact_is_actually_written_by_the_nightly():
    """An artifact in the table that the nightly never writes would report as missing
    every single run — a permanent false alarm."""
    src = NIGHTLY.read_text(encoding="utf-8")
    for name, _, _ in freshness.ARTIFACTS:
        assert name in src, f"{name} is audited but the nightly never mentions it"


def test_health_json_carries_a_timestamp_now():
    """It carried none at all, which is why nothing could tell a health report written
    tonight from one a failed step left behind three nights ago."""
    from equity_monitor import health
    built = health.build(str(LEDGER))
    assert "as_of" in built
    assert freshness.read_stamp(built, "as_of") is not None, \
        "the stamp has to be in the format the audit reads, not merely present"


@pytest.mark.skipif(not (LEDGER / "index.json").exists(), reason="no ledger committed")
def test_the_shipped_ledger_reports_its_own_health():
    payload = json.loads((LEDGER / "index.json").read_text())
    dh = payload.get("data_health")
    assert dh, "the board must carry the audit the terminal reads"
    assert {f["file"] for f in dh["files"]} == {n for n, _, _ in freshness.ARTIFACTS}
    assert dh["ok"] is True, f"the committed ledger is not internally current: {dh}"


def test_the_terminal_warns_for_any_artifact_not_only_the_calendar():
    html = TERMINAL.read_text(encoding="utf-8")
    assert "data_health" in html
    body = html.split("function renderDataHealth(", 1)[1].split("\nfunction ", 1)[0]
    assert "stale" in body and "lagging" in body
    assert "data-health-banner" in body

    shared = html.split("function staleNote(", 1)[1].split("\nfunction ", 1)[0]
    assert "stale_reason" in shared, "the reason must reach the reader, not just the log"

    ern = html.split("function renderEarnings(", 1)[1].split("\nfunction ", 1)[0]
    assert "staleNote(" in ern, "the calendar should use the shared banner, not its own"


def test_the_nightly_flags_a_failed_step_rather_than_only_printing_it():
    src = NIGHTLY.read_text(encoding="utf-8")
    assert "def _flag(" in src
    assert src.count("_flag(") >= 5, \
        "every artifact written inside a try/except needs its file flagged on failure"
    assert "freshness.write_audit" in src
