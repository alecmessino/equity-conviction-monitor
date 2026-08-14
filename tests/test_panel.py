"""Panel hygiene: the guards that decide which bars and which names a study may read.

Every test here corresponds to a specific way a real number in this project came out
wrong. The failures were not crashes — each one produced a plausible figure that survived
review, which is why they are pinned rather than left to inspection.
"""
from __future__ import annotations

import json

from equity_monitor import panel


# ---------------------------------------------------------------------------
# bar_flags
# ---------------------------------------------------------------------------
def test_reorganisation_bar_is_flagged():
    """CHRD: 0.073592 -> 19.011288 overnight on equity cancelled in Chapter 11.

    A single event of +42,216% from this one bar exceeded half of one screen's total
    measured alpha.
    """
    d = {"close": [0.5, 0.4, 0.073592, 19.011288, 19.2],
         "dates": ["2020-11-16", "2020-11-17", "2020-11-19", "2020-11-20", "2020-11-23"]}
    ok = panel.bar_flags(d)
    assert ok[3] is False


def test_a_real_biotech_print_survives_the_move_guard():
    """MDGL-class moves (+268% on phase-3 data) are genuine and must be kept.

    A per-name filter that dropped anything with a large single session removed these —
    hindsight-conditioned survivorship in reverse, biasing results down instead of up.
    """
    d = {"close": [10.0, 36.8], "dates": ["2022-12-16", "2022-12-19"]}
    assert panel.bar_flags(d) == [True, True]


def test_a_calendar_hole_is_flagged():
    """CHK is missing 1,313 calendar days; every window spanning it was computed blind."""
    d = {"close": [10.0, 10.1, 10.2], "dates": ["2021-03-01", "2021-03-02", "2024-10-01"]}
    assert panel.bar_flags(d)[2] is False


def test_a_holiday_cluster_is_not_a_hole():
    """Thanksgiving and Christmas weeks must not read as missing data."""
    d = {"close": [10.0, 10.1], "dates": ["2023-12-22", "2023-12-27"]}
    assert panel.bar_flags(d) == [True, True]


def test_a_non_positive_close_is_flagged_rather_than_dividing_by_zero():
    d = {"close": [10.0, 0.0, 5.0], "dates": ["2020-01-02", "2020-01-03", "2020-01-06"]}
    ok = panel.bar_flags(d)
    assert ok[1] is False and ok[2] is False


# ---------------------------------------------------------------------------
# episode_flags — the recycled-ticker guard, at both ends
# ---------------------------------------------------------------------------
def test_bars_after_the_delisting_belong_to_a_different_company():
    """S is Sprint through 2020-04-01 and SentinelOne after it.

    Same symbol, unrelated issuer. The price API offers no way to tell them apart, so the
    split has to come from the listing registry.
    """
    dates = ["2020-03-30", "2020-04-01", "2021-06-30", "2021-07-01"]
    assert panel.episode_flags(dates, None, "2020-04-01") == [True, True, False, False]


def test_bars_before_the_listing_belong_to_the_predecessor():
    """Guarding only the far end deletes Moderna as though it were the company that held
    MRNA until October 2018. Both edges of the slot matter."""
    dates = ["2018-09-04", "2018-12-07", "2020-01-02"]
    assert panel.episode_flags(dates, "2018-12-07", None) == [False, True, True]


def test_an_unbounded_episode_keeps_every_bar():
    assert panel.episode_flags(["2020-01-02", "2026-08-13"], None, None) == [True, True]


def test_the_final_tape_runs_a_few_sessions_past_the_recorded_delisting():
    """Settlement prints after the endDate are the same company and are kept.

    The grace window keeps this guard from truncating a genuine last week of trading, and
    it is far narrower than the months-to-years gap before a symbol is reissued.
    """
    assert panel.episode_flags(["2020-04-01", "2020-04-06"], None, "2020-04-01") == [True, True]


def test_a_wholly_recycled_file_keeps_nothing():
    """SYT: delisted 2018-01-22, every bar on disk dated 2023 or later."""
    dates = ["2023-03-31", "2024-01-02", "2026-08-13"]
    assert not any(panel.episode_flags(dates, "2000-11-13", "2018-01-22"))


# ---------------------------------------------------------------------------
# episodes — merging listings that are one company continuing
# ---------------------------------------------------------------------------
def test_an_exchange_transfer_is_not_a_delisting():
    """PNFP closed its NASDAQ row on 2025-12-31 and opened an NYSE one two sessions later.

    Reading endDate off a single row put 2,789 of the 2016 cohort in the ground where
    2,688 belong — live companies counted as casualties.
    """
    rows = [{"ticker": "PNFP", "startDate": "2000-08-22", "endDate": "2025-12-31"},
            {"ticker": "PNFP", "startDate": "2026-01-02", "endDate": "2026-08-13"}]
    assert panel.episodes(rows)["PNFP"] == [("2000-08-22", "2026-08-13")]


def test_a_reissued_symbol_stays_two_episodes():
    """S waited 455 days between Sprint and SentinelOne. Nothing sits near the threshold."""
    rows = [{"ticker": "S", "startDate": "1984-11-08", "endDate": "2020-04-01"},
            {"ticker": "S", "startDate": "2021-06-30", "endDate": "2026-08-13"}]
    assert panel.episodes(rows)["S"] == [("1984-11-08", "2020-04-01"),
                                         ("2021-06-30", "2026-08-13")]


def test_episodes_do_not_depend_on_registry_row_order():
    """The registry is not sorted, so the merge must not depend on which row arrives first."""
    a = [{"ticker": "X", "startDate": "2021-06-30", "endDate": "2026-08-13"},
         {"ticker": "X", "startDate": "1984-11-08", "endDate": "2020-04-01"}]
    assert panel.episodes(a) == panel.episodes(list(reversed(a)))


def test_duplicate_listings_across_venues_merge():
    """COHR carries identical 1990-03-26 NYSE and NASDAQ rows for one company."""
    rows = [{"ticker": "COHR", "startDate": "1990-03-26", "endDate": "2026-08-13"},
            {"ticker": "COHR", "startDate": "1990-03-26", "endDate": "2022-07-01"}]
    assert panel.episodes(rows)["COHR"] == [("1990-03-26", "2026-08-13")]


# ---------------------------------------------------------------------------
# resolve_episode — and how far to trust it
# ---------------------------------------------------------------------------
def test_a_file_is_matched_to_the_episode_its_own_bars_fall_in():
    """Moderna's file starts after the previous holder of MRNA died, so nothing is cut."""
    eps = [("1996-01-02", "2018-10-08"), ("2018-12-07", "2026-08-13")]
    assert panel.resolve_episode(eps, ["2018-12-07", "2026-08-13"]) == \
        ("2018-12-07", "2026-08-13", "clean")


def test_a_file_spanning_a_boundary_is_truncated_not_dropped():
    """AA holds two months of Alcoa Inc. before Alcoa Corp listed on 2016-10-18."""
    eps = [("1970-01-02", "2016-10-05"), ("2016-10-18", "2026-08-13")]
    first, last, status = panel.resolve_episode(
        eps, ["2016-08-15", "2016-09-01", "2016-10-18", "2020-01-02", "2026-08-13"])
    assert status == "truncated" and first == "2016-10-18"


def test_a_file_holding_only_successor_bars_is_reported_recycled():
    """SYT's file is 2023 onward; the company that was asked for died in January 2018."""
    dates = ["2023-03-31", "2026-08-13"]
    assert panel.resolve_episode([("2000-11-13", "2018-01-22")], dates)[2] == "recycled"


def test_the_episode_holding_most_of_the_file_wins_not_the_one_it_opens_in():
    """Anchoring on the first bar keeps AA's two months of Alcoa Inc. and throws away ten
    years of Alcoa Corp. The majority of the file decides which company it is."""
    eps = [("1970-01-02", "2016-10-05"), ("2016-10-18", "2026-08-13")]
    dates = ["2016-08-15", "2016-09-01", "2016-10-18", "2020-01-02", "2026-08-13"]
    first, _, status = panel.resolve_episode(eps, dates)
    assert (first, status) == ("2016-10-18", "truncated")


def test_a_registry_artifact_leaves_the_bounds_open_rather_than_deleting_the_name():
    """BRKB resolves to a nineteen-day row against a file holding ten years of Berkshire.

    Truncating to the row would delete the name. Reporting it unresolved keeps the bars
    and keeps the disagreement visible to whatever reads the panel.
    """
    eps = [("2017-09-08", "2017-09-27")]
    assert panel.resolve_episode(eps, ["2016-08-15", "2026-08-13"]) == (None, None, "unresolved")


def test_a_symbol_the_registry_has_never_heard_of_is_unresolved():
    assert panel.resolve_episode([], ["2020-01-02"]) == (None, None, "unresolved")


# ---------------------------------------------------------------------------
# clean_windows
# ---------------------------------------------------------------------------
def test_a_window_spanning_a_bad_bar_cannot_anchor_an_event():
    ok = [True] * 10
    ok[5] = False
    out = panel.clean_windows(ok, lookback=2, hold=1)
    assert out[4] is False and out[6] is False and out[7] is False
    assert out[2] is True and out[8] is True


def test_clean_windows_clamps_at_both_ends_of_the_series():
    """Near the edges the window is short, not wrapped or out of range."""
    assert panel.clean_windows([True, True, True], lookback=50, hold=50) == [True] * 3


def test_clean_windows_matches_the_slice_it_replaces():
    """The prefix-sum form exists for speed on a thirteen-year series; it must agree."""
    ok = [True] * 40
    for i in (3, 11, 12, 29, 39):
        ok[i] = False
    lookback, hold = 5, 3
    naive = [all(ok[max(0, i - lookback):min(len(ok), i + hold + 1)]) for i in range(len(ok))]
    assert panel.clean_windows(ok, lookback, hold) == naive


# ---------------------------------------------------------------------------
# instrument classification and provenance
# ---------------------------------------------------------------------------
def test_etfs_are_not_screened_as_common_stock():
    assert panel.is_etf("spy") and panel.is_etf("RSP")
    assert not panel.is_etf("ROL")


def test_manifest_records_enough_to_reproduce_a_result(tmp_path):
    """The panel was rewritten while a study ran — 968 of 1,077 files inside one hour —
    and a control statistic moved more than a point from roughly sixteen files landing.
    """
    d = {"dates": ["2020-01-02", "2020-01-03"], "close": [1.0, 1.1]}
    (tmp_path / "AAA.json").write_text(json.dumps(d))
    m = panel.manifest(str(tmp_path), ["AAA", "MISSING"])
    assert "MISSING" not in m
    assert m["AAA"]["bars"] == 2
    assert m["AAA"]["first"] == "2020-01-02" and m["AAA"]["last"] == "2020-01-03"
    assert len(m["AAA"]["sha1"]) == 12


def test_manifest_sha_changes_when_the_file_does(tmp_path):
    p = tmp_path / "AAA.json"
    p.write_text(json.dumps({"dates": ["2020-01-02"], "close": [1.0]}))
    before = panel.manifest(str(tmp_path), ["AAA"])["AAA"]["sha1"]
    p.write_text(json.dumps({"dates": ["2020-01-02"], "close": [1.01]}))
    assert panel.manifest(str(tmp_path), ["AAA"])["AAA"]["sha1"] != before
