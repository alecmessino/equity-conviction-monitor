"""Factor-snapshot persistence and score-change attribution."""
from __future__ import annotations

import json

import pytest

from equity_monitor import model, snapshots


BASE_PERCENTILES = {
    "p_roic": 0.8, "p_fcf_yield": 0.7, "p_gross_margin": 0.6, "p_leverage": 0.6,
    "p_earnings_stability": 0.7, "p_rs": 0.6, "p_trend": 0.6,
    "p_liquidity": 0.9, "p_value": 0.5, "p_lowvol": 0.6,
}


def row(symbol="AAPL", **percentiles):
    """A row whose conviction and pillars are computed by the model itself.

    Hand-setting conviction alongside hand-set pillars produces a fixture the model
    could never emit, and an attribution test against it measures the inconsistency
    rather than the attribution.
    """
    p = dict(BASE_PERCENTILES)
    p.update(percentiles)
    scored = model.score(p)
    return {
        "symbol": symbol, "sector": "Information Technology",
        **p, **scored,
        "data_confidence": 1.0, "price": 311.0, "market_cap": 4.5e12, "weight": 2.5,
    }


def test_round_trip_preserves_every_column(tmp_path):
    rows = [row("AAPL"), row("MSFT", p_roic=0.4)]
    snapshots.write(rows, str(tmp_path), on="2026-08-07", as_of="2026-08-07T00:00:00Z")
    got = snapshots.read(snapshots.snapshot_path(str(tmp_path), "2026-08-07"))

    assert set(got["rows"]) == {"AAPL", "MSFT"}
    assert got["rows"]["AAPL"]["conviction"] == rows[0]["conviction"]
    assert got["rows"]["MSFT"]["q_raw"] == pytest.approx(rows[1]["q_raw"], abs=1e-4)
    assert got["sectors"]["AAPL"] == "Information Technology"
    for col in snapshots.COLUMNS:
        assert col in got["rows"]["AAPL"], col


def test_snapshot_records_the_spec_that_produced_it(tmp_path):
    """An IC computed across a weight change is a number about two models.

    Segmenting by spec_hash is what keeps a long series interpretable.
    """
    snapshots.write([row()], str(tmp_path), on="2026-08-07")
    got = snapshots.read(snapshots.snapshot_path(str(tmp_path), "2026-08-07"))
    assert got["model_version"] == model.MODEL_VERSION
    assert got["spec_hash"] == model.spec_hash()


def test_unscored_names_are_not_persisted(tmp_path):
    """ETFs carry no conviction and would only pad the research dataset."""
    snapshots.write([row(), {"symbol": "SPY", "conviction": None}],
                    str(tmp_path), on="2026-08-07")
    got = snapshots.read(snapshots.snapshot_path(str(tmp_path), "2026-08-07"))
    assert set(got["rows"]) == {"AAPL"}


def test_columns_are_append_only():
    """Inserting a column mid-list would silently reinterpret every prior snapshot."""
    assert snapshots.COLUMNS[:5] == [
        "conviction", "q_raw", "c_raw", "r_raw", "mr_uplift",
    ]
    assert len(set(snapshots.COLUMNS)) == len(snapshots.COLUMNS)


def test_every_percentile_is_captured():
    """A factor absent from the snapshot cannot be studied later, and the omission
    would only be discovered once the history is already unrecoverable."""
    for key in model.ALL_PERCENTILES:
        assert key in snapshots.COLUMNS, f"{key} would not be persisted"


def test_pillar_factor_map_matches_the_model_weights():
    for pillar, keys in snapshots.PILLAR_FACTORS.items():
        assert set(keys) == set(model.WEIGHTS[pillar]), pillar


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------
def test_attribution_identifies_the_factor_that_moved():
    before = row()
    after = row(p_roic=0.95)
    got = snapshots.attribute(before, after)

    assert got["total"] == after["conviction"] - before["conviction"]
    top = next(iter(got["factors"]))
    assert top == "p_roic", f"expected p_roic to dominate, got {got['factors']}"
    assert got["factors"]["p_roic"] > 0


def test_attribution_separates_a_gain_from_an_offsetting_loss():
    """The case the whole feature exists for: 'improving ROIC, deteriorating valuation'."""
    before = row()
    after = row(p_roic=0.95, p_value=0.20)
    got = snapshots.attribute(before, after)
    assert got["factors"]["p_roic"] > 0, "improving ROIC should contribute positively"
    assert got["factors"]["p_value"] < 0, "deteriorating valuation should contribute negatively"


def test_attribution_residual_is_small_for_a_normal_move():
    """The expansion is first-order; the leftover must be reported, and it must be
    small enough that the decomposition is actually informative."""
    before = row()
    after = row(p_rs=0.75)
    got = snapshots.attribute(before, after)
    assert abs(got["residual"]) < 1.5, f"residual {got['residual']} is too large to be useful"


def test_attribution_declines_when_it_cannot_be_computed():
    assert snapshots.attribute(None, row()) is None
    assert snapshots.attribute(row(), None) is None
    assert snapshots.attribute(row(), {"conviction": None}) is None


def test_attribute_all_compares_against_the_previous_snapshot(tmp_path):
    before = row()
    snapshots.write([before], str(tmp_path), on="2026-08-06")
    current = [row(p_roic=0.95, p_rs=0.9)]
    got = snapshots.attribute_all(str(tmp_path), current)
    assert got["since"] == "2026-08-06"
    assert got["names"]["AAPL"]["total"] == current[0]["conviction"] - before["conviction"]


def test_attribute_all_is_empty_on_the_very_first_run(tmp_path):
    assert snapshots.attribute_all(str(tmp_path), [row()]) == {}


# ---------------------------------------------------------------------------
# series and trends
# ---------------------------------------------------------------------------
def test_series_and_trends_accumulate_across_dates(tmp_path):
    made = []
    for i, stamp in enumerate(["2026-08-04", "2026-08-05", "2026-08-06"]):
        r = row(p_roic=0.5 + 0.2 * i)
        made.append(r["conviction"])
        snapshots.write([r], str(tmp_path), on=stamp)

    assert snapshots.available(str(tmp_path)) == ["2026-08-04", "2026-08-05", "2026-08-06"]
    assert len(set(made)) > 1, "fixture must actually vary for this to test anything"

    ser = snapshots.series(str(tmp_path), "AAPL")
    assert [s["conviction"] for s in ser] == made
    assert all(s["spec_hash"] == model.spec_hash() for s in ser)

    trends = snapshots.build_trends(str(tmp_path))
    assert trends["dates"] == ["2026-08-04", "2026-08-05", "2026-08-06"]
    assert trends["series"]["AAPL"]["conviction"] == made


def test_latest_can_exclude_today(tmp_path):
    lo, hi = row(p_roic=0.1), row(p_roic=0.99)
    snapshots.write([lo], str(tmp_path), on="2026-08-05")
    snapshots.write([hi], str(tmp_path), on="2026-08-06")
    assert snapshots.latest(str(tmp_path))["rows"]["AAPL"]["conviction"] == hi["conviction"]
    assert snapshots.latest(str(tmp_path), before="2026-08-06")["rows"]["AAPL"]["conviction"] == lo["conviction"]
    assert snapshots.latest(str(tmp_path), before="2020-01-01") is None


def test_writing_twice_for_one_date_overwrites_rather_than_duplicates(tmp_path):
    second = row(p_roic=0.2)
    snapshots.write([row()], str(tmp_path), on="2026-08-07")
    snapshots.write([second], str(tmp_path), on="2026-08-07")
    assert snapshots.available(str(tmp_path)) == ["2026-08-07"]
    got = snapshots.read(snapshots.snapshot_path(str(tmp_path), "2026-08-07"))
    assert got["rows"]["AAPL"]["conviction"] == second["conviction"]


def test_encoding_is_columnar(tmp_path):
    """Repeated key names would be most of the file at 1,000 names committed daily."""
    path = snapshots.write([row()], str(tmp_path), on="2026-08-07")
    raw = json.loads(open(path).read())
    assert isinstance(raw["data"]["AAPL"], list)
    assert len(raw["data"]["AAPL"]) == len(snapshots.COLUMNS)


def test_attribution_never_compares_a_day_against_itself(tmp_path):
    """The build overwrites today's snapshot, so an unguarded 'most recent' lookup
    compares this run to an earlier run on the same date and reports intraday rebuild
    noise as if it were a daily move."""
    morning = row(p_roic=0.30)
    snapshots.write([morning], str(tmp_path), on="2026-08-07")

    evening = [row(p_roic=0.95)]
    got = snapshots.attribute_all(str(tmp_path), evening, today="2026-08-07")
    assert got == {}, f"attributed against the same date: {got}"

    # With a genuine prior date present, it compares against that.
    snapshots.write([morning], str(tmp_path), on="2026-08-06")
    got = snapshots.attribute_all(str(tmp_path), evening, today="2026-08-07")
    assert got["since"] == "2026-08-06"


def test_exact_conviction_reconstructs_the_score_from_stored_pillars():
    """Snapshots store conviction as an integer. Differencing integers mixes the real
    move with up to a full point of rounding, which then shows up as residual and
    makes the decomposition look broken when it is not."""
    r = row(p_roic=0.73, p_rs=0.41)
    exact = snapshots.exact_conviction(r)
    assert exact is not None
    assert abs(exact - r["conviction"]) < 0.5, "must round to the stored conviction"


def test_residual_excludes_rounding_and_stays_small():
    before, after = row(), row(p_roic=0.95, p_value=0.3)
    got = snapshots.attribute(before, after)
    # The expansion should explain nearly all of the *exact* move.
    assert abs(got["residual"]) < 0.5, f"residual {got['residual']}"
    assert got["exact_total"] is not None
    # Rounding is reported separately rather than folded into the residual.
    assert abs(got["rounding"]) <= 1.0


def test_headline_always_explains_the_direction_of_the_move():
    """Top-two-by-magnitude produced labels like '+5 points, driven by -0.4 trend and
    -0.4 relative strength' — two true numbers arranged into a false summary."""
    before = row()
    after = row(p_roic=0.99, p_value=0.05, p_rs=0.55)
    got = snapshots.attribute(before, after)
    headline = got["headline"]
    assert headline, "a move must name at least one driver"
    signs = {v > 0 for _, v in headline}
    if got["exact_total"] > 0:
        assert True in signs, f"a positive move must name a positive driver: {headline}"
    if any(v < 0 for v in got["factors"].values()):
        assert False in signs, f"an offsetting drag must be named: {headline}"


def test_pillar_contributions_sum_to_the_move_exactly():
    """The log form is additive by construction, so this is an identity, not a
    tolerance. A first-order expansion left a median residual of 2.2 points on a
    2-4 point move — a decomposition that stops adding up precisely where the
    interesting moves are is not much of a decomposition."""
    import random
    rng = random.Random(11)
    for _ in range(300):
        before = row(**{k: round(rng.random(), 4) for k in BASE_PERCENTILES})
        after = row(**{k: round(rng.random(), 4) for k in BASE_PERCENTILES})
        got = snapshots.attribute(before, after)
        if got is None:
            continue
        assert abs(got["residual"]) < 1e-9, (
            f"pillars sum to {sum(p['points'] for p in got['pillars'].values())} "
            f"but the move was {got['exact_total']}")


def test_factor_contributions_sum_to_their_pillar():
    before = row()
    after = row(p_roic=0.2, p_fcf_yield=0.95, p_rs=0.9, p_value=0.1)
    got = snapshots.attribute(before, after)
    for pillar, keys in snapshots.pillar_factors_for(got["profile"]).items():
        if pillar not in got["pillars"]:
            continue
        share = sum(got["factors"].get(k, 0.0) for k in keys)
        assert abs(share - got["pillars"][pillar]["points"]) < 0.01, pillar
