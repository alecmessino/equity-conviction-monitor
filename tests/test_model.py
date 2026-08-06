"""Model v3 behaviour — including regression tests for each way v2 failed.

Every test below named ``test_regression_*`` encodes a specific defect that shipped
to production and sat there undetected. They are the point of this file.
"""
from __future__ import annotations

import math

import pytest

from equity_monitor import model


def uniform(p: float, **overrides) -> dict:
    """A row whose every percentile is `p`, so pillar arithmetic is easy to reason about."""
    row = {k: p for k in model.ALL_PERCENTILES}
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def test_median_name_lands_in_the_hold_band():
    """A perfectly median company should score around the middle of the scale.

    Under v2 this same input produced 100*0.5*0.5*0.5 = 12, i.e. AVOID. The published
    tier thresholds described a structure the arithmetic could not reach.
    """
    out = model.score(uniform(0.5))
    assert 50 <= out["conviction"] <= 65
    assert out["signal"] == "HOLD"


def test_score_is_monotonic_in_every_pillar():
    base = model.score(uniform(0.5))["conviction"]
    for key in model.ALL_PERCENTILES:
        better = model.score(uniform(0.5, **{key: 1.0}))["conviction"]
        worse = model.score(uniform(0.5, **{key: 0.0}))["conviction"]
        assert worse <= base <= better, f"{key} is not monotonic"


def test_full_range_is_reachable():
    assert model.score(uniform(1.0))["conviction"] >= 95
    assert model.score(uniform(0.0))["conviction"] <= 30
    assert model.score(uniform(1.0))["signal"] == "STRONG"
    assert model.score(uniform(0.0))["signal"] == "AVOID"


def test_every_signal_tier_is_reachable():
    """v2 could only ever emit AVOID; a tier no input can produce is not a tier."""
    seen = {model.score(uniform(p / 100))["signal"] for p in range(0, 101)}
    assert seen == {"STRONG", "BUY", "HOLD", "WATCH", "AVOID"}


def test_geometric_mean_preserves_the_veto():
    """A collapsed pillar must still dominate — that property is why it's multiplicative."""
    strong_but_illiquid = uniform(0.95, p_liquidity=0.0, p_value=0.0, p_lowvol=0.0)
    assert model.score(strong_but_illiquid)["conviction"] < model.score(uniform(0.95))["conviction"] - 20


# ---------------------------------------------------------------------------
# quality mean-reversion — the strategy the repository is named after
# ---------------------------------------------------------------------------
def test_drawdown_lifts_high_quality_names():
    flat = uniform(0.8, drawdown_52w=0.0)
    beaten = uniform(0.8, drawdown_52w=0.40)
    assert model.score(beaten)["conviction"] > model.score(flat)["conviction"]
    assert model.score(beaten)["mr_uplift"] > 1.0


def test_regression_drawdown_does_not_lift_low_quality_names():
    """v2 applied its mean-reversion kicker regardless of quality.

    That inverts the stated thesis: it rewards falling knives exactly as much as it
    rewards quality businesses on sale.
    """
    junk_flat = uniform(0.2, drawdown_52w=0.0)
    junk_beaten = uniform(0.2, drawdown_52w=0.40)
    assert model.score(junk_beaten)["mr_uplift"] == 1.0
    assert model.score(junk_beaten)["conviction"] == model.score(junk_flat)["conviction"]


# ---------------------------------------------------------------------------
# missing data
# ---------------------------------------------------------------------------
def test_regression_missing_fundamentals_do_not_force_avoid():
    """v2's quality floor `(raw - 0.20)/0.60` mapped any name with neutral inputs to
    Q = 0, hence conviction 0, hence a confident AVOID on 48 of 48 names. Absent
    evidence must not masquerade as negative evidence."""
    rows = [
        {"symbol": "KNOWN", "sector": "Tech", "asset_class": "Equity", "roic": 0.2,
         "fcf_yield": 0.05, "gross_margin": 0.5, "net_debt_ebitda": 1.0,
         "earnings_stability": 0.8, "rs_blend": 0.1, "trend": 0.6,
         "adv_usd": 1e9, "vol_1y": 0.25, "earnings_yield": 0.04},
        {"symbol": "BLANK", "sector": "Tech", "asset_class": "Equity"},
    ]
    model.score_rows(rows)
    blank = next(r for r in rows if r["symbol"] == "BLANK")
    assert blank["conviction"] > 0
    assert blank["signal"] != "AVOID"
    assert blank["data_confidence"] < 0.2


def test_data_confidence_tracks_observed_inputs():
    rows = [
        {"symbol": f"S{i}", "sector": "Tech", "asset_class": "Equity",
         "roic": 0.1 * i, "fcf_yield": 0.01 * i, "gross_margin": 0.3,
         "net_debt_ebitda": 1.0, "earnings_stability": 0.5, "rs_blend": 0.1 * i,
         "trend": 0.5, "adv_usd": 1e8 * (i + 1), "vol_1y": 0.3,
         "earnings_yield": 0.03}
        for i in range(6)
    ]
    rows.append({"symbol": "SPARSE", "sector": "Tech", "asset_class": "Equity",
                 "adv_usd": 1e8, "vol_1y": 0.3})
    model.score_rows(rows)
    assert rows[0]["data_confidence"] == 1.0
    assert rows[-1]["data_confidence"] < 0.5
    assert "p_roic" in rows[-1]["imputed"]


# ---------------------------------------------------------------------------
# cross-sectional preparation
# ---------------------------------------------------------------------------
def test_percentile_rank_spans_the_unit_interval():
    assert model.percentile_rank([1, 2, 3, 4, 5]) == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_percentile_rank_averages_ties_and_passes_none_through():
    assert model.percentile_rank([5, 5, 5]) == [0.5, 0.5, 0.5]
    assert model.percentile_rank([1, None, 3]) == [0.0, None, 1.0]
    assert model.percentile_rank([]) == []
    assert model.percentile_rank([7]) == [0.5]


def test_leverage_and_volatility_rank_inverted():
    """Lower net debt and lower volatility must earn the *higher* percentile."""
    rows = [
        {"symbol": s, "sector": "Tech", "asset_class": "Equity",
         "net_debt_ebitda": nd, "vol_1y": v, "roic": 0.1, "fcf_yield": 0.01,
         "gross_margin": 0.3, "earnings_stability": 0.5, "rs_blend": 0.0,
         "trend": 0.5, "adv_usd": 1e9, "earnings_yield": 0.03}
        for s, nd, v in [("LOW", 0.2, 0.15), ("MID", 2.0, 0.30), ("HIGH", 6.0, 0.60)]
    ]
    model.prepare(rows)
    by = {r["symbol"]: r for r in rows}
    assert by["LOW"]["p_leverage"] > by["MID"]["p_leverage"] > by["HIGH"]["p_leverage"]
    assert by["LOW"]["p_lowvol"] > by["HIGH"]["p_lowvol"]


def test_regression_cheap_names_score_better_than_expensive_ones():
    """v2's `clamp01((pe - 20) / 20)` was one-sided: it could penalise an expensive
    name but never reward a cheap one, so the terminal's own CHEAP tier was
    mathematically unreachable and val_zscore was 0 for every name in production."""
    rows = [
        {"symbol": s, "sector": "Tech", "asset_class": "Equity", "earnings_yield": ey,
         "roic": 0.15, "fcf_yield": 0.03, "gross_margin": 0.4, "net_debt_ebitda": 1.0,
         "earnings_stability": 0.7, "rs_blend": 0.0, "trend": 0.5,
         "adv_usd": 1e9, "vol_1y": 0.25}
        for s, ey in [("CHEAP", 0.10), ("FAIR", 0.05), ("RICH", 0.01)]
    ]
    model.score_rows(rows)
    by = {r["symbol"]: r for r in rows}
    assert by["CHEAP"]["p_value"] > by["FAIR"]["p_value"] > by["RICH"]["p_value"]
    assert by["CHEAP"]["conviction"] > by["RICH"]["conviction"]


def test_etfs_are_not_scored():
    """An index fund has no ROIC. Scoring one is a category error, not a data gap."""
    rows = [
        {"symbol": "SPY", "asset_class": "ETF", "sector": ""},
        {"symbol": "AAPL", "asset_class": "Equity", "sector": "Tech", "roic": 0.3,
         "fcf_yield": 0.05, "gross_margin": 0.45, "net_debt_ebitda": 0.5,
         "earnings_stability": 0.9, "rs_blend": 0.2, "trend": 0.8,
         "adv_usd": 5e9, "vol_1y": 0.22, "earnings_yield": 0.03},
    ]
    model.score_rows(rows)
    assert rows[0]["conviction"] is None
    assert rows[0]["signal"] == "BENCHMARK"
    assert rows[1]["conviction"] is not None


def test_sector_relative_ranking_does_not_penalise_low_margin_industries():
    """A grocer's 25% gross margin is excellent for a grocer. Ranking it against
    software would turn the quality score into an unintentional sector bet."""
    def row(sym, sector, gm):
        return {"symbol": sym, "sector": sector, "asset_class": "Equity",
                "gross_margin": gm, "roic": 0.15, "fcf_yield": 0.03,
                "net_debt_ebitda": 1.0, "earnings_stability": 0.7, "rs_blend": 0.0,
                "trend": 0.5, "adv_usd": 1e9, "vol_1y": 0.25, "earnings_yield": 0.04}

    rows = ([row(f"SW{i}", "Information Technology", 0.60 + i * 0.02) for i in range(6)]
            + [row(f"GR{i}", "Consumer Staples", 0.20 + i * 0.02) for i in range(6)])
    model.prepare(rows)
    best_grocer = max((r for r in rows if r["sector"] == "Consumer Staples"),
                      key=lambda r: r["p_gross_margin"])
    worst_software = min((r for r in rows if r["sector"] == "Information Technology"),
                         key=lambda r: r["p_gross_margin"])
    assert best_grocer["p_gross_margin"] > worst_software["p_gross_margin"]


def test_dispersion_flags_a_degenerate_universe():
    flat = [dict(uniform(0.5), symbol=f"S{i}", asset_class="Equity", sector="Tech")
            for i in range(20)]
    for r in flat:
        r.update(model.score(r))
    assert model.dispersion(flat) < 1.0

    varied = [dict(uniform(i / 19), symbol=f"S{i}", asset_class="Equity", sector="Tech")
              for i in range(20)]
    for r in varied:
        r.update(model.score(r))
    assert model.dispersion(varied) > 5.0


def test_score_is_pure():
    """The browser runs this same function; it must not depend on hidden state."""
    row = uniform(0.62, drawdown_52w=0.2)
    first = model.score(dict(row))
    for _ in range(5):
        assert model.score(dict(row)) == first


def test_weights_sum_to_one_per_pillar():
    for pillar, weights in model.WEIGHTS.items():
        assert math.isclose(sum(weights.values()), 1.0), pillar
