"""Contract tests for the data layer.

These are deliberately offline and pure. Their job is to pin the parsing and arithmetic
that sits between a vendor response and a score — the layer where the previous
implementation failed silently. A vendor changing a field name should break a test here
rather than quietly zero a column on the published board.
"""
from __future__ import annotations

import math
from datetime import date

import pytest

from equity_monitor import features, universe as uni
from equity_monitor.sources import edgar, prices


# ---------------------------------------------------------------------------
# prices
# ---------------------------------------------------------------------------
def test_nasdaq_number_parsing():
    """Nasdaq returns display-formatted strings; equities carry '$', ETFs do not."""
    assert prices._num("$311.00") == 311.0
    assert prices._num("769.79") == 769.79
    assert prices._num("49,438,760") == 49438760.0
    assert prices._num("N/A") is None
    assert prices._num("") is None
    assert prices._num(None) is None
    assert prices._num("--") is None


def synthetic_bars(closes: list[float], symbol: str = "TEST") -> prices.Bars:
    n = len(closes)
    return prices.Bars(
        symbol=symbol,
        dates=[f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)],
        open=list(closes),
        high=[c * 1.01 for c in closes],
        low=[c * 0.99 for c in closes],
        close=list(closes),
        volume=[1_000_000.0] * n,
        source="test",
    )


def test_price_features_are_derived_from_the_series():
    """Every price feature comes from the bars, so none can be silently absent.

    v2 read daily change out of a vendor 'meta' blob that did not contain the field,
    so every name's change was 0 and relative strength carried no information at all.
    """
    closes = [100.0] * 200 + [float(100 + i) for i in range(1, 101)]  # 300 bars, rising
    f = prices.features(synthetic_bars(closes))

    assert f["price"] == pytest.approx(200.0)
    assert f["chg_1d"] == pytest.approx(1 / 199, rel=1e-6)
    assert f["ret_12m"] is not None and f["ret_12m"] > 0
    assert f["hi_52w"] == pytest.approx(202.0)      # the 200.0 close printed a 202.0 high
    # Rising series: the last close is the highest close, but the last *high* is 1% above
    # it, so a monotone series sits one bar's high below its own peak rather than at 0.
    assert f["drawdown_52w"] == pytest.approx(1.0 - 1.0 / 1.01)
    assert f["ma50"] is not None and f["ma200"] is not None
    assert f["px_vs_ma50"] > 0
    assert f["adv_usd"] > 0
    assert f["bars"] == 300


def test_returns_are_none_when_the_window_is_too_short():
    f = prices.features(synthetic_bars([100.0 + i for i in range(40)]))
    assert f["ret_1m"] is not None      # 21 days available
    assert f["ret_12m"] is None         # 252 days are not
    assert f["ma200"] is None


def test_drawdown_measures_from_the_52_week_intraday_peak():
    """The 52-week band is taken from real highs and lows, not from closes.

    ``synthetic_bars`` sets high = close x 1.01, so a 100.0 close prints a 101.0 high and
    the drawdown is measured from there. This is not a cosmetic choice: the swing layer
    anchors its retracement leg to ``bars.high`` (``swing.selloff_leg``), and while this
    function read closes, ``drawdown_52w`` and ``leg_high`` were measured from different
    peaks while both fed the same reward-to-risk gate.
    """
    closes = [50.0] * 10 + [100.0] + [75.0] * 10
    f = prices.features(synthetic_bars(closes))
    assert f["hi_52w"] == pytest.approx(101.0)          # the 100.0 close printed a 101.0 high
    assert f["lo_52w"] == pytest.approx(49.5)           # and the 50.0 close a 49.5 low
    assert f["drawdown_52w"] == pytest.approx(1.0 - 75.0 / 101.0)


def test_atr_uses_real_highs_and_lows():
    """v2 approximated ATR from close-to-close moves, which understates intraday range
    and therefore places stops too tight."""
    bars = synthetic_bars([100.0] * 30)          # flat closes, but high/low span 2%
    a = prices.atr(bars)
    assert a == pytest.approx(2.0, rel=1e-6)     # 101 - 99


def test_realized_vol_is_annualised_and_zero_for_a_flat_series():
    assert prices.realized_vol([100.0] * 100) == pytest.approx(0.0)
    rising = [100.0 * (1.001 ** i) for i in range(100)]
    assert prices.realized_vol(rising) == pytest.approx(0.0, abs=1e-9)


def test_ytd_return_uses_the_prior_year_close():
    bars = prices.Bars(
        symbol="T", dates=["2025-12-30", "2025-12-31", "2026-01-02", "2026-01-03"],
        open=[0] * 4, high=[0] * 4, low=[0] * 4,
        close=[90.0, 100.0, 105.0, 110.0], volume=[0] * 4,
    )
    assert prices.ytd_return(bars, date(2026, 8, 6)) == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# edgar
# ---------------------------------------------------------------------------
def test_period_helpers_track_the_run_date():
    today = date(2026, 8, 6)
    assert edgar.annual_periods(today, 3) == ["CY2025", "CY2024", "CY2023"]
    # August 2026 sits in Q3, so the most recent filed balance sheet is Q2.
    assert edgar.instant_periods(today, 2) == ["CY2026Q2I", "CY2026Q1I"]
    assert edgar.quarter_periods(today) == ("CY2026Q2", "CY2025Q2")


def test_instant_periods_roll_across_the_year_boundary():
    assert edgar.instant_periods(date(2026, 2, 10), 2) == ["CY2025Q4I", "CY2025Q3I"]


def test_earnings_stability_rewards_consistency():
    steady = edgar.earnings_stability([100, 105, 98, 102, 101])
    erratic = edgar.earnings_stability([100, -50, 300, 20, 90])
    assert steady > 0.9
    assert erratic < 0.3
    assert 0.0 <= erratic <= 1.0


def test_earnings_stability_declines_to_answer_without_a_distribution():
    """Two observations are not a distribution; returning a number would look more
    confident than the evidence supports."""
    assert edgar.earnings_stability([100, 105]) is None
    assert edgar.earnings_stability([]) is None


def facts(**vals) -> edgar.CompanyFacts:
    c = edgar.CompanyFacts(cik=1, ticker="T")
    for k, v in vals.items():
        c.facts[k] = edgar.Fact(value=float(v), end="2026-06-30", accn="x", tag=k)
    return c


def test_derive_computes_the_ratio_set():
    c = facts(revenue=1000e6, gross_profit=400e6, operating_income=200e6,
              cfo=250e6, capex=50e6, dna=40e6, equity=500e6, cash=100e6,
              debt_lt=200e6, shares=10e6)
    c.net_income_history = [100e6, 105e6, 98e6, 103e6]
    d = edgar.derive(c, market_cap=2000e6)

    assert d["gross_margin"] == pytest.approx(0.40)
    # invested = 200 + 500 - 100 = 600; NOPAT = 200 * 0.79 = 158
    assert d["roic"] == pytest.approx(158 / 600, rel=1e-6)
    assert d["fcf_yield"] == pytest.approx(200 / 2000)          # (250 - 50) / 2000
    # net debt = 200 - 100 = 100; EBITDA = 200 + 40 = 240
    assert d["net_debt_ebitda"] == pytest.approx(100 / 240, rel=1e-6)
    assert d["shares"] == pytest.approx(10e6)


def test_gross_margin_is_reconstructed_from_cost_of_revenue():
    """Many filers report cost of revenue but never tag GrossProfit."""
    c = facts(revenue=1000e6, cost_of_revenue=650e6)
    assert edgar.derive(c, market_cap=1e9)["gross_margin"] == pytest.approx(0.35)


def test_roic_is_suppressed_when_invested_capital_is_negative():
    """Buyback-heavy companies can carry negative book equity. ROIC there has a
    confident sign and no meaning — Booking Holdings scored -92% before this guard,
    which would have ranked a highly profitable business last in its sector."""
    c = facts(revenue=1000e6, operating_income=200e6, equity=-400e6,
              cash=100e6, debt_lt=200e6)
    assert edgar.derive(c, market_cap=5000e6)["roic"] is None


def test_missing_inputs_produce_none_not_zero():
    """A zero ROIC is a claim about a company; an absent one is not."""
    d = edgar.derive(facts(revenue=1000e6), market_cap=1e9)
    for key in ("roic", "fcf_yield", "net_debt_ebitda", "earnings_stability"):
        assert d[key] is None, key


def test_share_count_falls_back_across_tags():
    assert edgar.derive(facts(shares_common=5e6), None)["shares"] == pytest.approx(5e6)
    assert edgar.derive(facts(shares_diluted=7e6), None)["shares"] == pytest.approx(7e6)
    both = edgar.derive(facts(shares=1e6, shares_common=5e6), None)
    assert both["shares"] == pytest.approx(1e6)   # dei cover-page tag wins


def test_ttm_rolls_the_year_forward_and_degrades_to_the_annual_figure():
    fy = edgar.Fact(value=1000.0, end="2025-12-31", tag="Revenues")
    now = edgar.Fact(value=300.0, end="2026-03-31", tag="Revenues")
    prior = edgar.Fact(value=250.0, end="2025-03-31", tag="Revenues")
    assert edgar._ttm(fy, now, prior).value == pytest.approx(1050.0)
    assert edgar._ttm(fy, None, prior).value == pytest.approx(1000.0)
    assert edgar._ttm(None, now, prior) is None


# ---------------------------------------------------------------------------
# universe
# ---------------------------------------------------------------------------
ISHARES_EXPORT = """iShares Russell 1000 ETF
Fund Holdings as of,"Aug 05, 2026"
Inception Date,"May 15, 2000"

Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Price,P/E
AAPL,APPLE INC,Information Technology,Equity,"1,234,567","5.21","311.00","38.3"
JPM,JPMORGAN CHASE & CO,Financials,Equity,"900,000","3.10","250.00","14.2"
XLK,TECHNOLOGY SELECT SECTOR,-,ETF,"10,000","0.04","280.00","-"
USD,USD CASH,-,Cash,"5,000","0.02","1.00","-"
"""


def test_parses_a_vendor_holdings_export_with_preamble():
    members = uni.parse_constituents(ISHARES_EXPORT)
    by = {m.symbol: m for m in members}
    assert set(by) == {"AAPL", "JPM", "XLK"}          # the cash line is dropped
    assert by["AAPL"].sector == "Information Technology"
    assert by["AAPL"].index_weight == pytest.approx(5.21)
    assert by["AAPL"].vendor_pe == pytest.approx(38.3)
    assert by["XLK"].is_etf and not by["AAPL"].is_etf


def test_sector_labels_are_normalised_across_vendor_spellings():
    assert uni.normalise_sector("Technology") == "Information Technology"
    assert uni.normalise_sector("consumer defensive") == "Consumer Staples"
    assert uni.normalise_sector("Financial Services") == "Financials"
    assert uni.normalise_sector("") == ""


def test_fallback_universe_is_usable_and_fully_classified():
    members, provenance = uni.load(limit=20)
    equities = [m for m in members if not m.is_etf]
    assert len(equities) == 20
    assert all(m.sector for m in equities), "every fallback name needs a sector"
    assert uni.BENCHMARK in {m.symbol for m in members}, "benchmark must always be present"
    assert provenance


def test_every_sector_has_a_proxy_etf():
    sectors = {s for _, _, s in uni.FALLBACK}
    for s in sectors:
        assert uni.sector_proxy(s) in {sym for sym, _ in uni.ETFS}, s


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def test_excess_return_is_volatility_normalised():
    """Otherwise a momentum ranking is a high-beta screen in disguise."""
    bench = {"ret_1m": 0.01, "ret_3m": 0.03, "ret_6m": 0.05, "ret_12m": 0.10}
    calm = {**{k: v + 0.05 for k, v in bench.items()}, "vol_1y": 0.10}
    wild = {**{k: v + 0.05 for k, v in bench.items()}, "vol_1y": 0.50}
    calm_rs, legs = features._excess(calm, bench)
    wild_rs, _ = features._excess(wild, bench)
    assert calm_rs > wild_rs           # same excess, less risk taken -> ranks higher
    assert legs["ret_12m"] == pytest.approx(0.05)


def test_excess_return_is_none_without_overlapping_horizons():
    assert features._excess({"vol_1y": 0.2}, {})[0] is None


def test_trend_separates_recovering_from_still_falling():
    above = features._trend({"px_vs_ma50": 0.10, "px_vs_ma200": 0.20})
    below = features._trend({"px_vs_ma50": -0.10, "px_vs_ma200": -0.20})
    assert above > 0.5 > below
    assert features._trend({}) is None


def test_coverage_ignores_benchmarks():
    rows = [
        {"asset_class": "Equity", "roic": 0.2},
        {"asset_class": "Equity", "roic": None},
        {"asset_class": "ETF", "roic": None},
    ]
    assert features.coverage(rows, ["roic"])["roic"] == pytest.approx(0.5)
