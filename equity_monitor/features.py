"""Feature assembly: universe + prices + filings -> one raw feature dict per name.

This module produces *raw, uncomparable* quantities (a 31% ROIC, a 0.8 net-debt/EBITDA,
a +14% 6-month excess return). Turning those into comparable, rankable scores is
``model.prepare``'s job. Keeping the two apart is what lets the scoring function stay
pure and therefore portable to the browser.

Nothing here substitutes a default for a missing measurement. A field we could not
observe is ``None``, and it stays ``None`` until the model decides — explicitly, and
while recording the decision — what to do about it.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from . import universe as uni
from .sources import edgar, prices

# Relative-strength horizons and their weights. Longer horizons dominate: 12-month
# momentum is the academically durable effect, while 1-month is largely reversal noise
# and is included mainly to flag names that have just broken down.
RS_HORIZONS: list[tuple[str, float]] = [
    ("ret_1m", 0.15),
    ("ret_3m", 0.25),
    ("ret_6m", 0.30),
    ("ret_12m", 0.30),
]


def fetch_bars(symbols: list[str], cache_dir: str = "", workers: int = 24,
               on_progress=None, prefer_cache: bool = False) -> dict[str, prices.Bars]:
    """Daily bars for every symbol, fetched concurrently.

    The per-host throttle in ``sources._http`` caps the actual request rate, so the
    pool buys latency overlap without hitting the endpoint any harder. At Russell
    1000 scale this is the difference between a 30-minute and a 10-minute build:
    the work is latency-bound (each request costs seconds), not rate-bound.
    """
    etf_symbols = {s for s, _ in uni.ETFS}
    out: dict[str, prices.Bars] = {}

    def one(sym: str):
        ac = "etf" if sym in etf_symbols else "stocks"
        return sym, prices.load(sym, asset_class=ac, cache_dir=cache_dir,
                                prefer_cache=prefer_cache)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (sym, bars) in enumerate(pool.map(one, symbols), 1):
            if bars:
                out[sym] = bars
            if on_progress:
                on_progress(i, len(symbols), sym, bars is not None)
    return out


def _excess(target: dict, bench: dict) -> tuple[float | None, dict]:
    """Vol-normalised multi-horizon excess return vs a benchmark.

    Dividing by realised volatility is what makes a utility's +6% comparable to a
    semiconductor's +18%: it asks how many units of risk the outperformance took,
    not how large it was. Without it, momentum rankings are a high-beta screen
    wearing a disguise.
    """
    vol = target.get("vol_1y") or target.get("vol_3m")
    legs: dict[str, float] = {}
    total = 0.0
    weight_seen = 0.0
    for key, weight in RS_HORIZONS:
        t, b = target.get(key), bench.get(key)
        if t is None or b is None:
            continue
        legs[key] = t - b
        total += weight * (t - b)
        weight_seen += weight
    if weight_seen == 0:
        return None, legs
    blended = total / weight_seen
    if vol and vol > 0.02:
        blended = blended / vol
    return blended, legs


def _trend(f: dict) -> float | None:
    """Trend state in [0, 1]: how convincingly price sits above its moving averages.

    A name above both averages with the fast one above the slow scores near 1.
    Momentum that has already rolled over scores near 0, which is the distinction
    between "cheap and recovering" and "cheap and still falling".
    """
    above50, above200 = f.get("px_vs_ma50"), f.get("px_vs_ma200")
    if above50 is None and above200 is None:
        return None
    score, parts = 0.0, 0
    for v, weight in ((above50, 0.4), (above200, 0.6)):
        if v is None:
            continue
        # +/-20% from the average saturates the contribution.
        score += weight * max(0.0, min(1.0, 0.5 + v / 0.40))
        parts += weight
    return score / parts if parts else None


def build(members: list[uni.Member], bars: dict[str, prices.Bars],
          today: date | None = None, fundamentals: dict | None = None) -> list[dict]:
    """One raw feature dict per member that has usable price data."""
    today = today or date.today()

    price_features: dict[str, dict] = {}
    for sym, b in bars.items():
        f = prices.features(b)
        f["ret_ytd"] = prices.ytd_return(b, today)
        price_features[sym] = f

    bench = price_features.get(uni.BENCHMARK)
    if bench is None:
        raise RuntimeError(
            f"benchmark {uni.BENCHMARK} has no price history — relative strength is "
            "undefined without it, and a momentum score computed against nothing is "
            "exactly the failure mode this rebuild exists to remove"
        )

    equities = [m for m in members if not m.is_etf and m.symbol in price_features]
    facts = fundamentals if fundamentals is not None else edgar.load(
        [m.symbol for m in equities], today
    )

    rows: list[dict] = []
    for m in members:
        pf = price_features.get(m.symbol)
        if not pf:
            continue
        row: dict = {
            "symbol": m.symbol,
            "name": m.name,
            "sector": m.sector,
            "asset_class": m.asset_class,
            "index_weight": m.index_weight,
            **pf,
        }

        rs_blend, rs_legs = _excess(pf, bench)
        row["rs_blend"] = rs_blend
        row["rs_legs"] = {k: round(v, 4) for k, v in rs_legs.items()}
        proxy = price_features.get(uni.sector_proxy(m.sector))
        row["rs_sector"] = _excess(pf, proxy)[0] if proxy else None
        row["trend"] = _trend(pf)

        if m.is_etf:
            # ETFs carry price features for benchmarking but are never scored.
            row.update(market_cap=None, roic=None, fcf_yield=None, gross_margin=None,
                       net_debt_ebitda=None, earnings_stability=None,
                       earnings_yield=None, ebitda_yield=None, pe=None, as_of={})
            rows.append(row)
            continue

        comp = facts.get(m.symbol)
        derived = edgar.derive(comp, None) if comp else {}
        shares = derived.get("shares")
        market_cap = shares * pf["price"] if (shares and pf["price"]) else None

        # Sanity-check the share count against revenue. Dual-class filers report share
        # counts as dimensional facts, and the bulk frames API can return a single
        # class or a stub — Dillard's resolved to 15,655 shares against $6.5bn of
        # revenue, implying a $9.7m market cap for a Russell 1000 constituent. A
        # company trading below 2% of sales is effectively in liquidation and would
        # not be in this index, so treat the figure as unknown rather than publish it.
        revenue = derived.get("revenue_ttm")
        if market_cap and revenue and revenue > 0 and market_cap < 0.02 * revenue:
            market_cap = None

        if comp and market_cap:
            # FCF yield needs market cap, which needs price — so re-derive once we have it.
            derived = edgar.derive(comp, market_cap)

        net_income = derived.get("net_income_ttm")
        ebitda = derived.get("ebitda_ttm")
        enterprise_value = None
        if market_cap is not None:
            net_debt_ebitda = derived.get("net_debt_ebitda")
            if net_debt_ebitda is not None and ebitda:
                enterprise_value = market_cap + net_debt_ebitda * ebitda

        row.update(
            market_cap=market_cap,
            roic=derived.get("roic"),
            fcf_yield=derived.get("fcf_yield"),
            roe=derived.get("roe"),
            efficiency_ratio=derived.get("efficiency_ratio"),
            equity_to_assets=derived.get("equity_to_assets"),
            ffo=derived.get("ffo"),
            ffo_yield=derived.get("ffo_yield"),
            debt_to_assets=derived.get("debt_to_assets"),
            cfo_yield=derived.get("cfo_yield"),
            gross_margin=derived.get("gross_margin"),
            net_debt_ebitda=derived.get("net_debt_ebitda"),
            earnings_stability=derived.get("earnings_stability"),
            revenue_ttm=derived.get("revenue_ttm"),
            net_income_ttm=net_income,
            fcf_ttm=derived.get("fcf_ttm"),
            ebitda_ttm=ebitda,
            # Earnings yield rather than P/E: it stays finite and correctly ordered
            # through zero and negative earnings, where P/E flips sign and becomes
            # actively misleading in a ranking.
            earnings_yield=(net_income / market_cap) if (net_income and market_cap) else None,
            ebitda_yield=(ebitda / enterprise_value) if (ebitda and enterprise_value) else None,
            pe=(market_cap / net_income) if (net_income and net_income > 0 and market_cap) else None,
            vendor_pe=m.vendor_pe,
            as_of=derived.get("as_of", {}),
        )
        rows.append(row)
    return rows


def coverage(rows: list[dict], fields: list[str]) -> dict[str, float]:
    """Fraction of scoreable rows carrying a real value for each field.

    Surfaced in the run log and the terminal. The whole reason the previous
    dashboard could ship broken for weeks is that nothing measured this.
    """
    scoreable = [r for r in rows if r.get("asset_class") != "ETF"]
    n = len(scoreable) or 1
    out = {}
    for f in fields:
        present = sum(1 for r in scoreable
                      if r.get(f) is not None and not (
                          isinstance(r.get(f), float) and math.isnan(r[f])))
        out[f] = round(present / n, 4)
    return out
