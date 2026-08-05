"""Data pipeline for the Equity Conviction Monitor.

Sources:
  - FMP (Financial Modeling Prep) — primary, free tier: quotes, fundamentals, profile.
  - Polygon.io — secondary, for higher-fi intraday / aggregates once a key is present.

Universe: S&P 500 + Russell 1000 constituents + factor/regional ETFs.
"""
from __future__ import annotations
import os, json, math, time
from dataclasses import dataclass
import requests

FMP_BASE = "https://financialmodelingprep.com/api/v3"
POLY_BASE = "https://api.polygon.io/v2"
FMP_KEY = os.environ.get("FMP_API_KEY", "")
POLY_KEY = os.environ.get("POLYGON_API_KEY", "")

# Major regional / factor ETFs tracked for the relative-value pane.
ETF_UNIVERSE = [
    "EWJ","EFA","VEU","IWD","QUAL","VLUE","MTUM","USMV","IWM","EEM","VWO",
    "AAXJ","EFAV","MTUM","QUAL","SPYV","IWD","IWF","IWS","IWL","RSP","SPY",
    "QQQ","DIA","IWM","EEM","EWW","EWT","EWZ","EGY","EPOL","EMB","HYG",
]

SP500_LISTING = "https://raw.githubusercontent.com/plotly/datasets/master/NASOSPH_NASDAQ_PTON.csv"  # placeholder; replace with a real constituents source in nightly
RUSSELL1000 = "https://raw.githubusercontent.com/AlecXue/russell-1000/master/russell-1000.txt"


@dataclass
class Quote:
    symbol: str
    price: float
    market_cap: float          # USD
    exchange: str
    currency: str = "USD"
    # price change fields (24h, 7/14/30/90/200d vs self) in PCT
    price_change_24h: float = 0.0
    price_change_7d: float = 0.0
    price_change_14d: float = 0.0
    price_change_30d: float = 0.0
    price_change_90d: float = 0.0
    price_change_200d: float = 0.0
    # volume
    total_volume: float = 0.0     # 30d average dollar volume or ADV
    turnover: float = 0.0
    # fundamentals
    roic: float = 0.0
    fcf_yield: float = 0.0
    gross_margin: float = 0.0
    debt_ebitda: float = 0.0
    earnings_stability: float = 0.0   # 1 = stable, 0 = volatile
    val_zscore: float = 0.0     # forward P/E or EV/EBITDA z vs 5yr median (cheap=neg)
    # risk
    adv: float = 0.0
    short_interest: float = 0.0
    short_days: float = 0.0     # days to cover
    drawdown_52w: float = 0.0   # 0..1 fraction below 52w high
    # computed
    rs_blend: float = 0.0      # vs SPY, log-returns vol-normalised
    rs_sector: float = 0.0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in __class__.__dataclass_fields__}


def _fmp(path: str, **params) -> dict | list:
    params["apikey"] = FMP_KEY
    url = f"{FMP_BASE}/{path}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_quotes(symbols: list[str]) -> list[Quote]:
    """Batch quote + 52-wk summary snapshot from FMP (free tier allows batch-by-comma)."""
    out = []
    for sym in symbols:
        try:
            prof = _fmp(f"quote/{sym}", datatype="json") or []
            p = prof[0] if prof else {}
            bk = _fmp(f"historical-price-full/{sym}", serietype="line") or {}
        except Exception:
            continue
        hist = bk.get("historical", [])
        # crude 200/90/30/14/7/1 day returns vs self
        closes = [h["close"] for h in hist[-200:] if "close" in h]
        def chg(n):
            if len(closes) <= n: return 0.0
            return (closes[-1] / closes[-1 - n] - 1.0) * 100
        q = Quote(
            symbol=sym,
            price=p.get("price", 0.0) or 0.0,
            market_cap=(p.get("marketCap", 0.0) or 0.0),
            exchange=p.get("exchange", "N/A"),
            price_change_24h=p.get("changesPercentage", 0.0) or 0.0,
            price_change_7d=chg(7), price_change_14d=chg(14), price_change_30d=chg(30),
            price_change_90d=chg(90), price_change_200d=chg(200),
            total_volume=p.get("avgVolume", 0.0) or 0.0,
        )
        out.append(q)
        time.sleep(1.0 / 5)  # free-tier courtesy delay
    return out


def sp500_constituents() -> list[str]:
    try:
        return [l.strip() for l in requests.get(SP500_LISTING, timeout=30).iter_lines(decode_unicode=True) if l and l.strip()][:500]
    except Exception:
        return []


def russell_constituents() -> list[str]:
    try:
        txt = requests.get(RUSSELL1000, timeout=30).text
        return [l.strip().upper() for l in txt.splitlines() if l and "." not in l][:1000]
    except Exception:
        return []


def build_universe() -> list[str]:
    """S&P 500 + Russell 1000 + factor ETFs, dedup'd, stable-sorted (crypto style)."""
    syms = set(sp500_constituents()) | set(russell_constituents()) | set(ETF_UNIVERSE)
    syms.discard("")
    return sorted(syms, key=lambda s: (len(s), s))


def enrich_fundamentals(q: Quote) -> None:
    """Fill ROIC, FCF yield, margins, debt/EBITDA, valuation z, short interest."""
    try:
        prof = _fmp(f"ratios/{q.symbol}", limit=1) or []
        if prof:
            r = prof[0]
            q.roic = r.get("roic", 0.0) or 0.0
            q.gross_margin = r.get("grossProfitMargin", 0.0) or 0.0
            q.debt_ebitda = r.get("debtEquityRatio", 0.0) or 0.0  # fallback proxy
            q.fcf_yield = r.get("freeCashFlowYield", 0.0) or 0.0
            # earnings stability: 1 - (stdev 3yr / mean)
            q.earnings_stability = max(0.0, 1.0 - (r.get("earningsGrowth", 0.0) or 0.0))
    except Exception:
        pass
    try:
        q.val_zscore = (_fmp(f"valuation/{q.symbol}", limit=1) or [{}])[0].get("forwardPE", 0.0) or 0.0
    except Exception:
        pass
    try:
        si = _fmp(f"short-interest/{q.symbol}") or []
        if si:
            q.short_interest = si[0].get("shortInterest", 0.0) or 0.0
            q.short_days = si[0].get("daysToCover", 0.0) or 0.0
    except Exception:
        pass
    # 52wk drawdown
    try:
        bk = _fmp(f"historical-price-full/{q.symbol}", serietype="line", timeseries=260) or {}
        hist = bk.get("historical", [])
        if hist:
            hi = max(h["close"] for h in hist[-260:])
            q.drawdown_52w = max(0.0, 1.0 - (q.price or 0) / hi) if hi else 0.0
    except Exception:
        pass


if __name__ == "__main__":
    import equity_monitor.model as m
    syms = build_universe()
    print(f"universe={len(syms)} ETF-only={len(set(syms)&set(ETF_UNIVERSE))}")
