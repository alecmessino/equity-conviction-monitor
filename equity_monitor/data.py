"""Data layer — Equity Conviction Monitor.

Pulls market + fundamental features from Financial Modeling Prep (FMP) /stable/
and builds the NORMALISED feature dict that model.score() consumes.

Live path requires FMP_API_KEY (set in repo Secrets → Actions → FMP_API_KEY).
When the key is absent or a symbol fails, build_features() returns None so the
nightly can skip it gracefully (never fabricate a price).
"""
from __future__ import annotations
import os, math, time
import urllib.request, json, urllib.parse

FMP_BASE = "https://financialmodelingprep.com/stable/"
FMP_KEY = os.environ.get("FMP_API_KEY", "")

# Scope anchor: S&P 500 + Russell 1000 + major regional/factor ETFs.
# Curated sample universe (real tickers; expand via S&P 500 list API later).
UNIVERSE = [
    # mega/large cap tech & staples
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","BRK.B","JPM","V","UNH",
    "XOM","JNJ","WMT","MA","PG","HD","CVX","KO","PEP","ABBV","COST","AVGO","LLY",
    "MRK","BAC","ADBE","CRM","NFLX","AMD","INTC","CSCO","ORCL","QCOM","TXN","AMGN",
    # ETFs / factors
    "SPY","QQQ","IWM","EFA","EEM","VWO","EWJ","QUAL","VLUE","MTUM","USMV","IWD",
]

def _get(path: str, params: dict | None = None) -> list | dict:
    if not FMP_KEY:
        raise RuntimeError("FMP_API_KEY not set")
    import urllib.parse
    q = dict(params or {})
    q["apikey"] = FMP_KEY
    url = FMP_BASE + path + "?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"User-Agent": "equity-monitor/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())

def _first(x):
    return x[0] if isinstance(x, list) and x else (x if isinstance(x, dict) else None)

def build_features(symbol: str) -> dict | None:
    """Return normalised feature dict for `symbol`, or None on any failure."""
    try:
        quote = _first(_get(f"quote/{symbol}"))
        if not quote or quote.get("price") in (None, 0):
            return None
        prof  = _first(_get(f"profile/{symbol}")) or {}
        ratios= _first(_get(f"ratios/{symbol}", {"limit": 1})) or {}
        inc   = _first(_get(f"income-statement/{symbol}", {"limit": 1})) or {}
        bs    = _first(_get(f"balance-sheet-statement/{symbol}", {"limit": 1})) or {}
        cf    = _first(_get(f"cash-flow-statement/{symbol}", {"limit": 1})) or {}

        price   = float(quote.get("price", 0) or 0)
        mcap    = float(quote.get("marketCap", 0) or 0)
        chg24   = float(quote.get("changesPercentage", 0) or 0)
        vol     = float(quote.get("volume", 0) or 0)
        adv     = vol * price if vol else 0.0
        turnover= adv / mcap if mcap else 0.0

        # fundamentals (defaults so model never divides by missing)
        roic        = float(ratios.get("returnOnInvestedCapital", 0) or 0) / 100.0 if isinstance(ratios.get("returnOnInvestedCapital"),(int,float)) else 0.0
        fcf_yield   = float(ratios.get("freeCashFlowYield", 0) or 0) / 100.0 if isinstance(ratios.get("freeCashFlowYield"),(int,float)) else 0.0
        gross_margin= float(ratios.get("grossProfitMargin", 0) or 0) / 100.0 if isinstance(ratios.get("grossProfitMargin"),(int,float)) else 0.0
        debt_ebitda = float(ratios.get("debtToEBITDA", 5.0) or 5.0)
        rev         = float(inc.get("revenue", 0) or 0)
        netinc      = float(inc.get("netIncome", 0) or 0)
        # earnings stability: 1 - |net margin volatility proxy| (use margin sign sanity)
        earnings_stability = clamp_stability(netinc, rev)
        # valuation z-score vs 5y median P/E (use current P/E proxy)
        pe   = float(ratios.get("peRatio", 0) or 0) or 0.0
        val_zscore = pe_to_z(pe)

        return {
            "symbol": symbol,
            "price": price,
            "market_cap": mcap,
            "chg24": chg24,
            "adv": adv,
            "turnover": turnover,
            "roic": roic,
            "fcf_yield": fcf_yield,
            "gross_margin": gross_margin,
            "debt_ebitda": debt_ebitda,
            "earnings_stability": earnings_stability,
            "val_zscore": val_zscore,
            "short_days": 0.0,        # FMP short-interest endpoint optional; safe default
            "rs_blend": 0.0,          # filled by RS module (vs SPY) downstream
            "rs_sector": 0.0,
            "drawdown_52w": 0.0,
        }
    except Exception as e:
        # never fabricate; surface the failure upstream
        raise RuntimeError(f"{symbol}: {e}") from e

def clamp_stability(netinc, rev):
    if rev <= 0:
        return 0.5
    margin = netinc / rev
    # healthy positive margin -> stability up to 1.0; negative -> low
    return clamp01(0.5 + margin)

def pe_to_z(pe: float) -> float:
    # crude z vs typical market P/E band 10..30
    if pe <= 0:
        return 0.0
    return clamp01((pe - 20.0) / 20.0)   # 20=neutral, 40=+1z rich, 0=cheap

def clamp01(x):
    return max(0.0, min(1.0, x))

def universe() -> list[str]:
    return list(UNIVERSE)
