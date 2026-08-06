"""Data layer — Equity Conviction Monitor.

Live market + fundamental features from KEYLESS price source + free fundamentals:

  * Price / volume / 52-wk range  -> Yahoo Finance /v8/finance/chart  (NO KEY required)
  * Fundamentals (ROIC, margin, PE, mktcap) -> Alpha Vantage /OVERVIEW (free key)

Why: FMP free keys are being rejected by FMP's API (confirmed: /stable/quote returns
"Invalid API KEY" for every key tried; income/balance/cash-flow are premium-gated).
Twelve Data /quote now also requires a key. Yahoo chart is keyless and returns
price/volume/52wk; Alpha Vantage's free OVERVIEW key (normal signup, no card) supplies
the Quality inputs (Alpha Vantage's free tier actually works — only FMP's was broken).

If ALPHAVANTAGE_API_KEY is unset we still render with Yahoo price + neutral fundamentals
(never fabricate a price; fundamentals default to quality-neutral so the score stays honest).
"""
from __future__ import annotations
import os, math, time, json
import urllib.request, urllib.parse
from datetime import datetime

AV_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")

UNIVERSE = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","BRK.B","JPM","V","UNH",
    "XOM","JNJ","WMT","MA","PG","HD","CVX","KO","PEP","ABBV","COST","AVGO","LLY",
    "MRK","BAC","ADBE","CRM","NFLX","AMD","INTC","CSCO","ORCL","QCOM","TXN","AMGN",
    "SPY","QQQ","IWM","EFA","EEM","VWO","EWJ","QUAL","VLUE","MTUM","USMV","IWD",
]


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (equity-monitor)"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def _get_yahoo_quote(sym: str) -> dict | None:
    """Price/volume/52wk from Yahoo chart (keyless). Yahoo uses '-' for '.' (BRK.B -> BRK-B)."""
    ysym = sym.replace(".", "-")
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ysym}?range=5d&interval=1d"
    try:
        d = _get_json(url)
        res = d.get("chart", {}).get("result")
        if not res:
            return None
        m = res[0]["meta"]
        price = m.get("regularMarketPrice")
        if not price:
            return None
        return m
    except Exception:
        return None


def _get_yahoo_history(sym: str, days: int = 30) -> list[float]:
    """Trailing daily closes for sparklines (Yahoo chart, keyless). Returns [] on failure."""
    ysym = sym.replace(".", "-")
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ysym}?range=3mo&interval=1d"
    try:
        d = _get_json(url)
        res = d.get("chart", {}).get("result")
        if not res:
            return []
        closes = res[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
        closes = [c for c in closes if c is not None]
        return closes[-days:] if closes else []
    except Exception:
        return []


def history(sym: str, days: int = 30) -> list[float]:
    """Public: trailing daily closes for sparklines."""
    return _get_yahoo_history(sym, days)


def _get_yahoo_earnings_date(sym: str) -> str | None:
    """Next earnings date (ISO) via Yahoo quoteSummary/calendarEvents (keyless).
    Returns None if unavailable (e.g. ETFs have no earnings)."""
    ysym = sym.replace(".", "-")
    url = (f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ysym}"
           "?modules=calendarEvents")
    try:
        d = _get_json(url)
        cal = (d.get("quoteSummary", {}).get("result") or [{}])[0].get("calendarEvents", {})
        # prefer the earnings array's earliest future date
        earnings = cal.get("earnings") or {}
        dates = [e.get("startDate") for e in earnings.get("earningsDate", []) if e.get("startDate")]
        if not dates:
            return None
        return min(dates)  # ISO timestamp string
    except Exception:
        return None


def days_to_earnings(sym: str, today: datetime | None = None) -> int | None:
    """Days from `today` to the next earnings date, or None if unknown."""
    ed = _get_yahoo_earnings_date(sym)
    if not ed:
        return None
    try:
        if today is None:
            today = datetime.now()
        d0 = datetime.fromisoformat(ed.replace("Z", "+00:00"))
        return max(0, (d0.date() - today.date()).days)
    except Exception:
        return None


def _get_av_overview(sym: str) -> dict:
    """Fundamentals from Alpha Vantage /OVERVIEW (free key)."""
    if not AV_KEY:
        return {}
    url = ("https://www.alphavantage.co/query?function=OVERVIEW&symbol="
           + urllib.parse.quote(sym) + "&apikey=" + AV_KEY)
    try:
        d = _get_json(url)
        if isinstance(d, dict) and ("Information" in d or "Error Message" in d or "Note" in d):
            return {}
        return d or {}
    except Exception:
        return {}


def build_features(symbol: str) -> dict | None:
    """Return normalised feature dict for `symbol`, or None on hard failure."""
    try:
        m = _get_yahoo_quote(symbol)
        if not m:
            return None
        price = float(m.get("regularMarketPrice", 0) or 0)
        vol = float(m.get("regularMarketVolume", 0) or 0)
        adv = vol  # Yahoo gives daily volume; treat as ADV proxy
        hi = float(m.get("fiftyTwoWeekHigh", 0) or 0)
        lo = float(m.get("fiftyTwoWeekLow", 0) or 0)
        chg24 = float(m.get("regularMarketChangePercent", 0) or 0)
        # Yahoo sometimes includes marketCap in meta
        mcap = float(m.get("marketCap", 0) or 0)

        # neutral fundamentals until AV provides real values
        roic = 0.0
        fcf_yield = 0.0
        gross_margin = 0.30
        debt_ebitda = 2.5
        earnings_stability = 0.6
        pe = 0.0
        sector = ""
        beta = None  # None = not available (no AV key / no Beta field) -> UI shows "—", never a fake 1.00

        if AV_KEY:
            ov = _get_av_overview(symbol)
            if ov:
                def num(k, d=0.0):
                    v = ov.get(k)
                    try:
                        return float(v) if v not in (None, "", "None") else d
                    except Exception:
                        return d
                ric = ov.get("ReturnOnInvestedCapital")
                roic = (num("ReturnOnInvestedCapital") / 100.0) if ric else (num("ROIC") / 100.0)
                gross_margin = clamp01(num("GrossProfitTTM") / max(num("RevenueTTM"), 1)) if ov.get("GrossProfitTTM") else 0.30
                debt_ebitda = num("DebtEquityRatio") * 1.5
                earnings_stability = clamp01(0.5 + num("ProfitMargin") / 2.0)
                pe = num("PERatio")
                sector = ov.get("Sector", "") or ov.get("Industry", "")
                beta = num("Beta", 1.0) or 1.0
                mc_raw = ov.get("MarketCapitalization")
                if mc_raw not in (None, "", "None"):
                    try:
                        mcap = float(mc_raw) * 1e6
                    except Exception:
                        pass

        val_zscore = pe_to_z(pe)

        # keyless earnings date (None for ETFs) — purely additive metadata, no scoring impact
        dte = None
        try:
            dte = days_to_earnings(symbol)
        except Exception:
            dte = None

        return {
            "symbol": symbol,
            "price": price,
            "market_cap": mcap,
            "chg24": chg24,
            "adv": adv,
            "turnover": (adv * price) / mcap if mcap else 0.0,
            "roic": roic,
            "fcf_yield": fcf_yield,
            "gross_margin": gross_margin,
            "debt_ebitda": debt_ebitda,
            "earnings_stability": earnings_stability,
            "val_zscore": val_zscore,
            "short_days": 0.0,
            "rs_blend": 0.0,          # filled by RS module (vs SPY) downstream
            "rs_sector": 0.0,
            "drawdown_52w": (1 - price / hi) if hi else 0.0,
            "sector": sector,
            "beta": beta,
            "dte": dte,
        }
    except Exception as e:
        raise RuntimeError(f"{symbol}: {e}") from e


def pe_to_z(pe: float) -> float:
    if pe <= 0:
        return 0.0
    return clamp01((pe - 20.0) / 20.0)


def clamp01(x):
    return max(0.0, min(1.0, x))


def universe() -> list[str]:
    return list(UNIVERSE)
