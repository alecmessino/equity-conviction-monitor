"""Daily OHLCV with a source fallback chain.

Design principle: fetch **one series per symbol and derive everything from it.**

The previous implementation read individual scalars out of a vendor's quote
"meta" blob — ``regularMarketChangePercent``, ``marketCap``, ``fiftyTwoWeekHigh``.
Yahoo's chart endpoint does not return the first two, so every name's daily
change was silently 0, which made relative strength identically 0 across the
entire universe. Nothing in the code path failed; the dashboard just quietly
stopped discriminating between companies.

Deriving from a price series cannot fail that way. Either we have bars or we do
not, and if we do, every price feature follows arithmetically.

Chain: Nasdaq (works reliably from datacenter IPs) -> Yahoo v8 chart (works from
GitHub Actions, rate-limits elsewhere) -> the committed history cache.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import date, timedelta

from ._http import fetch_json

NASDAQ = ("https://api.nasdaq.com/api/quote/{sym}/historical"
          "?assetclass={ac}&fromdate={frm}&todate={to}&limit=900")
YAHOO = "https://query2.finance.yahoo.com/v8/finance/chart/{sym}?range=2y&interval=1d"

_NASDAQ_KW = dict(headers={"Accept": "application/json"},
                  throttle_key="nasdaq", min_interval=0.15, retries=2)
_YAHOO_KW = dict(throttle_key="yahoo", min_interval=0.4, retries=2)


@dataclass
class Bars:
    """A chronological daily OHLCV series (oldest first)."""

    symbol: str
    dates: list[str]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[float]
    source: str = ""

    def __len__(self) -> int:
        return len(self.close)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "source": self.source, "dates": self.dates,
            "open": self.open, "high": self.high, "low": self.low,
            "close": self.close, "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Bars | None":
        closes = d.get("close") or d.get("closes")
        if not closes:
            return None
        n = len(closes)

        def col(key: str) -> list[float]:
            v = d.get(key)
            return list(v) if v and len(v) == n else list(closes)

        return cls(
            symbol=d.get("symbol", ""), dates=d.get("dates") or [""] * n,
            open=col("open"), high=col("high"), low=col("low"),
            close=list(closes), volume=d.get("volume") or [0.0] * n,
            source=d.get("source", "cache"),
        )


def _num(x) -> float | None:
    """Parse Nasdaq's display-formatted numbers: '$311.00', '49,438,760', 'N/A'."""
    if x is None:
        return None
    s = str(x).replace("$", "").replace(",", "").strip()
    if not s or s in {"N/A", "--"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _from_nasdaq(symbol: str, asset_class: str) -> Bars | None:
    today = date.today()
    url = NASDAQ.format(sym=symbol, ac=asset_class,
                        frm=(today - timedelta(days=760)).isoformat(), to=today.isoformat())
    payload = fetch_json(url, **_NASDAQ_KW)
    rows = (((payload.get("data") or {}).get("tradesTable") or {}).get("rows")) or []
    if not rows:
        return None
    bars = Bars(symbol, [], [], [], [], [], [], source="nasdaq")
    for row in reversed(rows):  # Nasdaq returns newest-first
        c = _num(row.get("close"))
        if c is None:
            continue
        m, d_, y = (row.get("date") or "//").split("/")
        bars.dates.append(f"{y}-{m}-{d_}" if y else "")
        bars.close.append(c)
        bars.open.append(_num(row.get("open")) or c)
        bars.high.append(_num(row.get("high")) or c)
        bars.low.append(_num(row.get("low")) or c)
        bars.volume.append(_num(row.get("volume")) or 0.0)
    return bars if len(bars) >= 30 else None


def _from_yahoo(symbol: str) -> Bars | None:
    payload = fetch_json(YAHOO.format(sym=symbol.replace(".", "-")), **_YAHOO_KW)
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        return None
    node = result[0]
    quote = ((node.get("indicators") or {}).get("quote") or [{}])[0]
    stamps = node.get("timestamp") or []
    closes = quote.get("close") or []
    bars = Bars(symbol, [], [], [], [], [], [], source="yahoo")
    for i, c in enumerate(closes):
        if c is None:
            continue
        ts = stamps[i] if i < len(stamps) else None
        bars.dates.append(date.fromtimestamp(ts).isoformat() if ts else "")
        bars.close.append(float(c))
        for key, target in (("open", bars.open), ("high", bars.high), ("low", bars.low)):
            v = (quote.get(key) or [None] * len(closes))[i]
            target.append(float(v) if v is not None else float(c))
        v = (quote.get("volume") or [None] * len(closes))[i]
        bars.volume.append(float(v) if v is not None else 0.0)
    return bars if len(bars) >= 30 else None


def _from_cache(symbol: str, cache_dir: str) -> Bars | None:
    path = os.path.join(cache_dir, f"{symbol}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            bars = Bars.from_dict(json.load(fh))
    except Exception:
        return None
    if bars:
        bars.symbol, bars.source = symbol, "cache"
    return bars


def load(symbol: str, *, asset_class: str = "stocks", cache_dir: str = "") -> Bars | None:
    """Daily bars for `symbol`, trying each source until one yields a usable series.

    Returns None only if every source fails, which the caller must treat as
    "drop this name" rather than "this name is worth zero".
    """
    for attempt in (
        lambda: _from_nasdaq(symbol, asset_class),
        lambda: _from_nasdaq(symbol, "etf" if asset_class == "stocks" else "stocks"),
        lambda: _from_yahoo(symbol),
        lambda: _from_cache(symbol, cache_dir) if cache_dir else None,
    ):
        try:
            bars = attempt()
        except Exception:
            bars = None
        if bars and len(bars) >= 30:
            return bars
    return None


# ---------------------------------------------------------------------------
# price-derived features — all computed locally from the bars above
# ---------------------------------------------------------------------------
def _ret(closes: list[float], lookback: int) -> float | None:
    """Simple return over `lookback` trading days, or None if the series is too short."""
    if len(closes) <= lookback:
        return None
    prior = closes[-1 - lookback]
    if prior <= 0:
        return None
    return closes[-1] / prior - 1.0


def realized_vol(closes: list[float], window: int = 63) -> float | None:
    """Annualised stdev of daily log returns over the trailing `window` days."""
    tail = closes[-(window + 1):]
    if len(tail) < 20:
        return None
    rets = [math.log(tail[i] / tail[i - 1]) for i in range(1, len(tail)) if tail[i - 1] > 0]
    if len(rets) < 10:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252)


def atr(bars: Bars, window: int = 14) -> float | None:
    """Wilder's true range average — uses real highs and lows.

    The prior version approximated ATR from close-to-close moves, which
    systematically understates intraday range and therefore placed stops too tight.
    """
    n = len(bars)
    if n < window + 1:
        return None
    trs = []
    for i in range(n - window, n):
        prev_close = bars.close[i - 1]
        trs.append(max(
            bars.high[i] - bars.low[i],
            abs(bars.high[i] - prev_close),
            abs(bars.low[i] - prev_close),
        ))
    return sum(trs) / len(trs) if trs else None


def _sma(closes: list[float], window: int) -> float | None:
    if len(closes) < window:
        return None
    return sum(closes[-window:]) / window


def features(bars: Bars) -> dict:
    """Every price-derived feature, from one series."""
    c = bars.close
    price = c[-1]
    prev = c[-2] if len(c) > 1 else price
    window52 = c[-252:] if len(c) >= 252 else c
    hi52, lo52 = max(window52), min(window52)

    dollar_volume = [bars.close[i] * bars.volume[i] for i in range(len(c))][-63:]
    adv_usd = sum(dollar_volume) / len(dollar_volume) if dollar_volume else 0.0

    ma50, ma200 = _sma(c, 50), _sma(c, 200)
    a = atr(bars)

    return {
        "price": price,
        "chg_1d": (price / prev - 1.0) if prev > 0 else 0.0,
        "ret_1m": _ret(c, 21),
        "ret_3m": _ret(c, 63),
        "ret_6m": _ret(c, 126),
        "ret_12m": _ret(c, 252),
        "ret_ytd": None,  # filled by the caller, which knows the year boundary
        "vol_3m": realized_vol(c, 63),
        "vol_1y": realized_vol(c, 252),
        "atr14": a,
        "atr_pct": (a / price) if (a and price > 0) else None,
        "hi_52w": hi52,
        "lo_52w": lo52,
        "drawdown_52w": (hi52 - price) / hi52 if hi52 > 0 else None,
        "pct_off_low": (price - lo52) / lo52 if lo52 > 0 else None,
        "ma50": ma50,
        "ma200": ma200,
        "px_vs_ma50": (price / ma50 - 1.0) if ma50 else None,
        "px_vs_ma200": (price / ma200 - 1.0) if ma200 else None,
        "adv_usd": adv_usd,
        "bars": len(c),
        "source": bars.source,
        "last_date": bars.dates[-1] if bars.dates else "",
    }


def ytd_return(bars: Bars, today: date | None = None) -> float | None:
    """Return since the last close of the prior calendar year."""
    year = (today or date.today()).year
    base = None
    for i, d in enumerate(bars.dates):
        if d and d[:4] == str(year):
            base = bars.close[i - 1] if i > 0 else None
            break
    if not base or base <= 0:
        return None
    return bars.close[-1] / base - 1.0
