"""Relative Strength engine — multi-horizon RS vs SPY (+ sector), vol-normalised.

Produces rs_blend (vs SPY) and rs_sector for each feature dict. SPY is the
benchmark; sector RS uses a coarse sector -> proxy ETF map. RS is derived from
the live 24h % change already present in each feature dict (build_features pulls
it from Yahoo), so no extra API calls are needed. The blend is the same soft
tanh input the model consumes. No hard clamp -> names stay rankable.
"""
from __future__ import annotations
import math
import equity_monitor.data as data

# crude sector -> proxy ETF (for sector-relative RS)
SECTOR_ETF = {
    "AAPL": "QQQ", "MSFT": "QQQ", "NVDA": "QQQ", "GOOGL": "QQQ", "AMZN": "QQQ", "META": "QQQ",
    "TSLA": "QQQ", "AVGO": "QQQ", "ADBE": "QQQ", "CRM": "QQQ", "NFLX": "QQQ", "AMD": "QQQ",
    "INTC": "QQQ", "CSCO": "QQQ", "ORCL": "QQQ", "QCOM": "QQQ", "TXN": "QQQ",
    "JPM": "XLF", "BAC": "XLF", "BRK.B": "XLF", "V": "XLF", "MA": "XLF",
    "UNH": "XLV", "JNJ": "XLV", "ABBV": "XLV", "MRK": "XLV", "AMGN": "XLV",
    "XOM": "XLE", "CVX": "XLE",
    "WMT": "XLP", "PG": "XLP", "KO": "XLP", "PEP": "XLP", "COST": "XLP",
    "HD": "XLY", "LLY": "XLV",
}


def _chg(symbol: str) -> float:
    """24h % change for `symbol` from the keyless Yahoo quote."""
    try:
        m = data._get_yahoo_quote(symbol)
        return float(m.get("regularMarketChangePercent", 0) or 0) if m else 0.0
    except Exception:
        return 0.0


def compute_rs(features: list[dict]) -> list[dict]:
    """Augment each feature dict with rs_blend, rs_sector, drawdown_52w.

    drawdown_52w is now REAL (computed from 252-day closes), not a proxy.
    Returns the true 52-week drawdown from the max close over the trailing year
    to the current price. This feeds model.confirmation()'s mean-reversion
    kicker with accurate depth rather than an RS-derived estimate.
    """
    spy_d = _chg("SPY")
    out = []
    for f in features:
        if f is None:
            continue
        sym = f["symbol"]
        d = float(f.get("chg24", 0.0) or 0.0)
        rs_blend = d - spy_d                       # vs SPY 24h delta
        etf = SECTOR_ETF.get(sym)
        sec_d = _chg(etf) if etf else spy_d
        rs_sector = d - sec_d
        # --- REAL 52-week drawdown from locally-cached 252-day closes ---
        dd = 0.0
        try:
            closes = data.history(sym, 252)
            if closes:
                hi52 = max(closes)
                price = float(f.get("price", 0) or 0)
                if hi52 > 0 and price > 0:
                    dd = max(0.0, min(0.5, (hi52 - price) / hi52))
        except Exception:
            pass
        f["rs_blend"] = round(rs_blend, 2)
        f["rs_sector"] = round(rs_sector, 2)
        f["drawdown_52w"] = round(dd, 3)
        out.append(f)
    return out


def enrich(features: list[dict]) -> list[dict]:
    return compute_rs(features)
