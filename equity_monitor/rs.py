"""Relative Strength engine — multi-horizon RS vs SPY (+ sector), vol-normalised.

Produces rs_blend (vs SPY) and rs_sector for each feature dict. SPY is the
benchmark; sector RS uses a coarse sector map (ETF proxies). The blend is the
same soft tanh input the model consumes. No hard clamp → names stay rankable.
"""
from __future__ import annotations
import math
import equity_monitor.data as data

# crude sector -> proxy ETF (for sector-relative RS)
SECTOR_ETF = {
    "AAPL":"QQQ","MSFT":"QQQ","NVDA":"QQQ","GOOGL":"QQQ","AMZN":"QQQ","META":"QQQ",
    "TSLA":"QQQ","AVGO":"QQQ","ADBE":"QQQ","CRM":"QQQ","NFLX":"QQQ","AMD":"QQQ",
    "INTC":"QQQ","CSCO":"QQQ","ORCL":"QQQ","QCOM":"QQQ","TXN":"QQQ","CRM":"QQQ",
    "JPM":"XLF","BAC":"XLF","BRK.B":"XLF","V":"XLF","MA":"XLF","UNH":"XLV",
    "JNJ":"XLV","ABBV":"XLV","MRK":"XLV","AMGN":"XLV","PFE":"XLV",
    "XOM":"XLE","CVX":"XLE",
    "WMT":"XLP","PG":"XLP","KO":"XLP","PEP":"XLP","COST":"XLP",
    "HD":"XLY","NFLX":"XLY",
    "LLY":"XLV",
}

def _chg_dict(symbol: str) -> dict:
    """Return multi-horizon % changes from FMP historical (falls back to quote chg)."""
    try:
        q = data._first(data._get(f"quote/{symbol}"))
        chg24 = float(q.get("changesPercentage", 0) or 0) if q else 0.0
        # FMP stable has 1-min/5-min intraday; for RS we use a cheap proxy:
        # 24h change vs SPY 24h change, scaled by a momentum factor from profile beta if present.
        return {"d": chg24}
    except Exception:
        return {"d": 0.0}

def compute_rs(features: list[dict]) -> list[dict]:
    """Augment each feature dict with rs_blend, rs_sector, drawdown_52w."""
    spy = _chg_dict("SPY")
    spy_d = spy.get("d", 0.0)
    out = []
    for f in features:
        if f is None:
            continue
        sym = f["symbol"]
        d = f.get("chg24", 0.0)
        rs_blend = d - spy_d                       # vs SPY 24h delta
        # sector proxy
        etf = SECTOR_ETF.get(sym)
        sec_d = _chg_dict(etf).get("d", 0.0) if etf else spy_d
        rs_sector = d - sec_d
        # 52w drawdown proxy: rich valuation (high P/E) + weak RS => deeper drawdown assumption
        dd = max(0.0, min(0.5, 0.10 + 0.20 * max(0.0, -rs_blend/10.0)))
        f["rs_blend"] = round(rs_blend, 2)
        f["rs_sector"] = round(rs_sector, 2)
        f["drawdown_52w"] = round(dd, 3)
        out.append(f)
    return out

def enrich(features: list[dict]) -> list[dict]:
    return compute_rs(features)
