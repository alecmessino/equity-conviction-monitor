"""Core scoring model — Equity Conviction Monitor (v2, multiplicative).

    Conviction = 100 * Quality * Confirmation * RiskAdjustment

Quality  (Q): fundamentals — ROIC, FCF yield, gross margin, leverage, earnings stability.
Confirmation (C): multi-horizon relative strength vs SPY (+ sector), soft tanh, NO hard clamp.
Risk (R): liquidity (dollar ADV tiers), short-interest drag, valuation z-score.

All inputs are NORMALISED feature dicts carrying the keys produced by
equity_monitor.data.build_features(). The same feature dict drives both the
Python score() and the JS port in web/terminal.html (parity-gated).
"""
from __future__ import annotations
import math
from .utils import clamp01

# ---- signal tiers (identical to crypto v2) ----
def signal(c: float) -> str:
    if c >= 80: return "STRONG"
    if c >= 70: return "BUY"
    if c >= 55: return "HOLD"
    if c >= 40: return "WATCH"
    return "AVOID"


# ---- Q: Structural Quality (range-normalised 0..1) ----
def _q(metric, threshold, value):
    return clamp01(value / threshold) if threshold else 0.0

def quality(t: dict):
    roic = _q("roic", 0.25, t.get("roic", 0.0) or 0.0)
    fcf  = _q("fcf", 0.08, t.get("fcf_yield", 0.0) or 0.0)
    bm   = _q("margin", 0.50, t.get("gross_margin", 0.0) or 0.0)
    lev  = clamp01(1.0 - (t.get("debt_ebitda", 5.0) or 5.0) / 3.0)
    g    = clamp01(t.get("earnings_stability", 0.5) or 0.5)
    raw  = 0.30*roic + 0.25*fcf + 0.15*bm + 0.20*lev + 0.10*g
    q    = clamp01((raw - 0.20) / 0.60)
    comp = {
        "quality": round(q*20, 1), "roic": round(roic, 2), "fcf_yield": round(fcf, 2),
        "gross_margin": round(bm, 2), "debt_ebitda": round(lev, 2),
        "earnings_stability": round(g, 2),
    }
    return q, comp


# ---- C: Market Confirmation (multi-horizon RS vs SPY + sector, vol-normalised) ----
def confirmation(t: dict):
    rs = t.get("rs_blend", 0.0) or 0.0
    c  = 0.10 + 0.90 * ((math.tanh(rs / 25.0) + 1.0) / 2.0)   # [0.10, 0.91]
    dd = t.get("drawdown_52w", 0.0) or 0.0
    if dd >= 0.15:
        c = min(0.95, c * (1 + 0.15 * (dd - 0.10)))   # mean-reversion kicker on deep drawdown
    comp = {
        "rs_blend": round(rs, 2), "rs_sector": round(t.get("rs_sector", 0.0), 2),
        "drawdown_52w": round(dd, 3), "confirmation": round(c*20, 1),
    }
    return c, comp


# ---- R: Risk Adjustment — liquidity (dollar ADV), short interest, valuation ----
def risk_adjustment(t: dict):
    # Liquidity fit from DOLLAR ADV tiers (varies meaningfully across the universe).
    # $1M/d=0.0  $10M=0.25  $100M=0.5  $1B=0.75  $10B+=1.0  (log scale)
    adv = float(t.get("adv", 0) or 0)
    if adv <= 0:
        liq = 0.15
    else:
        liq = clamp01((math.log10(adv) - 6.0) / 4.0)
    # Thin but not dead: floor at 0.25 so micro-caps aren't zeroed, never a blanket mega-cap grant
    liq = max(0.25, min(1.0, liq))

    dtc = t.get("short_days", 0.0) or 0.0
    short_mult = clamp01(1.0 - max(0.0, dtc - 3.0) / 30.0)
    val_z = t.get("val_zscore", 0.0) or 0.0
    val_mult = clamp01(1.0 + 0.15 * (-val_z))
    risk = liq * short_mult * val_mult
    risk = max(0.4, min(1.0, risk))
    comp = {
        "liquidity_fit": round(liq, 3), "short_days": round(dtc, 1),
        "short_mult": round(short_mult, 3), "val_zscore": round(val_z, 2),
        "val_mult": round(val_mult, 3), "risk": round(risk, 3),
    }
    return risk, comp


def score(t: dict, bench: dict | None = None, sector: dict | None = None):
    b, s = bench or {"symbol": "SPY"}, sector or {}
    q, qc = quality(t)
    c, cc = confirmation(t)
    r, rc = risk_adjustment(t)
    raw = 100 * q * c * r
    conv = int(round(max(0, min(100, raw))))
    comp = {
        "quality": qc["quality"], "confirmation": cc["confirmation"], "risk": rc["risk"],
        "rs_blend": cc["rs_blend"], "liquidity_fit": rc["liquidity_fit"],
        "val_zscore": rc["val_zscore"], "short_days": rc["short_days"],
    }
    return q, c, r, conv, signal(conv), comp
