"""Core scoring model — Equity Conviction Monitor.

Philosophy (mirrors crypto v2): multiplicative, no hard clamp on confirmation,
quality is the structural floor, risk is a bounded adjustment.

    Conviction = 100 * Quality * Confirmation * RiskAdjustment

Inputs are normalised quote dicts (see data.FMPQuote) carrying at minimum:
    symbol, market_cap, total_revenue, ebitda, net_debt, ebit, price_change_24h,
    price_change_7d/14d/30d/90d/200d (vs self), rs_blend (vs SPY), rs_sector,
    turnover (vol/mc), adv, short_interest, short_days, fcf_yield, roic,
    gross_margin, debt_ebitda, earnings_stability, drawdown_52w
"""
from __future__ import annotations
import math
from .utils import clamp01, zscore

# ---- signal tiers (same bands as crypto v2; readability kept consistent) ----
def signal(c: float) -> str:
    """STRONG>=80 BUY>=70 HOLD>=55 WATCH>=40 AVOID<40 — identical to crypto v2."""
    if c >= 80: return "STRONG"
    if c >= 70: return "BUY"
    if c >= 55: return "HOLD"
    if c >= 40: return "WATCH"
    return "AVOID"


# ---- Q: Structural Quality (fundamentals) — range-normalised, 0..1 ----
# Each metric is mapped 0..1 against an "excellent" threshold via clamp01; no
# universe-dependent z-score (score() runs on a single asset). Strong mega-caps
# (ROIC>25%, FCF yield>6%, clean sheet, stable earnings) land q in 0.8..1.0.
def _q(metric, threshold, value):
    """value/threshold clamped to [0,1]; inverted for debt (lower=better handled inline)."""
    return clamp01(value / threshold) if threshold else 0.0

def quality(t: dict) -> tuple[float, dict]:
    roic = _q("roic", 0.25, t.get("roic", 0.0) or 0.0)          # 25% ROIC = excellent
    fcf = _q("fcf", 0.08, t.get("fcf_yield", 0.0) or 0.0)     # 8% FCF yield = excellent
    bm = _q("margin", 0.50, t.get("gross_margin", 0.0) or 0.0)  # 50% gross = excellent
    lev = clamp01(1.0 - (t.get("debt_ebitda", 5.0) or 5.0) / 3.0)  # 0.0 debt/ebitda -> 1.0, >3 -> 0
    g = clamp01((t.get("earnings_stability", 0.5) or 0.5))       # 0..1 already
    raw = 0.30 * roic + 0.25 * fcf + 0.15 * bm + 0.20 * lev + 0.10 * g
    q = clamp01((raw - 0.2) / 0.6)  # shift so strong names ~0.9, middling ~0.5, weak ~0.1
    # blue-chip floor: never let mcap-weighted quality die for a $1B+ name
    mc = t.get("market_cap", 0) or 1
    depth = clamp01((math.log10(mc) - 6.0) / 4.0)
    if depth >= 0.90:
        q = max(q, 0.4)
    comp = {"quality": round(q * 20, 1), "roic": round(roic, 2), "fcf_yield": round(fcf, 2),
            "gross_margin": round(bm, 2), "debt_ebitda": round(lev, 2),
            "earnings_stability": round(g, 2), "depth": round(depth * 20, 1)}
    return q, comp


# ---- C: Market Confirmation (multi-horizon RS vs SPY + sector, vol-normalised) ----
def confirmation(t: dict, bench: dict) -> tuple[float, dict]:
    """Soft sigmoid over rs_blend — NO hard clamp, so names within a tier stay
    rankable by RS strength (the v1 collision fix)."""
    rs = t.get("rs_blend", 0.0) or 0.0
    c = 0.10 + 0.90 * ((math.tanh(rs / 25.0) + 1.0) / 2.0)  # [0.10, 0.91]
    # mean-reversion kicker: deeper 52w drawdown of a HIGH-quality name is confirmation
    # of an unwinding, not a risk — lift C modestly (capped to preserve monotonicity)
    dd = t.get("drawdown_52w", 0.0) or 0.0   # 0..1, fraction below 52w high
    if dd >= 0.15 and t.get("roic", 0) >= 0:
        c = min(0.95, c * (1 + 0.15 * (dd - 0.10)))
    comp = {"rs_blend": round(rs, 2), "rs_sector": round(t.get("rs_sector", 0.0), 2),
            "drawdown_52w": round(dd, 3), "confirmation": round(c * 20, 1)}
    return c, comp


# ---- R: Risk Adjustment — liquidity + short interest + valuation ----
def risk_adjustment(t: dict, bench: dict) -> tuple[float, dict]:
    """Liquidity fit (ADV/mc), short-interest drag (days-to-cover), and a
    valuation z-score modifier. Bounded [0.4, 1.0]; blue-chips (depth>=0.90)
    get the liquidity floor so they are not punished for low %. """
    mc = t.get("market_cap", 0) or 1
    adv = t.get("adv", 0) or t.get("turnover", 0) or 1e9
    turn = adv / mc if mc else 0
    mc_depth = math.log10(mc)
    # liquidity fit: warm turn (0.5%-1.5%) is the sweet spot for large caps
    if turn <= 0:
        liq = 0.4
    elif turn <= 0.015:
        liq = 0.4 + (turn / 0.015) * 0.6          # ramp to 1.0 by 1.5% turnover
    elif turn <= 0.03:
        liq = 1.0
    else:
        liq = 1.0 - 0.15 * math.log10(turn / 0.03)   # mild decay for frothy turnover
    liq = max(0.4, min(1.0, liq))
    # blue-chip floor (mirrors crypto: high-mcap perps/ETFs not punished for low turn)
    if mc_depth >= 9.0:   # ~$1B+ log10 floor; effectively all real blue-chips
        liq = max(liq, 0.85)
    # short-interest drag: days-to-cover >= 8 is material risk
    dtc = t.get("short_days", 0.0) or 0.0
    short_mult = clamp01(1.0 - max(0.0, dtc - 3.0) / 30.0)   # 3d->1.0, 33d->0.0
    # valuation: cheap vs 5yr median -> mild boost, expensive -> mild haircut
    val_z = t.get("val_zscore", 0.0) or 0.0
    val_mult = clamp01(1.0 + 0.15 * (-val_z))    # cheap (neg z) lifts, rich (pos z) dings
    risk = liq * short_mult * val_mult
    risk = max(0.4, min(1.0, risk))
    comp = {"liquidity_fit": round(liq, 3), "short_days": round(dtc, 1),
            "short_mult": round(short_mult, 3), "val_zscore": round(val_z, 2),
            "val_mult": round(val_mult, 3), "risk": round(risk, 3)}
    return risk, comp


# ---- top-level ----
def score(t: dict, bench: dict | None = None, sector: dict | None = None) -> tuple[float, float, float, float, str, dict]:
    """Equity v2 score.

    Returns (quality, confirmation, risk, conviction, signal, comp).
    """
    b = bench or {"symbol": "SPY"}
    s = sector or {}
    q, qc = quality(t)
    c, cc = confirmation(t, b)
    r, rc = risk_adjustment(t, b)
    raw = 100 * q * c * r
    conv = int(round(max(0, min(100, raw))))  # clamp to [0,100] int, like crypto v2 nightly
    comp = {"quality": qc["quality"], "confirmation": cc["confirmation"],
            "risk": rc["risk"], "rs_blend": cc["rs_blend"],
            "liquidity_fit": rc["liquidity_fit"], "val_zscore": rc["val_zscore"],
            "depth": qc["depth"]}
    sig = signal(conv)
    return q, c, r, conv, sig, comp
