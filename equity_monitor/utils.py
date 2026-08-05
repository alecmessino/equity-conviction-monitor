"""Shared helpers for the equity scorer."""
from __future__ import annotations

# --- feature z-score reference bands (5-yr norms) ---
# These anchor relative-strength + fundamental z-scores to a common 0..1 scale.
# Values tuned to large-cap US equity distribution; micro-cap / distressed
# tails are intentionally NOT clamped at the input stage (the multiplicative
# gate handles outliers via the mcap-aware risk floor).
ZSCOREF = {
    "roic": (0.08, 0.12),        # (mean, std) 8% mean ROIC, 12pp spread
    "fcf_yield": (0.04, 0.06),   # 4% FCF yield, 6pp
    "gross_margin": (0.40, 0.20),
    "debt_ebitda": (1.5, 1.0),   # higher = worse; z inverted in model
    "earnings_stability": (0.70, 0.15),  # 3-yr earnings std-dev inverted: 0.70 mean
    "val_zscore": (0.0, 1.0),    # forward P/E / EV-EBITDA vs 5yr median; already a z
    "rs_blend_spy": (0.0, 0.30), # 30pp std on RS vs SPY
    "short_days": (3.0, 3.0),    # 3 days mean, 3 days std
    "turnover": (0.01, 0.008),   # 1% mean turnover
}

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def zscore(value: float, band: str) -> float:
    """Standardise `value` against the reference band, NO clamping (outliers pass
    through; the multiplicative structure attenuates them via the risk floor)."""
    mean, std = ZSCOREF.get(band, (0.0, 1.0))
    if std <= 0:
        return 0.0
    return (value - mean) / std
