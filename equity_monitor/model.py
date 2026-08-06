"""Scoring model v3 — cross-sectional, geometric.

    Conviction = 100 · (Q · C · R)^(1/3)

Two deliberate departures from v2, both of which were causes of the all-zero board:

1. **Percentile ranks, not absolute thresholds.** v2 asked "is ROIC above 25%?" and
   divided by a hand-picked constant. That makes the score a statement about the
   constant as much as about the company, and it collapses whenever the input is
   missing. v3 asks "where does this company sit against its sector?" — the same
   question Morningstar and YCharts answer — which guarantees dispersion, adapts to
   changing rate and margin regimes, and degrades gracefully.

2. **Geometric mean, not raw product.** v2 multiplied three sub-unit factors, so a
   thoroughly median company scored 0.5·0.5·0.5 → 12/100 and no name could plausibly
   reach the BUY threshold of 70. The geometric mean keeps the property that mattered
   — any pillar near zero vetoes the whole score, because you cannot recover from a
   zero factor — while placing a median name near 58. Rescaling was not cosmetic:
   under v2 the published thresholds described a tier structure the arithmetic could
   never actually reach.

``prepare()`` does the cross-sectional work and runs only in Python. ``score()`` is a
**pure function of already-ranked inputs**, which is what allows the identical function
to run in the browser (see the parity gate in tests/test_parity.py) and lets the
terminal re-weight the pillars live without a round trip.
"""
from __future__ import annotations

import math

# --- pillar weights. Mirrored verbatim in web/terminal.html; parity-gated. ---
WEIGHTS = {
    "quality": {
        "p_roic": 0.30,
        "p_fcf_yield": 0.25,
        "p_gross_margin": 0.15,
        "p_leverage": 0.15,
        "p_earnings_stability": 0.15,
    },
    "confirmation": {
        "p_rs": 0.70,
        "p_trend": 0.30,
    },
    "risk": {
        "p_liquidity": 0.40,
        "p_value": 0.35,
        "p_lowvol": 0.25,
    },
}

# Pillar output ranges. The floors are non-zero on purpose: a company in the bottom
# percentile of its sector is a poor holding, not a nonexistent one, and mapping it to
# exactly 0 destroys all information below the floor.
Q_FLOOR, Q_SPAN = 0.02, 0.98
C_FLOOR, C_SPAN = 0.10, 0.85
R_FLOOR, R_SPAN = 0.35, 0.75
C_CEILING = 0.98

# Quality mean-reversion: the thesis this terminal is named after. A high-quality
# business trading well below its 52-week high gets a confirmation uplift; a low-quality
# one in the same drawdown gets nothing, because that is a falling knife, not a sale.
# v2 applied its kicker to every name regardless of quality, which is the opposite of
# the stated strategy.
MR_QUALITY_GATE = 0.55
MR_DRAWDOWN_GATE = 0.15
MR_DRAWDOWN_SPAN = 0.35
MR_MAX_UPLIFT = 0.30

SIGNAL_TIERS = [(80, "STRONG"), (70, "BUY"), (55, "HOLD"), (40, "WATCH")]

# Metrics ranked within GICS sector. Margins, leverage and valuation are only
# meaningful against peers — a software company's 70% gross margin and a grocer's 25%
# are both excellent, and ranking them against each other is a sector bet in disguise.
SECTOR_RELATIVE = {
    "p_roic", "p_fcf_yield", "p_gross_margin", "p_leverage",
    "p_earnings_stability", "p_value",
}
MIN_SECTOR_GROUP = 5

NEUTRAL = 0.5  # percentile assigned to an unobservable metric: the group median


def signal(conviction: float) -> str:
    for threshold, name in SIGNAL_TIERS:
        if conviction >= threshold:
            return name
    return "AVOID"


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# pure scoring — mirrored in JS
# ---------------------------------------------------------------------------
def _pillar(percentiles: dict, weights: dict) -> float:
    """Weighted mean of the available percentiles, renormalised over what is present."""
    total = 0.0
    seen = 0.0
    for key, weight in weights.items():
        v = percentiles.get(key)
        if v is None:
            continue
        total += weight * v
        seen += weight
    return (total / seen) if seen > 0 else NEUTRAL


def score(p: dict, weights: dict | None = None) -> dict:
    """Conviction from ranked inputs. Pure: same input always gives the same output.

    `p` carries percentile keys (``p_roic`` … ``p_lowvol``), each in [0, 1], plus the
    raw ``drawdown_52w`` used by the mean-reversion term.
    """
    w = weights or WEIGHTS

    q_raw = _pillar(p, w["quality"])
    c_raw = _pillar(p, w["confirmation"])
    r_raw = _pillar(p, w["risk"])

    Q = Q_FLOOR + Q_SPAN * q_raw
    C = C_FLOOR + C_SPAN * c_raw
    R = R_FLOOR + R_SPAN * r_raw

    drawdown = p.get("drawdown_52w") or 0.0
    uplift = 1.0
    if q_raw >= MR_QUALITY_GATE and drawdown > MR_DRAWDOWN_GATE:
        depth = min(1.0, (drawdown - MR_DRAWDOWN_GATE) / MR_DRAWDOWN_SPAN)
        uplift = 1.0 + MR_MAX_UPLIFT * depth
    C = min(C_CEILING, C * uplift)

    raw = 100.0 * (Q * C * R) ** (1.0 / 3.0)
    conviction = int(round(max(0.0, min(100.0, raw))))

    return {
        "conviction": conviction,
        "signal": signal(conviction),
        "q": round(Q, 4),
        "c": round(C, 4),
        "r": round(R, 4),
        "q_raw": round(q_raw, 4),
        "c_raw": round(c_raw, 4),
        "r_raw": round(r_raw, 4),
        "mr_uplift": round(uplift, 4),
    }


# ---------------------------------------------------------------------------
# cross-sectional preparation — Python only
# ---------------------------------------------------------------------------
def percentile_rank(values: list[float | None]) -> list[float | None]:
    """Fractional ranks in [0, 1], ties averaged. None passes through as None.

    A single observation ranks at the median: with no peers there is no evidence it
    is better or worse than one.
    """
    present = [(i, v) for i, v in enumerate(values) if v is not None]
    out: list[float | None] = [None] * len(values)
    n = len(present)
    if n == 0:
        return out
    if n == 1:
        out[present[0][0]] = NEUTRAL
        return out
    present.sort(key=lambda t: t[1])
    i = 0
    while i < n:
        j = i
        while j + 1 < n and present[j + 1][1] == present[i][1]:
            j += 1
        rank = (i + j) / 2.0
        for k in range(i, j + 1):
            out[present[k][0]] = rank / (n - 1)
        i = j + 1
    return out


# (percentile key, source field, higher-is-better)
RANK_SPEC: list[tuple[str, str, bool]] = [
    ("p_roic", "roic", True),
    ("p_fcf_yield", "fcf_yield", True),
    ("p_gross_margin", "gross_margin", True),
    ("p_leverage", "net_debt_ebitda", False),
    ("p_earnings_stability", "earnings_stability", True),
    ("p_rs", "rs_blend", True),
    ("p_trend", "trend", True),
    ("p_liquidity", "adv_usd", True),
    ("p_lowvol", "vol_1y", False),
]

ALL_PERCENTILES = [k for k, _, _ in RANK_SPEC] + ["p_value"]


def _value_metric(row: dict) -> float | None:
    """Composite cheapness: average of earnings yield and EBITDA/EV where both exist.

    Yields rather than multiples, so the measure stays finite and correctly ordered
    when earnings are zero or negative. v2's ``clamp01((pe - 20) / 20)`` was one-sided
    — it could penalise expensive names but never reward cheap ones, which made the
    terminal's own "CHEAP" tier unreachable.
    """
    parts = [v for v in (row.get("earnings_yield"), row.get("ebitda_yield")) if v is not None]
    if not parts:
        return None
    return sum(parts) / len(parts)


def _rank_within(rows: list[dict], indices: list[int], key: str,
                 field: str, higher_is_better: bool) -> None:
    values = []
    for i in indices:
        v = rows[i].get(field)
        if v is None:
            values.append(None)
        else:
            values.append(v if higher_is_better else -v)
    for i, pct in zip(indices, percentile_rank(values)):
        rows[i][key] = pct


def prepare(rows: list[dict]) -> list[dict]:
    """Attach percentile ranks and data confidence to every scoreable row.

    ETFs are passed through untouched — an index fund has no ROIC, and scoring one
    on quality would be a category error rather than a data gap.
    """
    scoreable = [i for i, r in enumerate(rows) if r.get("asset_class") != "ETF"]
    if not scoreable:
        return rows

    for r in rows:
        r["value_metric"] = _value_metric(r)

    by_sector: dict[str, list[int]] = {}
    for i in scoreable:
        by_sector.setdefault(rows[i].get("sector") or "", []).append(i)
    # Sectors too small to rank within fall back to the whole universe: a percentile
    # drawn from three peers is noise dressed as a measurement.
    small = [i for sec, idx in by_sector.items()
             if len(idx) < MIN_SECTOR_GROUP or not sec for i in idx]
    groups = {sec: idx for sec, idx in by_sector.items()
              if len(idx) >= MIN_SECTOR_GROUP and sec}
    if small:
        groups["__universe__"] = small

    for key, field, higher in RANK_SPEC + [("p_value", "value_metric", True)]:
        if key in SECTOR_RELATIVE:
            for idx in groups.values():
                _rank_within(rows, idx, key, field, higher)
        else:
            _rank_within(rows, scoreable, key, field, higher)

    for i in scoreable:
        r = rows[i]
        observed = [k for k in ALL_PERCENTILES if r.get(k) is not None]
        r["data_confidence"] = round(len(observed) / len(ALL_PERCENTILES), 3)
        r["imputed"] = [k for k in ALL_PERCENTILES if r.get(k) is None]
        # Substitute the group median for anything unobservable, so a missing input
        # neither fabricates a signal nor silently reads as a confident AVOID — the
        # precise failure that made every name in v2 look like a sell.
        for k in r["imputed"]:
            r[k] = NEUTRAL
    return rows


def score_rows(rows: list[dict], weights: dict | None = None) -> list[dict]:
    """prepare() + score() over a universe, writing results back onto each row."""
    prepare(rows)
    for r in rows:
        if r.get("asset_class") == "ETF":
            r["conviction"] = None
            r["signal"] = "BENCHMARK"
            continue
        r.update(score(r, weights))
    return rows


def dispersion(rows: list[dict]) -> float:
    """Population stdev of conviction — the single number that would have caught v2.

    Under v2 every name scored 0 or 1, so dispersion was ~0.2. Any healthy
    cross-section produces double digits.
    """
    vals = [r["conviction"] for r in rows if r.get("conviction") is not None]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
