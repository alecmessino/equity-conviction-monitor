"""Macro regime series from FRED's keyless CSV endpoint.

Replaces the terminal's previous "regime" line, which was
``pos > rows.length * 0.6 ? 'Bullish' : 'Bearish'`` — a statement about the
model's own output dressed up as a statement about the market. These are four
observable series that say something the conviction scores do not.
"""
from __future__ import annotations

from ._http import fetch

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

# FRED's edge silently stalls on browser-impersonating User-Agents (a request with a
# Chrome UA hangs until timeout; the same request identifying itself as this tool
# returns in ~2s). Declare what we are.
_FRED_HEADERS = {
    "User-Agent": "equity-conviction-monitor/3.0 "
                  "(+https://github.com/alecmessino/equity-conviction-monitor)",
    "Accept-Encoding": "gzip, deflate",
}

SERIES = {
    "DGS10": ("10Y Treasury", "%"),
    "T10Y2Y": ("2s10s Spread", "%"),
    "VIXCLS": ("VIX", ""),
    "BAMLH0A0HYM2": ("HY OAS", "%"),
}


def _series(series_id: str, lookback: int = 260) -> list[tuple[str, float]]:
    """Trailing observations as [(date, value)], oldest first, gaps dropped."""
    text = fetch(FRED_CSV.format(sid=series_id), headers=_FRED_HEADERS,
                 throttle_key="fred", min_interval=0.2, retries=2,
                 timeout=25).decode("utf-8", "replace")
    out: list[tuple[str, float]] = []
    for line in text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            out.append((parts[0], float(parts[1])))
        except ValueError:
            continue  # FRED marks missing observations with '.'
    return out[-lookback:]


def load() -> dict:
    """Current level, 1-month change, and a sparkline series for each indicator.

    A failed series is omitted rather than defaulted — the terminal renders what
    is present and says so when something is missing.
    """
    out: dict[str, dict] = {}
    for sid, (label, unit) in SERIES.items():
        try:
            obs = _series(sid)
        except Exception:
            continue
        if not obs:
            continue
        latest_date, latest = obs[-1]
        prior = obs[-22][1] if len(obs) >= 22 else obs[0][1]
        out[sid] = {
            "label": label,
            "unit": unit,
            "value": latest,
            "change_1m": latest - prior,
            "as_of": latest_date,
            "series": [v for _, v in obs[-130:]],
        }
    return out
