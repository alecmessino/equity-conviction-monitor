"""Panel hygiene: which bars are usable, and which names belong in a study at all.

A backtest harness that globs a directory is trusting whatever a vendor happened to
serve. Three things in this panel were not survivable, and each produced a plausible
number rather than an error:

**Unadjusted reorganisations.** CHRD closes at 0.073592 on 2020-11-19 and 19.011288 on
2020-11-20 — a 25,733% single-session move on equity that was cancelled in Chapter 11.
It passes a dip screen on exactly the features the strategy is built from (drawdown 0.970,
RSI 29.6) and contributes a single event of +42,216%, which by itself exceeded half of one
screen's total measured alpha. Four more of the same class are on disk: CHK, STI, QXO,
BMNR.

**Holes.** CHK is missing 1,313 calendar days between 2021-03 and 2024-10 — 1,906 bars
where 2,811 belong. Every 252-bar momentum, drawdown and beta window spanning that hole is
computed across a gap the code cannot see, and a forward window indexed positionally lands
years away from where the calendar says it should.

**Instruments that are not the thing being studied.** Eleven ETFs and eight dual-class
pairs were being screened as though they were ordinary common stock.

The guard is per **bar**, not per name. A per-name filter that drops anything with a large
single session removes MDGL (+268% on phase-3 data), SMMT, VKTX, GME and others whose moves
are entirely real — hindsight-conditioned survivorship in reverse, biasing results down
instead of up. Marking individual bars unusable and excluding only the windows that span
them keeps every genuine observation a name has to offer.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os

# A single session beyond this is not a stock moving; it is a reorganisation, a reverse
# split, or a vendor error. Real single-day moves in this universe top out well below it
# — the largest genuine ones on disk are biotech phase-3 prints in the +250-270% range,
# which sit under the threshold and are kept.
MAX_ABS_SESSION_MOVE = 3.0        # +300% / -75% in one session

# Consecutive bars further apart than this have a hole between them. Ten calendar days
# clears every US market holiday cluster including the Thanksgiving and Christmas weeks.
MAX_BAR_GAP_DAYS = 10

# Screened as common stock and are not. The dual-class pairs are left in — they are real
# equities — but are listed so a study can collapse them if it wants one vote per company.
KNOWN_ETFS = {
    "SPY", "RSP", "IWB", "IWV", "IWM", "VTI", "QQQ", "EEM", "EFA", "EWJ",
    "IWD", "MTUM", "QUAL", "XLE", "XLF", "XLK", "XLV", "VWO", "VEA", "AGG",
    "LQD", "HYG", "TLT", "GLD", "SLV", "XLY", "XLP", "XLI", "XLU", "XLB",
    "XLRE", "XLC", "DIA", "MDY", "IJH", "IJR", "IVV", "VOO", "VUG", "VTV",
}

DUAL_CLASS_PAIRS = [("GOOG", "GOOGL"), ("FOX", "FOXA"), ("NWS", "NWSA"),
                    ("BRKA", "BRKB"), ("BFA", "BFB"), ("LEN", "LENB"),
                    ("HEI", "HEIA"), ("UHAL", "UHALB")]


def bar_flags(d: dict) -> list[bool]:
    """Per-bar usability. False where the bar itself is not trustworthy.

    Flags the bar *after* a suspect move as well as the bar itself, because a level shift
    corrupts the return spanning it in both directions.
    """
    c, dates = d["close"], d.get("dates") or []
    n = len(c)
    ok = [True] * n
    for i in range(1, n):
        if c[i - 1] <= 0 or c[i] <= 0:
            ok[i] = False
            continue
        move = c[i] / c[i - 1] - 1.0
        if move > MAX_ABS_SESSION_MOVE or move < -(1 - 1 / (1 + MAX_ABS_SESSION_MOVE)):
            ok[i] = False
        if i < len(dates) and dates[i] and dates[i - 1]:
            try:
                a = dt.date.fromisoformat(dates[i - 1])
                b = dt.date.fromisoformat(dates[i])
                if (b - a).days > MAX_BAR_GAP_DAYS:
                    ok[i] = False
            except ValueError:
                ok[i] = False
    return ok


def clean_windows(ok: list[bool], lookback: int, hold: int) -> list[bool]:
    """True where a bar can anchor an event: no bad bar in [i-lookback, i+hold].

    Computed as a running count rather than a slice per bar, so it stays O(n) on a
    thirteen-year series.
    """
    n = len(ok)
    pref = [0] * (n + 1)
    for i in range(n):
        pref[i + 1] = pref[i] + (0 if ok[i] else 1)
    out = [False] * n
    for i in range(n):
        lo = max(0, i - lookback)
        hi = min(n - 1, i + hold)
        out[i] = (pref[hi + 1] - pref[lo]) == 0
    return out


def is_etf(sym: str) -> bool:
    return sym.upper() in KNOWN_ETFS


def manifest(history_dir: str, symbols: list[str]) -> dict:
    """Symbol -> (bars, first, last, sha1 of the file).

    Written beside every result. The panel on disk was rewritten continuously while this
    study ran — 968 of 1,077 files inside one hour — and a control statistic moved more
    than a percentage point from roughly sixteen files landing. Without a manifest no
    number here is reproducible, including the ones that look stable.
    """
    out = {}
    for s in symbols:
        p = os.path.join(history_dir, f"{s}.json")
        try:
            raw = open(p, "rb").read()
            d = json.loads(raw)
        except Exception:
            continue
        out[s] = {
            "bars": len(d.get("close") or []),
            "first": (d.get("dates") or [""])[0],
            "last": (d.get("dates") or [""])[-1],
            "sha1": hashlib.sha1(raw).hexdigest()[:12],
        }
    return out
