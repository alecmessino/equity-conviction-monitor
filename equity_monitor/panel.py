"""Panel hygiene: which bars are usable, and which names belong in a study at all.

A backtest harness that globs a directory is trusting whatever a vendor happened to
serve. Four things in this panel were not survivable, and each produced a plausible
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

**Recycled tickers.** A symbol is a slot on an exchange, not an identity. Thirty-five of
the 290 names in the delisted panel carry bars belonging to a *different issuer* that took
the symbol over later: S is Sprint until 2020-04-01 and SentinelOne after it, STI is
SunTrust then a 2024 relisting, CA is CA Inc. then an unrelated 2023 name. Fifteen of the
thirty-five contain no bars from the original company at all — the delisted-cohort study
was reading a successor's entire price history as the dead company's. The price API offers
no way to tell them apart, so the split has to come from the listing registry: bars dated
past a ticker's true last trading day are a different security and are marked unusable.

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

# Slack between a registry endDate and the last bar the original issuer actually printed.
# Settlement and the final tape run a few sessions past the recorded delisting; a
# reissued symbol reappears months or years later, so nothing sits near the boundary.
RECYCLE_GRACE_DAYS = 10

# A follow-on listing opening within a week of the previous one closing is the same company
# continuing — an exchange transfer (PNFP, NASDAQ to NYSE over a New Year) or a relisting
# after reorganisation (BTU, three days). A symbol reissued to an unrelated company sits
# idle for months: S waited 455 days between Sprint and SentinelOne.
CONTINUATION_GAP_DAYS = 7

# A listing episode shorter than this is a registry artifact, not a company's life on an
# exchange. BRKB resolves to a nineteen-day row against a file holding ten years of bars.
MIN_EPISODE_DAYS = 90

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


def episode_flags(dates: list[str], first: str | None, last: str | None) -> list[bool]:
    """False on bars outside [first, last] — those belong to a different holder of the symbol.

    A ticker is a slot on an exchange, not an identity, and both ends of the slot matter.
    Bars *after* `last` are the successor that took the symbol over: S is Sprint through
    2020-04-01 and SentinelOne from 2021-06-30. Bars *before* `first` are the predecessor:
    the company holding MRNA in 2016 stopped trading in October 2018 and Moderna listed
    that December. Guarding only the far end deletes Moderna's entire history as though it
    were the dead company's; guarding only the near end reads SentinelOne as Sprint.

    A None bound is open — the episode has no known edge on that side.
    """
    lo = _as_date(first)
    hi = _as_date(last)
    if hi is not None:
        hi += dt.timedelta(days=RECYCLE_GRACE_DAYS)
    if lo is None and hi is None:
        return [True] * len(dates)
    out = []
    for d in dates:
        v = _as_date(d)
        out.append(v is not None and (lo is None or v >= lo) and (hi is None or v <= hi))
    return out


def _as_date(v):
    try:
        return dt.date.fromisoformat(v)
    except (ValueError, TypeError):
        return None


def episodes(rows: list[dict]) -> dict:
    """Ticker -> its listing episodes as sorted (start, end) pairs, continuations merged.

    Tiingo's registry is one row per (ticker, exchange, listing), so a single company holds
    several: PNFP has a NASDAQ row closing 2025-12-31 and an NYSE row opening two sessions
    later, and BTU relisted three days after its reorganisation. Those are the same company
    continuing and are merged. A symbol reissued to an unrelated company sits idle for
    months first — S waited 455 days between Sprint and SentinelOne — so the gap threshold
    separates the two cases cleanly with nothing near the boundary.

    Overlapping rows are merged too. They are usually one listing recorded twice across
    venues (COHR carries identical 1990-03-26 NYSE and NASDAQ rows), and occasionally a
    predecessor's book backfilled onto the symbol (DOC holds Physicians Realty from 2013
    and Healthpeak's history from 1987). The registry does not distinguish them, so both
    merge — which errs toward keeping data rather than deleting a live company's history.
    """
    by: dict[str, list] = {}
    for r in rows:
        t, s_, e = r.get("ticker"), r.get("startDate") or "", r.get("endDate") or ""
        if t and s_:
            by.setdefault(t, []).append((s_, e))
    out = {}
    for t, eps in by.items():
        eps.sort()
        merged = [list(eps[0])]
        for s_, e in eps[1:]:
            prev = merged[-1]
            pe, ps = _as_date(prev[1]), _as_date(s_)
            gap = (ps - pe).days if (pe and ps) else 0
            if not prev[1] or gap <= CONTINUATION_GAP_DAYS:
                if not e or (prev[1] and e > prev[1]):
                    prev[1] = e
            else:
                merged.append([s_, e])
        out[t] = [(a, b or None) for a, b in merged]
    return out


def resolve_episode(eps: list, dates: list[str]) -> tuple:
    """Which listing episode the bars on disk belong to, and how far to trust the answer.

    Returns ``(first, last, status)``. Status is the point of the function: a guard that
    silently truncates whenever the registry disagrees with the file does more damage than
    the contamination it removes. Measured on this panel, matching each file to the
    episode covering its own first bar resolves 1,317 of 1,349 names outright; of the
    thirty-two that straddle a boundary, eleven hold nothing but a successor's history.

        clean       every bar falls inside one episode — nothing to do
        truncated   the file spans a boundary; the bars outside it are a different company
        recycled    no bar falls inside the episode at all; the file is entirely successor
        unresolved  registry and file disagree structurally — bounds are left open

    The unresolved class exists because the registry is not clean either. BRKB resolves to
    a nineteen-day row that is plainly an artifact of the dashed-symbol convention, against
    a file holding ten years of Berkshire. Truncating to the row would delete the name;
    reporting it as unresolved keeps the bars and keeps the disagreement visible.
    """
    if not eps or not dates:
        return (None, None, "unresolved")
    # Bind to whichever episode holds the most of the file. Anchoring on the first bar
    # instead looks reasonable and is wrong on every symbol the vendor backfills: AA's
    # file opens with two months of Alcoa Inc. before Alcoa Corp listed, and matching on
    # that opening bar keeps the two months and discards ten years of the company the file
    # is actually about. Counting settles it without a rule about which end to believe.
    best, first, last = -1, None, None
    for a, b in sorted(eps):
        span = ((_as_date(b) or dt.date.today()) - (_as_date(a) or dt.date.min)).days
        if span < MIN_EPISODE_DAYS:
            continue
        n = sum(episode_flags(dates, a, b))
        if n >= best:                       # ties go to the later episode
            best, first, last = n, a, b
    if first is None or best <= 0:
        return (None, None, "unresolved") if first is None else (first, last, "recycled")
    if best == len(dates):
        return (first, last, "clean")
    return (first, last, "truncated")


def episode_covering(eps: list, date: str) -> tuple:
    """The (start, end) episode containing `date`, or (None, None) if none does."""
    for s_, e in eps or ():
        if s_ <= date and (e is None or date <= e):
            return (s_, e)
    return (None, None)


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
