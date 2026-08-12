"""Model health and conviction persistence, read from the snapshots already recorded.

**No schema extension was needed.** Every snapshot already carries per-symbol conviction,
price and weight on a dated file, which is the whole input for cohort tracking. Adding a
"cohort" field to the write path would have stored a derived value that can be recomputed
exactly from what is there — and stored values drift from their definitions the moment the
definition changes, while a recomputation cannot.

Two measurements, both descriptions of what the model has done rather than forecasts:

* **Stickiness** — share of one session's leading cohort still in the next session's.
  This is the churn a holder of that book would have paid for. Rank correlation across
  the whole universe does not measure it: a board can reorder its tail violently while
  the leaders sit still, and score a low correlation for movement nobody would trade.

* **Persistence** — how much of a name's recorded life it has spent backed, and its
  longest consecutive run.

Two things this file is careful about, both learned from the crypto sibling:

* **Conviction is stored as a rounded integer**, and this universe has 1,016 names with a
  maximum of 87 and only eleven at 80 or above. A top-ten cohort therefore sits exactly
  on a wall of ties, and its membership would flicker on tie-breaking order rather than
  on anything the model did. The cohort is a decile, and ties break deterministically by
  symbol so the same board always yields the same cohort.

* **Every figure carries its window.** "Persistence 88.4% (30D)" is the badge a terminal
  naturally writes; with five snapshots on file it would be a fabrication. The window is
  computed and reported, never assumed.
"""
from __future__ import annotations

import json
import os

from . import performance, snapshots

LEDGER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ledger")

# The leading cohort, as a share of the scored universe. A decile rather than a fixed
# ten: conviction is a rounded integer over ~1,000 names, so the very top is a wall of
# ties and a ten-name cohort would churn on sort order rather than on the model.
COHORT_SHARE = 0.10
COHORT_MIN = 20
# Conviction at or above this counts as the model backing a name. Matches the terminal's
# BUY band; kept as its own constant so changing the reporting line is deliberate rather
# than a silent consequence of a scoring edit.
BACKED_LEVEL = 70.0
# Stated convention, not a historical norm — there is no history to average yet, and a
# badge graded against "normal for this model" would invent the baseline it claims to
# measure against.
STICKY_WARN = 0.70
SPIKE_GAP = 15.0
MAX_NAMES = 40
# Nights added to the ranking denominator. A name seen once and backed once has a raw
# share of 1.0 and would outrank one backed on nine sessions of eleven, which is the
# opposite of what persistent means.
SHRINK = 2.0


def _cohort(convs: dict, size: int) -> set:
    """The top `size` symbols, ties broken by symbol so the result is reproducible."""
    ranked = sorted(convs.items(), key=lambda kv: (-kv[1], kv[0]))
    return {s for s, _ in ranked[:size]}


def _by_session(ledger_dir: str) -> tuple[dict, list, int]:
    """{date: {symbol: conviction}}, plus the axis, oldest first, and how many boards
    carry a recorded session.

    The axis is session dates only when **every** board has one, and filenames
    otherwise. Mixing the two silently loses boards: the 2026-08-11 snapshot records
    session 2026-08-10, which collides with the 2026-08-10 snapshot's filename, and one
    overwrites the other in the dict — five boards on disk became four on the axis with
    nothing reporting a problem. An axis that is uniformly a day late is worse than a
    session axis but is at least uniform, and `dates_are_sessions` says which it is.
    """
    boards = []
    for stamp in snapshots.available(ledger_dir):
        try:
            snap = snapshots.read(snapshots.snapshot_path(ledger_dir, stamp))
        except Exception:  # noqa: BLE001
            continue
        convs = {sym: float(r["conviction"]) for sym, r in snap["rows"].items()
                 if r.get("conviction") is not None}
        if convs:
            boards.append((stamp, snap.get("session_date"), convs))
    dated = sum(1 for _, s, _ in boards if s)
    use_sessions = bool(boards) and dated == len(boards)
    out = {(s if use_sessions else stamp): convs for stamp, s, convs in boards}
    return out, sorted(out), dated


def stickiness(ledger_dir: str = LEDGER) -> dict:
    """Retention of the leading cohort from one recorded session to the next."""
    by_session, sessions, dated = _by_session(ledger_dir)
    # Only pairs the return curve considers a real holding period. A weekend snapshot
    # pair is the same board twice; counting it as 100% sticky would flatter the number
    # for a day on which the market never opened.
    real = {(l["from"], l["to"]) for l in performance.legs(ledger_dir) if l.get("usable")}
    pairs, retained, latest = 0, 0.0, None
    for a, b in zip(sessions, sessions[1:]):
        prev, curr = by_session[a], by_session[b]
        if real and (a, b) not in real:
            continue
        size = max(COHORT_MIN, int(round(min(len(prev), len(curr)) * COHORT_SHARE)))
        if min(len(prev), len(curr)) < size:
            continue
        tp, tc = _cohort(prev, size), _cohort(curr, size)
        held = len(tp & tc) / size
        retained += held
        pairs += 1
        latest = {"from": a, "to": b, "retained": len(tp & tc), "of": size,
                  "entered": sorted(tc - tp)[:12], "left": sorted(tp - tc)[:12]}
    return {
        "stickiness": round(retained / pairs, 4) if pairs else None,
        "pairs": pairs, "sessions": len(sessions), "sticky_warn": STICKY_WARN,
        "cohort_share": COHORT_SHARE, "cohort_min": COHORT_MIN,
        "latest": latest,
        "sessions_dated": dated, "dates_are_sessions": dated == len(sessions) and dated > 0,
        "basis": (f"Share of the top {COHORT_SHARE:.0%} by conviction retained from one "
                  "recorded session to the next, over the pairs the return curve counts "
                  "as real holding periods. A decile rather than a fixed ten because "
                  "conviction is a rounded integer over ~1,000 names, so the very top is "
                  "a wall of ties whose membership would flicker on sort order. No "
                  "comparison to a historical norm is drawn: this window is too short to "
                  "be one."),
    }


def persistence(ledger_dir: str = LEDGER) -> dict:
    """Which names hold conviction across sessions, and which clear the bar once."""
    by_session, sessions, _ = _by_session(ledger_dir)
    series: dict[str, dict] = {}
    for d in sessions:
        for sym, c in by_session[d].items():
            series.setdefault(sym, {})[d] = c
    rows = []
    for sym, seen in series.items():
        convs = [seen[d] for d in sessions if d in seen]
        if not convs:
            continue
        mean = sum(convs) / len(convs)
        sd = (sum((c - mean) ** 2 for c in convs) / (len(convs) - 1)) ** 0.5 if len(convs) > 1 else 0.0
        best = run = 0
        for d in sessions:
            c = seen.get(d)
            run = run + 1 if (c is not None and c >= BACKED_LEVEL) else 0
            best = max(best, run)
        above = sum(1 for c in convs if c >= BACKED_LEVEL)
        peak = max(convs)
        rows.append({
            "symbol": sym, "nights": len(convs), "of": len(sessions),
            "mean": round(mean, 1), "sd": round(sd, 1),
            "peak": round(peak, 1), "latest": round(convs[-1], 1),
            "nights_above": above,
            "share_above": round(above / len(convs), 3),
            "persistence": round(above / (len(convs) + SHRINK), 3),
            "best_streak": best, "current_streak": run,
            "spike": peak >= BACKED_LEVEL and (peak - mean) >= SPIKE_GAP
                     and above <= max(1, len(convs) // 4),
            # None where the name was not on that board — a different cell from a low
            # score, and colouring them alike would invent a reading never taken.
            "cells": [round(seen[d], 1) if d in seen else None for d in sessions],
        })
    rows.sort(key=lambda r: (-r["persistence"], -r["best_streak"], -r["mean"]))
    return {
        "dates": sessions, "level": BACKED_LEVEL, "window": len(sessions),
        "shrink": SHRINK, "rows": rows[:MAX_NAMES],
        "n_backed": sum(1 for r in rows if r["nights_above"] > 0),
        "n_spikes": sum(1 for r in rows if r["spike"]),
        "basis": (f"Share of a name's recorded sessions at conviction {BACKED_LEVEL:.0f} "
                  "or above, with its longest consecutive run beside it. A description "
                  "of what the score has done, not a forecast — a name can be perfectly "
                  "persistent and still be wrong. Ranking uses a shrunk share so a name "
                  "seen once cannot outrank one backed across the window."),
    }


def flips(ledger_dir: str = LEDGER) -> dict:
    """Names crossing the backed line between the last two recorded sessions.

    Symbols travel with the counts: a delta counter exists to get you to the names, and
    a count you cannot act on is a decoration.
    """
    by_session, sessions, _ = _by_session(ledger_dir)
    if len(sessions) < 2:
        return {"pending": True, "into_backed": [], "out_of_backed": []}
    prev, curr = by_session[sessions[-2]], by_session[sessions[-1]]
    shared = set(prev) & set(curr)
    into = sorted(s for s in shared if prev[s] < BACKED_LEVEL <= curr[s])
    out = sorted(s for s in shared if curr[s] < BACKED_LEVEL <= prev[s])
    return {"pending": False, "level": BACKED_LEVEL,
            "from": sessions[-2], "to": sessions[-1],
            "into_backed": into[:25], "out_of_backed": out[:25],
            "n_into": len(into), "n_out": len(out)}


def build(ledger_dir: str = LEDGER) -> dict:
    return {"stickiness": stickiness(ledger_dir),
            "persistence": persistence(ledger_dir),
            "flips": flips(ledger_dir)}


def write(ledger_dir: str = LEDGER) -> str:
    path = os.path.join(ledger_dir, "health.json")
    with open(path, "w") as fh:
        json.dump(build(ledger_dir), fh, separators=(",", ":"))
    return path
