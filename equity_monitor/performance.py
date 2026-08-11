"""Paper return of the published book, chained across recorded snapshots.

Every other panel in this terminal describes the board *as it stands*. This one is the
first that describes what the board would have *earned*, which is a different and much
easier claim to get wrong. Three rules keep it honest:

**Only recorded dates.** A leg exists between two consecutive snapshots and nowhere
else. The series cannot be back-filled, because the weights it needs are the weights
that were actually published on the earlier night, and those were not recorded before
the ledger started. `rebuild.py` can reconstruct a *board*; it cannot reconstruct what
the book was holding, and a curve stitched from reconstructed weights would be a
backtest wearing a track record's clothes.

**Weights from the earlier night, prices from both.** The return of a leg is
``sum(w_i(t) * (p_i(t+1)/p_i(t) - 1))`` over the names held at ``t``. Using ``w(t+1)``
would be buying at the close and settling before it — the single most common way a
paper curve invents alpha it never had.

**The benchmark is read from the snapshot, not from the price cache.** Snapshots record
the index close on the night they were taken. The cache lags — it held 2026-08-05 on
the night of the 2026-08-08 run — and substituting a nearby close for a missing one
compares the book against a different holding period than the one it ran. A leg with no
recorded benchmark is reported as a gap, not filled.

What this is *not*: evidence that the model works. A handful of legs is a handful of
coin flips, and a curve is a far weaker instrument than the Information Coefficient the
snapshot series is being accumulated for. The rendering threshold below stops the chart
drawing a straight line through two points; it is not the point at which the number
starts meaning something, and the terminal says so beside the chart.
"""
from __future__ import annotations

import json
import os

from . import snapshots

LEDGER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ledger")

# Snapshot days required before the curve is drawn at all. Below this the chart would
# be one or two segments — a shape the eye reads as a trend and which is nothing of the
# kind. Chosen for legibility, not significance.
MIN_DAYS = 5

# A leg that loses more than this share of the book's weight to unpriceable or departed
# names is not a return, it is a data outage wearing one.
MAX_WEIGHT_LOSS = 0.10


def _book_leg(prev_rows: dict, curr_rows: dict) -> dict | None:
    """One night's return for the book as it was published on the earlier night."""
    num = held = missing = 0.0
    names = 0
    for sym, row in prev_rows.items():
        w = row.get("weight") or 0.0
        if w <= 0:
            continue
        held += w
        p0 = row.get("price")
        p1 = (curr_rows.get(sym) or {}).get("price")
        if not p0 or not p1 or p0 <= 0:
            # Left the universe, or unpriceable tonight. Dropping it renormalises the
            # rest rather than scoring it a silent zero.
            missing += w
            continue
        num += w * (p1 / p0 - 1.0)
        names += 1
    kept = held - missing
    if held <= 0 or kept <= 0:
        return None
    return {
        "return": num / kept,
        "names": names,
        "weight_lost": missing / held,
        "usable": (missing / held) <= MAX_WEIGHT_LOSS,
    }


MIN_NAMES_FOR_STALE = 5


def _nothing_traded(prev_rows: dict, curr_rows: dict) -> bool:
    """True when every shared name carries the identical price on both dates.

    A snapshot written when the market was shut holds the previous session's closes, so
    the pair is not a holding period — it is the same board twice. One or two unchanged
    names is an ordinary quiet day; a thousand unchanged to the cent is a calendar
    artifact, and chaining it into the curve adds a 0.00% leg that dilutes the series
    and inflates the day count the render gate reads.

    Detected from the data rather than from a trading-day calendar on purpose: a
    calendar has to be maintained, knows nothing about holidays in other venues, and
    would not have caught the actual cause here, which was a push-triggered workflow
    re-running the build on a Saturday.
    """
    shared = 0
    for sym, row in prev_rows.items():
        p0, p1 = row.get("price"), (curr_rows.get(sym) or {}).get("price")
        if not p0 or not p1 or p0 <= 0:
            continue
        shared += 1
        if p0 != p1:
            return False
    return shared >= MIN_NAMES_FOR_STALE


def _equal_weight_leg(prev_rows: dict, curr_rows: dict) -> float | None:
    """The same window, every scored name weighted equally.

    The control the book is actually trying to beat. Outperforming the index can be a
    sector or beta tilt; outperforming the universe the names were selected *from* is
    the claim the model is making, and it costs one extra loop to compute.
    """
    rets = []
    for sym, row in prev_rows.items():
        p0 = row.get("price")
        p1 = (curr_rows.get(sym) or {}).get("price")
        if p0 and p1 and p0 > 0:
            rets.append(p1 / p0 - 1.0)
    return sum(rets) / len(rets) if rets else None


def _session_of(snap: dict, filename_date: str) -> tuple:
    """The trading day a snapshot's closes belong to, and whether that was recorded.

    Snapshots written before `session_date` existed have no answer, and the run date is
    the only thing available. That is a fallback, not an equivalent — it is the very
    mislabelling this field was added to fix — so the caller is told which it got and
    the curve discloses it rather than presenting both kinds of date as one series.
    """
    s = snap.get("session_date")
    return (s, True) if s else (filename_date, False)


def legs(ledger_dir: str = LEDGER) -> list[dict]:
    """One entry per consecutive pair of recorded snapshots, oldest first.

    Legs are dated by the *session* the closes came from, not by the filename. A run at
    23:00 UTC on a Friday can still be reading Thursday's bar, so 2026-08-07 held
    Thursday's closes while 2026-08-08 held Friday's — a real Thursday-to-Friday move
    that the filenames labelled Friday-to-Saturday. Every date on the published curve
    was one day late, and the one weekend leg that genuinely had no trading in it was
    indistinguishable from the rest.
    """
    dates = snapshots.available(ledger_dir)
    out = []
    for a, b in zip(dates, dates[1:]):
        prev = snapshots.read(snapshots.snapshot_path(ledger_dir, a))
        curr = snapshots.read(snapshots.snapshot_path(ledger_dir, b))
        book = _book_leg(prev["rows"], curr["rows"])
        if not book:
            continue
        p0 = ((prev.get("benchmark") or {}).get("price"))
        p1 = ((curr.get("benchmark") or {}).get("price"))
        bench = (p1 / p0 - 1.0) if (p0 and p1 and p0 > 0) else None
        s_from, dated_from = _session_of(prev, a)
        s_to, dated_to = _session_of(curr, b)
        # Two snapshots reading the same session are the same board twice, whatever
        # their filenames say. This is the direct statement of the weekend case, and it
        # is stronger than comparing prices: it holds even if a vendor revises a close
        # between runs, which would make the boards differ while the session did not.
        stale = (s_from == s_to and dated_from and dated_to) \
            or _nothing_traded(prev["rows"], curr["rows"])
        out.append({
            "from": s_from, "to": s_to,
            "session_dated": dated_from and dated_to,
            "recorded_from": a, "recorded_to": b,
            "book": book["return"],
            "benchmark": bench,
            "equal_weight": _equal_weight_leg(prev["rows"], curr["rows"]),
            "names": book["names"],
            "weight_lost": book["weight_lost"],
            "stale": stale,
            "usable": book["usable"] and not stale,
            "spec_hash": curr.get("spec_hash"),
            "spec_from": prev.get("spec_hash"),
            "spec_changed": prev.get("spec_hash") != curr.get("spec_hash"),
        })
    return out


def build(ledger_dir: str = LEDGER) -> dict:
    """The cumulative series plus everything needed to render it honestly."""
    dates = snapshots.available(ledger_dir)
    ls = legs(ledger_dir)
    usable = [l for l in ls if l["usable"]]

    series, book, bench, eq = [], 1.0, 1.0, 1.0
    bench_live = False
    # The origin is where measurement starts — the first *usable* leg's earlier date,
    # not the first date on file. Anchoring at dates[0] when the opening legs were
    # dropped draws a segment spanning the gap and attributes one night's return to all
    # of it. Latent here (no leg has been dropped yet) and immediate on the crypto
    # ledger, where the first three legs lose too much of the book to be returns.
    if usable:
        series.append({"date": usable[0]["from"], "book": 0.0,
                       "benchmark": 0.0, "equal_weight": 0.0})
    for l in usable:
        book *= (1.0 + l["book"])
        if l["benchmark"] is not None:
            bench *= (1.0 + l["benchmark"])
            bench_live = True
        if l["equal_weight"] is not None:
            eq *= (1.0 + l["equal_weight"])
        series.append({
            "date": l["to"],
            "book": round((book - 1.0) * 100, 4),
            # None rather than a carried-forward value: a flat segment would read as
            # "the index did not move", which is a claim, where the truth is "the index
            # was not recorded that night".
            "benchmark": round((bench - 1.0) * 100, 4) if l["benchmark"] is not None else None,
            "equal_weight": round((eq - 1.0) * 100, 4) if l["equal_weight"] is not None else None,
        })

    # Both sides of every leg, not just the later one. Collecting the later hash alone
    # made a two-snapshot curve that straddled a specification change report exactly one
    # hash and call itself stable — the one thing this field exists to catch. A curve
    # spanning two specifications is two curves drawn end to end.
    # The origin is only a real point on a line that has real points. Emitting
    # ``benchmark: 0`` on a series where the benchmark was never recorded plants a dot
    # at break-even that no measurement produced — the precise thing the None-rather-
    # than-carry-forward rule above exists to prevent, reintroduced at index zero.
    if series and not bench_live:
        series[0]["benchmark"] = None
    if series and not any(pt["equal_weight"] is not None for pt in series[1:]):
        series[0]["equal_weight"] = None

    hashes = sorted({h for l in usable for h in (l.get("spec_hash"), l.get("spec_from")) if h})
    crossed = any(l["spec_changed"] for l in usable)
    # The dates the specification changed, so a chart can rule a line there rather than
    # relying on a reader to find the caveat in a footnote. A regime change that is only
    # described in prose beside the chart is a regime change most readers will miss.
    boundaries = sorted({l["to"] for l in usable if l["spec_changed"]})
    return {
        "days": len(dates),
        "min_days": MIN_DAYS,
        # The gate the terminal reads. Legs, not dates: a gap in the ledger leaves the
        # dates far apart with nothing measured between them.
        "renderable": len(usable) >= MIN_DAYS - 1,
        "legs": len(usable),
        "legs_dropped": len(ls) - len(usable),
        # Split out because the two causes need different responses: a leg dropped for
        # weight loss is a data-coverage problem, a stale one means the market was shut
        # and there was never a holding period there to measure.
        "legs_dropped_stale": sum(1 for l in ls if l.get("stale")),
        "from": usable[0]["from"] if usable else (dates[0] if dates else None),
        "to": usable[-1]["to"] if usable else (dates[-1] if dates else None),
        "recorded_from": dates[0] if dates else None,
        "recorded_to": dates[-1] if dates else None,
        # Whether the dates above are trading sessions or run dates. Snapshots written
        # before session_date existed can only be dated by their filename, which is the
        # mislabelling this exists to fix — so a curve mixing the two says so instead of
        # presenting one axis as if it were uniformly measured.
        "session_dated_legs": sum(1 for l in usable if l.get("session_dated")),
        "dates_are_sessions": bool(usable) and all(l.get("session_dated") for l in usable),
        "benchmark": "SPY",
        "benchmark_legs": sum(1 for l in usable if l["benchmark"] is not None),
        "benchmark_available": bench_live,
        "book_total": round((book - 1.0) * 100, 4) if usable else None,
        "benchmark_total": round((bench - 1.0) * 100, 4) if bench_live else None,
        "equal_weight_total": round((eq - 1.0) * 100, 4) if usable else None,
        "spec_hashes": hashes,
        "spec_boundaries": boundaries,
        "spec_stable": len(hashes) <= 1 and not crossed,
        "series": series,
        "basis": ("Weights as published on the earlier night, priced at both ends from "
                  "the snapshots themselves. Not back-filled and not reconstructed."),
    }


def write(ledger_dir: str = LEDGER) -> dict:
    payload = build(ledger_dir)
    path = os.path.join(ledger_dir, "performance.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    os.replace(tmp, path)
    return payload
