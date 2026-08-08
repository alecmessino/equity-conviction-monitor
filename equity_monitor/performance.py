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


def legs(ledger_dir: str = LEDGER) -> list[dict]:
    """One entry per consecutive pair of recorded snapshots, oldest first."""
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
        out.append({
            "from": a, "to": b,
            "book": book["return"],
            "benchmark": bench,
            "equal_weight": _equal_weight_leg(prev["rows"], curr["rows"]),
            "names": book["names"],
            "weight_lost": book["weight_lost"],
            "usable": book["usable"],
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
    if dates:
        series.append({"date": dates[0], "book": 0.0, "benchmark": 0.0, "equal_weight": 0.0})
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
    return {
        "days": len(dates),
        "min_days": MIN_DAYS,
        # The gate the terminal reads. Legs, not dates: a gap in the ledger leaves the
        # dates far apart with nothing measured between them.
        "renderable": len(usable) >= MIN_DAYS - 1,
        "legs": len(usable),
        "legs_dropped": len(ls) - len(usable),
        "from": dates[0] if dates else None,
        "to": dates[-1] if dates else None,
        "benchmark": "SPY",
        "benchmark_legs": sum(1 for l in usable if l["benchmark"] is not None),
        "benchmark_available": bench_live,
        "book_total": round((book - 1.0) * 100, 4) if usable else None,
        "benchmark_total": round((bench - 1.0) * 100, 4) if bench_live else None,
        "equal_weight_total": round((eq - 1.0) * 100, 4) if usable else None,
        "spec_hashes": hashes,
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
