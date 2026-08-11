"""Does conviction predict the next session's return?

This is the only question that decides whether the score is worth acting on, and it is a
different question from "is the book beating the index". The distinction is why this
module exists at all, and the crypto sibling is where it first mattered: that basket is
losing to its equal-weight control by about 283bp, which reads as proof the selection
subtracts value. The measurement does not support that reading — the information
coefficient over the same legs is +0.006 with a 95% interval of roughly [-0.09, +0.10].

A concentrated book with no *measurable* edge underperforms an equal-weight control as a
matter of course, because concentration adds variance without adding expected return.
The gap is the expected consequence of acting on an unmeasured signal, not evidence the
signal is negative. Neither is it evidence the signal is positive. There is simply not
enough history yet, and the honest output says exactly that.

So this reports the interval and the sample size before anything else. A mean IC quoted
alone invites a reader to treat +0.006 as "slightly positive"; only the interval makes
"cannot be distinguished from nothing" visible.

Two rules it inherits from the rest of the project:

* **Legs are session-dated.** A leg spans two trading sessions, taken from the snapshots'
  recorded ``session_date``, not their filenames. Snapshots that predate that field are
  measured on filenames and the fact is disclosed rather than smoothed over.

* **Specification boundaries segment the series.** An IC averaged across two scoring
  functions is a number about a model that never existed.
"""
from __future__ import annotations

import json
import os

from . import performance, snapshots

LEDGER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ledger")

# Below this a rank correlation over one session is noise with a decimal point.
MIN_NAMES = 20
QUINTILE = 5
# Legs needed before the mean IC is worth reading. Set so the standard error of the mean
# is at most about half a plausible true signal.
MIN_LEGS = 40
# Targets used to state how much history is still required. 0.03 is a respectable
# cross-sectional signal; 0.05 would be a strong one.
TARGET_ICS = (0.02, 0.03, 0.05)


def _spearman(a: list, b: list) -> float | None:
    """Rank correlation, ties averaged. None below three pairs."""
    n = len(a)
    if n < 3:
        return None

    def ranks(xs):
        order = sorted(range(n), key=lambda i: xs[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return None if da == 0 or db == 0 else round(num / (da * db), 6)


def legs(ledger_dir: str = LEDGER) -> list[dict]:
    """Per-leg information coefficient and quintile spread.

    Reuses ``performance.legs`` for the session dating, staleness rule and boundary
    handling rather than reimplementing them: a second definition of "which pairs of
    snapshots are a holding period" is a second thing to keep in step, and this one
    would drift silently because both would still produce plausible numbers.
    """
    ls = {(l["recorded_from"], l["recorded_to"]): l
          for l in performance.legs(ledger_dir) if l.get("usable")}
    out = []
    for (a, b), leg in sorted(ls.items()):
        prev = snapshots.read(snapshots.snapshot_path(ledger_dir, a))
        curr = snapshots.read(snapshots.snapshot_path(ledger_dir, b))
        pairs = []
        for sym, row in prev["rows"].items():
            nxt = curr["rows"].get(sym)
            conv, p0 = row.get("conviction"), row.get("price")
            p1 = (nxt or {}).get("price")
            if conv is None or not p0 or not p1 or p0 <= 0:
                continue
            pairs.append((float(conv), p1 / p0 - 1.0))
        if len(pairs) < MIN_NAMES:
            continue
        rho = _spearman([p[0] for p in pairs], [p[1] for p in pairs])
        ranked = sorted(pairs, key=lambda p: -p[0])
        k = max(3, len(ranked) // QUINTILE)
        top = sum(p[1] for p in ranked[:k]) / k
        bot = sum(p[1] for p in ranked[-k:]) / k
        out.append({"from": leg["from"], "to": leg["to"],
                    "session_dated": leg.get("session_dated", False),
                    "ic": rho, "names": len(pairs),
                    "top_quintile": round(top * 100, 4),
                    "bottom_quintile": round(bot * 100, 4),
                    "spread_bp": round((top - bot) * 1e4, 1)})
    return out


def build(ledger_dir: str = LEDGER) -> dict:
    """The edge measurement, with its interval and what is still missing."""
    ls = legs(ledger_dir)
    perf = performance.build(ledger_dir)
    ics = [l["ic"] for l in ls if l["ic"] is not None]
    spreads = [l["spread_bp"] for l in ls]
    base = {
        "legs": len(ls), "min_legs": MIN_LEGS, "series": ls,
        "spec_hashes": perf.get("spec_hashes"),
        "spec_boundaries": perf.get("spec_boundaries"),
        "dates_are_sessions": perf.get("dates_are_sessions"),
        "book_total": perf.get("book_total"),
        "equal_weight_total": perf.get("equal_weight_total"),
        "benchmark_total": perf.get("benchmark_total"),
        "benchmark": perf.get("benchmark"),
        "basis": ("Information coefficient = rank correlation between a session's "
                  "conviction and the next session's return, across the names scored on "
                  "both. It answers whether the ordering is informative, which is a "
                  "different question from whether the book beat the index — a "
                  "concentrated book with no measurable edge underperforms an "
                  "equal-weight control as a matter of course."),
    }
    if len(ics) < 2:
        return {**base, "measurable": False, "mean_ic": None, "ci": None,
                "verdict": ("Not enough measured legs to estimate anything. The curve "
                            "needs recorded sessions, and they cannot be back-filled.")}

    mean = sum(ics) / len(ics)
    var = sum((i - mean) ** 2 for i in ics) / (len(ics) - 1)
    se = (var / len(ics)) ** 0.5
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    measurable = len(ics) >= MIN_LEGS and (lo > 0 or hi < 0)
    nbar = sum(l["names"] for l in ls) / len(ls)
    per_leg = 1.0 / max(1.0, (nbar - 3)) ** 0.5
    needed = {f"{t:.2f}": int(round((per_leg / (t / 2.0)) ** 2)) for t in TARGET_ICS}
    return {
        **base,
        "measurable": measurable,
        "mean_ic": round(mean, 4),
        "median_ic": round(sorted(ics)[len(ics) // 2], 4),
        "ic_sd": round(var ** 0.5, 4),
        "ic_se": round(se, 4),
        "ci": [round(lo, 4), round(hi, 4)],
        "t_stat": round(mean / se, 3) if se else None,
        "legs_positive": sum(1 for i in ics if i > 0),
        "mean_spread_bp": round(sum(spreads) / len(spreads), 1) if spreads else None,
        "spreads_positive": sum(1 for s in spreads if s > 0),
        "per_leg_noise": round(per_leg, 3),
        "legs_needed": needed,
        "verdict": (
            "Conviction orders the universe informatively."
            if measurable and mean > 0 else
            "Conviction orders the universe backwards — the ranking is inverted."
            if measurable else
            "No measurable relationship between conviction and next-session return. "
            "The interval spans zero, so this is neither evidence the score works nor "
            "evidence it does not — there is simply not enough history yet."),
    }


def write(ledger_dir: str = LEDGER) -> str:
    path = os.path.join(ledger_dir, "edge.json")
    with open(path, "w") as fh:
        json.dump(build(ledger_dir), fh, separators=(",", ":"))
    return path
