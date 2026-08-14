#!/usr/bin/env python3
"""Measure the trade in volatility units, and separate where a name is on its path.

Two problems with a fixed rule applied across names, both of which show up the moment you
put two real names side by side:

**A fixed holding period is a different bet in every name.** Twenty-five sessions gives a
3.1%-a-day name a 15.6% travel budget and a 2.1%-a-day name 10.7%. The calendar is
constant; what the position can actually do in it is not. The fix is to stop quoting the
trade in days and percent and quote it in sigma: a stop at a x sigma_H and a target at
b x sigma_H, where sigma_H = sigma_daily x sqrt(H). Then reward-to-risk is b/a for every
name, and the probability of hitting either is a property of the *structure* rather than
of the ticker.

**"Down 30% from the high" hides where the name is on the path.** A name 28% below its
high that has already retraced 39% of the fall is in a different state from one 44% below
its high sitting one percent off its low, and a drawdown screen scores them alike. The
retracement fraction is the missing coordinate, and this script asks whether it pays.

The stop analysis is analytic where it can be. For a driftless walk with a stop at
-a sigma and a target at +b sigma, the probability of touching the stop first is
b/(a+b) — so the structure alone fixes the strike rate, and any edge has to come from
drift the screen supplies. Reported alongside the realised numbers so the two can be
compared rather than conflated.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY = os.path.join(ROOT, "ledger", "history")
BENCH = "SPY"
WARMUP = 252


def rsi_series(c, w=14):
    out = [None] * len(c)
    if len(c) < w + 1:
        return out
    g = ls = 0.0
    for i in range(1, w + 1):
        ch = c[i] - c[i - 1]
        g += max(ch, 0.0)
        ls += max(-ch, 0.0)
    ag, al = g / w, ls / w
    out[w] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(w + 1, len(c)):
        ch = c[i] - c[i - 1]
        ag = (ag * (w - 1) + max(ch, 0.0)) / w
        al = (al * (w - 1) + max(-ch, 0.0)) / w
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def sigma_series(c, w=60):
    """Trailing daily sigma of log returns — the unit everything else is quoted in."""
    out = [None] * len(c)
    r = [0.0] * len(c)
    for i in range(1, len(c)):
        r[i] = math.log(c[i] / c[i - 1]) if c[i - 1] > 0 else 0.0
    s = ss = 0.0
    for i in range(1, len(c)):
        s += r[i]
        ss += r[i] * r[i]
        if i > w:
            s -= r[i - w]
            ss -= r[i - w] ** 2
        if i >= w:
            m = s / w
            out[i] = math.sqrt(max(ss / w - m * m, 0.0))
    return out


def load(limit=None):
    out = {}
    for fn in sorted(os.listdir(HISTORY)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(HISTORY, fn)) as fh:
            d = json.load(fh)
        if len(d.get("close") or []) >= WARMUP + 60:
            out[fn[:-5]] = d
        if limit and len(out) >= limit and BENCH in out:
            break
    return out


def block_ci(rows, key, block_days=36, draws=3000, seed=9):
    import datetime as dt
    by = {}
    for r in rows:
        d = r["date"]
        b = dt.date(int(d[:4]), int(d[5:7]), int(d[8:10])).toordinal() // block_days
        by.setdefault(b, []).append(r[key])
    blocks = [v for _, v in sorted(by.items())]
    if len(blocks) < 15:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        pool = [v for _ in blocks for v in rng.choice(blocks)]
        means.append(sum(pool) / len(pool))
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    series = load(args.limit)
    bench = series.pop(BENCH)
    bidx = {d: i for i, d in enumerate(bench["dates"])}
    bc = bench["close"]
    H = args.hold

    events = []
    for sym, d in series.items():
        c, h, l, dates = d["close"], d["high"], d["low"], d["dates"]
        n = len(c)
        rsi = rsi_series(c)
        sig = sigma_series(c)
        nxt = WARMUP
        for i in range(WARMUP, n - H):
            if i < nxt or sig[i] is None or sig[i] <= 0:
                continue
            win_h = h[i - 251:i + 1]
            hi = max(win_h)
            peak = i - 251 + max(range(len(win_h)), key=lambda k: win_h[k])
            leg_low = min(l[peak:i + 1])
            dd = (hi - c[i]) / hi
            if dd < 0.15 or hi <= leg_low:
                continue
            retr = (c[i] - leg_low) / (hi - leg_low)
            if c[i - 252] <= 0:
                continue
            mom = c[i - 21] / c[i - 252] - 1.0
            bi = bidx.get(dates[i])
            if bi is None or bi + H >= len(bc):
                continue
            nxt = i + H
            sH = sig[i] * math.sqrt(H)
            fwd = c[i + H] / c[i] - 1.0
            mkt = bc[bi + H] / bc[bi] - 1.0
            events.append({
                "date": dates[i], "sym": sym, "dd": dd, "retr": retr, "mom": mom,
                "sigH": sH,
                "ex": fwd - mkt,
                # the same return expressed in the name's own volatility units, which is
                # the only way a utility and a semiconductor can be averaged together
                "z": (fwd - mkt) / sH,
                "mae": (min(l[i + 1:i + H + 1]) / c[i] - 1.0) / sH,
                "mfe": (max(h[i + 1:i + H + 1]) / c[i] - 1.0) / sH,
            })

    print(f"panel {len(series)} names   hold {H} sessions   events {len(events)}")
    print(f"all events are >=15% below the 52-week high\n")

    print("RETRACEMENT — how far the name has already come back off its low")
    print(f"{'retraced':<16}{'n':>6}{'med sigma_H':>13}{'excess %':>10}{'excess in sigma':>17}{'95% CI (sigma)':>20}{'win%':>7}")
    bands = ((0.00, 0.10, "0-10% (at lows)"), (0.10, 0.25, "10-25%"),
             (0.25, 0.50, "25-50%"), (0.50, 1.01, "50%+ (recovered)"))
    for lo, hi_, label in bands:
        sel = [e for e in events if lo <= e["retr"] < hi_]
        if len(sel) < 30:
            continue
        ci = block_ci(sel, "z")
        band = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "n/a"
        print(f"{label:<16}{len(sel):>6}{st.median([e['sigH'] for e in sel]):>12.1%}"
              f"{st.mean([e['ex'] for e in sel]):>+10.2%}{st.mean([e['z'] for e in sel]):>+17.3f}"
              f"{band:>20}{sum(1 for e in sel if e['ex']>0)/len(sel):>6.0%}")

    print("\nRETRACEMENT x 12-MONTH MOMENTUM  (excess in sigma units)")
    print(f"{'retraced':<18}{'mom < -10%':>14}{'mom -10..+10%':>16}{'mom > +10%':>14}")
    for lo, hi_, label in bands:
        row = f"{label:<18}"
        for mlo, mhi in ((-9, -0.10), (-0.10, 0.10), (0.10, 9)):
            sel = [e for e in events if lo <= e["retr"] < hi_ and mlo <= e["mom"] < mhi]
            row += (f"{st.mean([e['z'] for e in sel]):>+9.3f} ({len(sel):>3})"
                    if len(sel) >= 30 else f"{'-':>14}")
        print(row)

    print("\nSTOP PLACEMENT — in sigma_H units, what a stop actually costs you")
    print("  Under a driftless walk a stop at -a and a target at +b is touched first with")
    print("  probability b/(a+b). Realised numbers beside the theory:")
    print(f"{'stop at':<12}{'theory P(stop) vs +1s target':>32}{'realised touched':>19}{'median MAE':>13}")
    mae = [e["mae"] for e in events]
    for a in (0.5, 0.75, 1.0, 1.5, 2.0):
        theory = 1.0 / (a + 1.0)
        hit = sum(1 for m in mae if m <= -a) / len(mae)
        print(f"{-a:>+8.2f}s{theory:>30.0%}{hit:>19.0%}{st.median(mae):>13.2f}s")
    print(f"\n  A 3xATR stop sits near -1.0 sigma over a 25-session hold, which is why it")
    print(f"  was being touched about a third of the time before anything resolved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
