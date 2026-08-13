#!/usr/bin/env python3
"""Simulate an actual book, because averaging isolated events is not a return.

Every number produced so far is the mean excess return of a signal *event*. A book cannot
hold every event: it has a finite number of slots, it cannot take a position it has no
cash for, and when more signals fire than slots exist it must choose. Averaging events
implicitly assumes unlimited capital and equal weight in every signal simultaneously,
which flatters any strategy whose signals cluster — and dip signals cluster violently,
because everything dips together.

It also assumes the mean is achievable. It is not: at a 25-session hold the mean event
return is roughly three times the median, because a handful of once-a-decade rebounds
(Wayfair, March 2020, +346%) carry it. A book with N slots and a position cap cannot put
enough weight on those to earn the mean.

So this holds N slots, fills them from whatever signalled that day, exits on the horizon,
and charges a round-trip cost. What comes out is a return series comparable to owning an
index fund, which is the only comparison that decides anything.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from equity_monitor import panel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY = os.path.join(ROOT, "ledger", "history")
ETFS = {"SPY", "RSP", "IWB", "IWV", "IWM", "VTI"}


def rsi_series(c, w=14):
    out = [None] * len(c)
    if len(c) < w + 1:
        return out
    g = l_ = 0.0
    for i in range(1, w + 1):
        ch = c[i] - c[i - 1]
        g += max(ch, 0.0)
        l_ += max(-ch, 0.0)
    ag, al = g / w, l_ / w
    out[w] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(w + 1, len(c)):
        ch = c[i] - c[i - 1]
        ag = (ag * (w - 1) + max(ch, 0.0)) / w
        al = (al * (w - 1) + max(-ch, 0.0)) / w
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def rolling_max(v, w):
    dq, out = deque(), [0.0] * len(v)
    for i, x in enumerate(v):
        while dq and v[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - w:
            dq.popleft()
        out[i] = v[dq[0]]
    return out


SCREENS = {
    # Calibration control: buys anything. A book with enough slots must track the
    # equal-weighted pool. If it does not, the portfolio arithmetic is wrong and no
    # other row in the table means anything.
    "Z control: buy anything":
        lambda f, i: True,
    "A baseline (52w dd + oversold)":
        lambda f, i: f["dd"][i] >= 0.15 and f["rsi"][i] is not None and f["rsi"][i] <= 35,
    "G momentum winner + dip":
        lambda f, i: (f["mom"][i] is not None and f["mom"][i] > 0.10
                      and f["rsi"][i] is not None and f["rsi"][i] <= 35),
    "H momentum loser + dip":
        lambda f, i: (f["mom"][i] is not None and f["mom"][i] < -0.10
                      and f["rsi"][i] is not None and f["rsi"][i] <= 35),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--slots", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--benchmark", default="RSP")
    ap.add_argument("--pick", default="random", choices=("alpha", "random", "deepest", "weakest"),
                    help="how to choose among same-day candidates when signals outnumber "
                         "slots. At a 3-6%% fill rate this choice does most of the work, so "
                         "it must be varied rather than left at whatever was convenient.")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    series = {}
    for fn in sorted(os.listdir(HISTORY)):
        if not fn.endswith(".json"):
            continue
        sym = fn[:-5]
        with open(os.path.join(HISTORY, fn)) as fh:
            d = json.load(fh)
        if panel.is_etf(sym) and sym not in (args.benchmark, "SPY"):
            continue
        if len(d.get("close") or []) >= 320:
            series[sym] = d
    bench = series.get(args.benchmark) or series.get("SPY")
    if not bench:
        sys.exit("no benchmark on disk")
    bpx = dict(zip(bench["dates"], bench["close"]))
    for e in ETFS:
        series.pop(e, None)

    feats = {}
    for sym, d in series.items():
        c, h = d["close"], d["high"]
        hi = rolling_max(h, 252)
        feats[sym] = {
            "dates": d["dates"], "close": c,
            "open": d.get("open") or c,
            # Windows spanning an unadjusted reorg or a hole are unusable. Excluding the
            # window rather than the name keeps the genuine history a name does have.
            "ok": panel.clean_windows(panel.bar_flags(d), 252, 25),
            "rsi": rsi_series(c),
            "dd": [(hi[i] - c[i]) / hi[i] if hi[i] > 0 else 0.0 for i in range(len(c))],
            "mom": [(c[i - 21] / c[i - 252] - 1.0) if i >= 252 and c[i - 252] > 0 else None
                    for i in range(len(c))],
            "idx": {dt_: i for i, dt_ in enumerate(d["dates"])},
        }

    all_dates = sorted({d for f in feats.values() for d in f["dates"] if d in bpx})
    all_dates = [d for d in all_dates if d >= "2014-09-01"]
    cost = args.cost_bps / 10000.0
    import random as _r
    rng = _r.Random(args.seed)

    print(f"panel {len(series)} names   {all_dates[0]} .. {all_dates[-1]}")
    print(f"{args.slots} slots, {args.hold}-session hold, {args.cost_bps:.0f}bp round trip, "
          f"benchmark {args.benchmark}\n")
    print(f"{'strategy':<32}{'CAGR':>8}{'bench':>8}{'excess':>9}{'Sharpe':>8}"
          f"{'maxDD':>8}{'trades':>8}{'fill%':>7}{'invested':>8}")
    print("-" * 88)

    for label, test in SCREENS.items():
        # Cash plus per-slot positions, marked every day. The previous version only
        # touched equity on an exit and appended that stale value to the daily series,
        # so open positions were invisible between entry and exit — which overstates
        # Sharpe and materially understates max drawdown, since the drawdown happens
        # while positions are open.
        cash, slots_open, daily, trades, wanted, filled = 1.0, [], [], 0, 0, 0
        invested, invested_days = [], [0]
        for di, dt_ in enumerate(all_dates):
            still = []
            for p in slots_open:
                f = feats[p["sym"]]
                j = f["idx"].get(dt_)
                if j is None:
                    still.append(p)
                    continue
                p["mark"] = f["close"][j] / p["px"]
                if j - p["i"] >= args.hold:
                    # One round-trip charge per completed trade. Charging (1-cost) on
                    # both entry and exit bills the spread twice, which at ~25 turns per
                    # slot per year is a spurious drag of roughly 5 points annually.
                    cash += p["cap"] * p["mark"] * (1.0 - cost)
                    invested_days[0] += 1
                    trades += 1
                else:
                    still.append(p)
            slots_open = still
            # fill free slots from today's signals
            free = args.slots - len(slots_open)
            if free > 0:
                held = {p["sym"] for p in slots_open}
                cands = []
                for sym, f in feats.items():
                    if sym in held:
                        continue
                    i = f["idx"].get(dt_)
                    if i is None or i < 260 or i >= len(f["close"]) - 1:
                        continue
                    if not f["ok"][i]:
                        continue
                    try:
                        if test(f, i):
                            cands.append((sym, i, f["close"][i], f["dd"][i]))
                    except Exception:
                        pass
                wanted += len(cands)
                if args.pick == "alpha":
                    cands.sort(key=lambda t: t[0])
                elif args.pick == "random":
                    rng.shuffle(cands)
                elif args.pick == "deepest":
                    cands.sort(key=lambda t: -t[3])
                else:
                    cands.sort(key=lambda t: t[3])
                for sym, i, px, _dd in cands[:free]:
                    # Entry at the NEXT open. Filling at the close that generated the
                    # signal uses the same bar as both input and execution price, which
                    # is not a tradeable sequence.
                    f = feats[sym]
                    if i + 1 >= len(f["open"]):
                        continue
                    entry = f["open"][i + 1]
                    if entry <= 0:
                        continue
                    # Size against CURRENT equity, not the initial. Fixing the slot at
                    # 1/N of the starting capital stops the book compounding: positions
                    # stay the same absolute size while gains pile into idle cash, and
                    # the strategy silently de-risks toward a cash return over time.
                    equity_now = cash + sum(q["cap"] * q["mark"] for q in slots_open)
                    stake = min(equity_now / args.slots, cash)
                    if stake <= 0:
                        continue
                    cash -= stake
                    slots_open.append({"sym": sym, "i": i + 1, "px": entry,
                                       "cap": stake, "mark": 1.0})
                    filled += 1
            pos_val = sum(p["cap"] * p["mark"] for p in slots_open)
            equity = cash + pos_val
            if equity > 0:
                invested.append(pos_val / equity)
            if di > 0:
                b = bpx.get(dt_)
                pb = bpx.get(all_dates[di - 1])
                daily.append((dt_, equity, (b / pb - 1.0) if (b and pb) else 0.0))

        if trades < 50:
            print(f"{label:<32}   too few trades ({trades})")
            continue
        eq = [x[1] for x in daily]
        yrs = len(daily) / 252.0
        cagr = eq[-1] ** (1 / yrs) - 1
        bl = 1.0
        for _, _, br in daily:
            bl *= 1 + br
        bcagr = bl ** (1 / yrs) - 1
        rets = [eq[i] / eq[i - 1] - 1 for i in range(1, len(eq))]
        sd = st.pstdev(rets)
        sharpe = (st.mean(rets) / sd * math.sqrt(252)) if sd > 0 else 0.0
        peak, mdd = eq[0], 0.0
        for v in eq:
            peak = max(peak, v)
            mdd = min(mdd, v / peak - 1)
        print(f"{label:<32}{cagr:>+7.1%}{bcagr:>+8.1%}{cagr-bcagr:>+9.1%}{sharpe:>8.2f}"
              f"{mdd:>8.0%}{trades:>8}{filled/max(wanted,1):>6.0%}"
              f"{st.mean(invested) if invested else 0:>8.0%}")

    print(f"\n  fill% is the share of signals the book had room for. A low number means the")
    print(f"  event-average results were reporting trades this book could never have taken.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
