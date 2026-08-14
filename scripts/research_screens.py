#!/usr/bin/env python3
"""Compare entry screens head to head on one panel, with the exit held constant.

The reward-to-risk sweep established that the exit rule is not what decides this
strategy's outcome. That leaves the entry, and the entry has a problem that is visible in
the literature before it is visible in the data:

**The 52-week drawdown screen is a momentum-loser screen.** Jegadeesh and Titman's result
is that stocks ranked on 3-to-12-month returns *continue* in the same direction over the
following 3 to 12 months — losers keep losing. Short-term reversal is a different effect
living at a one-week-to-one-month horizon, and the two are so easily confused that J&T
themselves skip the most recent month when forming momentum, specifically to keep
reversal out of it.

Selecting on a 52-week drawdown puts a book squarely in momentum losers. Holding it for
5-25 sessions tries to harvest reversal. The selection is on the twelve-month axis and
the harvest is on the one-month axis, so the momentum penalty is paid in full and the
reversal premium is collected on whatever is left. That is a coherent explanation for a
matched random-date control returning -1% to -5.5% against the benchmark in the same
names: the *pool* is the problem, not the timing within it.

So this script does not tune the strategy. It asks which screen the pool should come from
at all, holding the exit fixed at a plain N-session time exit. No stop, deliberately: a
stop confounds the entry test, and the sweep already showed the stop is entangled with
the reward-to-risk gate rather than separable from it.

Every screen is measured the same way — forward excess return against SPY over the same
window, a matched random-date control in the same name, and a block bootstrap over
calendar time.

    python scripts/research_screens.py --hold 10
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equity_monitor import panel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "ledger")
HISTORY = os.path.join(LEDGER, "history")
BENCH = "SPY"
# Anything cap-weighted carries the same mega-cap concentration, so switching from SPY to
# the Russell 1000 or 3000 barely moves the bar (14.0% and 13.7% a year against SPY's
# 14.3% over 2013-2026). The equal-weighted S&P is the one that actually measures the
# average large-cap stock, at 11.7% — a 2.6%/yr difference that every alpha in this file
# was being charged.
#
# PANEL is stronger still: the equal-weighted return of the panel itself on the same day.
# It removes the market, the concentration *and* the panel's own survivorship tilt in one
# step, because the benchmark is drawn from the same biased pool as the candidates. It
# answers "did the screened names beat the average name available to the screen", which
# is the only question a stock-selection rule is actually responsible for.
BENCHMARKS = ("SPY", "RSP", "IWB", "IWV", "IWM", "VTI", "PANEL")
FALLBACK = "SPY"
# The panel is fixed at the longest horizon any run uses, so --hold never changes the
# universe underneath a comparison.
MAX_HOLD_FOR_PANEL = 25
WARMUP = 252


def rolling_rsi(closes: list[float], window: int = 14) -> list[float | None]:
    """Wilder RSI at every bar, in one pass.

    Recomputing from bar zero at each date is quadratic in the history depth, which is
    the whole point of having deep history. This is O(n).
    """
    out: list[float | None] = [None] * len(closes)
    if len(closes) < window + 1:
        return out
    gains = losses = 0.0
    for i in range(1, window + 1):
        ch = closes[i] - closes[i - 1]
        gains += max(ch, 0.0)
        losses += max(-ch, 0.0)
    ag, al = gains / window, losses / window
    out[window] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(window + 1, len(closes)):
        ch = closes[i] - closes[i - 1]
        ag = (ag * (window - 1) + max(ch, 0.0)) / window
        al = (al * (window - 1) + max(-ch, 0.0)) / window
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def rolling_max(vals: list[float], window: int) -> list[float]:
    """Trailing maximum. Monotonic deque — O(n), not O(n*window)."""
    from collections import deque
    dq: deque = deque()
    out = [0.0] * len(vals)
    for i, v in enumerate(vals):
        while dq and vals[dq[-1]] <= v:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - window:
            dq.popleft()
        out[i] = vals[dq[0]]
    return out


def rolling_mean_sd(vals: list[float], window: int):
    s = ss = 0.0
    means: list[float | None] = [None] * len(vals)
    sds: list[float | None] = [None] * len(vals)
    for i, v in enumerate(vals):
        s += v
        ss += v * v
        if i >= window:
            s -= vals[i - window]
            ss -= vals[i - window] ** 2
        if i >= window - 1:
            m = s / window
            means[i] = m
            sds[i] = math.sqrt(max(ss / window - m * m, 0.0))
    return means, sds


def rolling_atr(highs, lows, closes, window: int = 14) -> list[float | None]:
    out: list[float | None] = [None] * len(closes)
    trs = [0.0] * len(closes)
    for i in range(1, len(closes)):
        pc = closes[i - 1]
        trs[i] = max(highs[i] - lows[i], abs(highs[i] - pc), abs(lows[i] - pc))
    s = 0.0
    for i in range(1, len(closes)):
        s += trs[i]
        if i > window:
            s -= trs[i - window]
        if i >= window:
            out[i] = s / window
    return out


def rolling_beta(closes: list[float], bench_closes: list[float],
                 window: int = 252) -> list[float | None]:
    """Trailing beta against the benchmark, estimated only on data available at the bar.

    Without this, "excess return vs SPY" quietly rewards leverage: drawdown screens select
    high-beta names, and in a rising market a beta of 1.4 shows up as excess return that
    any levered index position would have produced. Beta-adjusting is what separates the
    screen from the leverage it happens to pick up.
    """
    n = min(len(closes), len(bench_closes))
    out: list[float | None] = [None] * len(closes)
    rs = [0.0] * n
    rb = [0.0] * n
    for i in range(1, n):
        rs[i] = closes[i] / closes[i - 1] - 1.0 if closes[i - 1] else 0.0
        rb[i] = bench_closes[i] / bench_closes[i - 1] - 1.0 if bench_closes[i - 1] else 0.0
    sxy = sxx = sx = sy = 0.0
    for i in range(1, n):
        sxy += rs[i] * rb[i]
        sxx += rb[i] * rb[i]
        sx += rb[i]
        sy += rs[i]
        if i > window:
            j = i - window
            sxy -= rs[j] * rb[j]
            sxx -= rb[j] * rb[j]
            sx -= rb[j]
            sy -= rs[j]
        if i >= window:
            k = window
            cov = sxy - sx * sy / k
            var = sxx - sx * sx / k
            out[i] = (cov / var) if var > 1e-12 else None
    return out


def features(d: dict) -> dict:
    """Every rolling input a screen might read, computed once per name."""
    c, h, l = d["close"], d["high"], d["low"]
    n = len(c)
    hi252 = rolling_max(h, 252)
    hi21 = rolling_max(h, 21)
    ma50, sd50 = rolling_mean_sd(c, 50)
    ma200, _ = rolling_mean_sd(c, 200)
    return {
        "dates": d["dates"], "close": c, "high": h, "low": l, "n": n,
        "rsi": rolling_rsi(c),
        "atr": rolling_atr(h, l, c),
        "dd252": [(hi252[i] - c[i]) / hi252[i] if hi252[i] > 0 else 0.0 for i in range(n)],
        "dd21": [(hi21[i] - c[i]) / hi21[i] if hi21[i] > 0 else 0.0 for i in range(n)],
        "ma50": ma50, "sd50": sd50, "ma200": ma200,
        # 12-month return skipping the most recent month, which is how momentum is
        # formed precisely so that short-term reversal does not contaminate it.
        "mom": [((c[i - 21] / c[i - 252] - 1.0) if i >= 252 and c[i - 252] > 0 else None)
                for i in range(n)],
    }


# --- the screens -------------------------------------------------------------
# Each returns True if the name is a candidate at bar i.
def _oversold(f, i, rsi_max=35.0, z_max=-1.5):
    r, ma, sd = f["rsi"][i], f["ma50"][i], f["sd50"][i]
    z = (f["close"][i] - ma) / sd if (ma is not None and sd) else None
    return (r is not None and r <= rsi_max) or (z is not None and z <= z_max)


def _rr(f, i, fib=0.382):
    """Reward-to-risk against a 3xATR stop, entry assumed at the low."""
    a = f["atr"][i]
    if not a:
        return 0.0
    hi = f["close"][i] / (1 - f["dd252"][i]) if f["dd252"][i] < 1 else f["close"][i]
    return (fib * (hi - f["close"][i])) / (3.0 * a)


SCREENS = {
    "A baseline (your spec)":
        lambda f, i, q: f["dd252"][i] >= 0.15 and _rr(f, i) >= 1.5 and _oversold(f, i),
    "B baseline + quality":
        lambda f, i, q: f["dd252"][i] >= 0.15 and _rr(f, i) >= 1.5 and _oversold(f, i)
                        and q is not None and q >= 0.65,
    "C deep drawdown only":
        lambda f, i, q: f["dd252"][i] >= 0.25,
    "D pure short-term oversold":
        lambda f, i, q: f["rsi"][i] is not None and f["rsi"][i] <= 30,
    "E dip in uptrend":
        lambda f, i, q: (f["ma200"][i] is not None and f["close"][i] > f["ma200"][i]
                         and f["rsi"][i] is not None and f["rsi"][i] <= 35),
    "F dip in uptrend + quality":
        lambda f, i, q: (f["ma200"][i] is not None and f["close"][i] > f["ma200"][i]
                         and f["rsi"][i] is not None and f["rsi"][i] <= 35
                         and q is not None and q >= 0.65),
    "G momentum winner + dip":
        lambda f, i, q: (f["mom"][i] is not None and f["mom"][i] > 0.10
                         and f["rsi"][i] is not None and f["rsi"][i] <= 35),
    "H momentum loser + dip":
        lambda f, i, q: (f["mom"][i] is not None and f["mom"][i] < -0.10
                         and f["rsi"][i] is not None and f["rsi"][i] <= 35),
}


def block_ci(rows, key="ex", block_days=36, draws=4000, seed=5):
    by: dict[int, list[float]] = {}
    for r in rows:
        d = r["date"]
        b = dt.date(int(d[:4]), int(d[5:7]), int(d[8:10])).toordinal() // block_days
        by.setdefault(b, []).append(r[key])
    blocks = [v for _, v in sorted(by.items())]
    if len(blocks) < 20:
        return None, len(blocks)
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        pool = [v for _ in blocks for v in rng.choice(blocks)]
        means.append(sum(pool) / len(pool))
    means.sort()
    return (means[int(0.025 * len(means))], means[int(0.975 * len(means))]), len(blocks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-adv", type=float, default=0.0,
                    help="minimum 21-day average dollar volume at entry. Previously "
                         "declared and never read, which mattered: the alpha concentrates "
                         "in deep-drawdown names and a corrupted entry bar traded at $0.07.")
    ap.add_argument("--min-price", type=float, default=0.0,
                    help="minimum close at entry")
    ap.add_argument("--manifest", default="",
                    help="write the panel manifest (symbol, bars, span, file hash) here. "
                         "Nothing in this repo is reproducible without one — the history "
                         "directory was rewritten continuously while it was being read.")
    ap.add_argument("--benchmark", default="SPY", choices=BENCHMARKS)
    args = ap.parse_args()

    # One panel, fixed independently of --hold, so results at different horizons are
    # comparable. The old loader admitted a name whenever it had WARMUP+hold+30 bars,
    # which silently changed the universe (1058/1057/1053 names) between runs.
    files = sorted(f for f in os.listdir(HISTORY) if f.endswith(".json"))
    series, dropped = {}, {"short": 0, "etf": 0, "unreadable": 0}
    keep_bench = {args.benchmark, FALLBACK} if hasattr(args, "benchmark") else {BENCH}
    for fn in files:
        sym = fn[:-5]
        try:
            with open(os.path.join(HISTORY, fn)) as fh:
                d = json.load(fh)
        except Exception:
            dropped["unreadable"] += 1
            continue
        if panel.is_etf(sym) and sym not in keep_bench:
            dropped["etf"] += 1
            continue
        if len(d.get("close") or []) < WARMUP + MAX_HOLD_FOR_PANEL + 30:
            dropped["short"] += 1
            continue
        series[sym] = d
    if args.limit:
        keep = [s for s in sorted(series) if s not in keep_bench][:args.limit]
        series = {s: series[s] for s in keep + [b for b in keep_bench if b in series]}
    for etf in BENCHMARKS:
        if etf != "PANEL" and etf in series and etf != args.benchmark:
            series.pop(etf)               # never screen an index fund
    if args.benchmark != "PANEL" and args.benchmark not in series:
        sys.exit(f"{args.benchmark} missing from {HISTORY}")

    if args.benchmark == "PANEL":
        # Equal-weighted daily return of every name in the panel, chained into a level.
        by_date: dict[str, list[float]] = {}
        for d in series.values():
            c, dts = d["close"], d["dates"]
            for i in range(1, len(c)):
                if c[i - 1] > 0:
                    by_date.setdefault(dts[i], []).append(c[i] / c[i - 1] - 1.0)
        days = sorted(by_date)
        lvl, bdates, bc = 1.0, [], []
        for dt_ in days:
            r = by_date[dt_]
            lvl *= 1.0 + sum(r) / len(r)
            bdates.append(dt_)
            bc.append(lvl)
        bidx = {d: i for i, d in enumerate(bdates)}
    else:
        bench = series.pop(args.benchmark)
        bdates = bench["dates"]
        bidx = {d: i for i, d in enumerate(bdates)}
        bc = bench["close"]

    ranks = {}
    path = os.path.join(LEDGER, "index.json")
    if os.path.exists(path):
        with open(path) as fh:
            rows = [r for r in json.load(fh).get("all", []) if r.get("q_raw") is not None]
        rows.sort(key=lambda r: r["q_raw"])
        ranks = {r["symbol"]: i / (len(rows) - 1) for i, r in enumerate(rows)}

    print(f"panel: {len(series)} names   benchmark {args.benchmark}   "
          f"hold {args.hold} sessions   exit: time only")
    print(f"  excluded from the panel: " +
          ", ".join(f"{k} {v}" for k, v in dropped.items() if v))
    if args.manifest:
        with open(args.manifest, "w") as fh:
            json.dump(panel.manifest(HISTORY, sorted(series)), fh, indent=0)
        print(f"  panel manifest -> {args.manifest}")
    span = (min(d["dates"][0] for d in series.values()),
            max(d["dates"][-1] for d in series.values()))
    print(f"history spans {span[0]} .. {span[1]}\n")

    # Align each name's bars to the benchmark's calendar before estimating beta, so the
    # regression is on same-day pairs rather than on two series that merely have the same
    # length.
    feats = {}
    for sym, d in series.items():
        f = features(d)
        # Per-bar usability, and the windows that span a bad bar. Excluding the window
        # rather than the name keeps every genuine observation the name has.
        f["ok"] = panel.clean_windows(panel.bar_flags(d), WARMUP, MAX_HOLD_FOR_PANEL)
        f["volume"] = d.get("volume") or [0.0] * len(d["close"])
        paired_b = [bc[bidx[dt_]] if dt_ in bidx else None for dt_ in d["dates"]]
        last = None
        for k, v in enumerate(paired_b):
            if v is None:
                paired_b[k] = last if last is not None else (bc[0] if bc else 1.0)
            else:
                last = v
        f["beta"] = rolling_beta(d["close"], paired_b)
        feats[sym] = f
    rng = random.Random(17)
    results = {k: {"ev": [], "ct": []} for k in SCREENS}
    skipped = {"bad window": 0, "no benchmark date": 0, "window misaligned": 0,
               "below min-adv": 0, "below min-price": 0, "no beta": 0, "no control": 0}
    controls_rng = {k: random.Random(1000 + h) for h, k in enumerate(SCREENS)}

    for sym, f in feats.items():
        q = ranks.get(sym)
        n = f["n"]
        nxt = {k: WARMUP for k in SCREENS}
        for i in range(WARMUP, n - args.hold):
            if not f["ok"][i]:
                skipped["bad window"] += 1
                continue
            date = f["dates"][i]
            bi = bidx.get(date)
            if bi is None or bi + args.hold >= len(bc):
                skipped["no benchmark date"] += 1
                continue
            # The forward window is indexed positionally on the name and on the
            # benchmark separately. A hole in either makes those two land on different
            # calendar days — CHK produced one event whose name-side window ran 1,312
            # days longer than the benchmark's, recorded as +2,574% of alpha.
            if f["dates"][i + args.hold] != bdates[bi + args.hold]:
                skipped["window misaligned"] += 1
                continue
            if args.min_adv > 0:
                w = f["volume"][max(0, i - 20):i + 1]
                px = f["close"][max(0, i - 20):i + 1]
                adv = sum(a * b for a, b in zip(w, px)) / max(1, len(w))
                if adv < args.min_adv:
                    skipped["below min-adv"] += 1
                    continue
            if f["close"][i] < args.min_price:
                skipped["below min-price"] += 1
                continue
            mkt = bc[bi + args.hold] / bc[bi] - 1.0
            fwd = f["close"][i + args.hold] / f["close"][i] - 1.0
            beta = f["beta"][i]
            if beta is None:
                skipped["no beta"] += 1
                continue
            beta = max(0.2, min(3.0, beta))     # trim, do not let a bad fit dominate
            alpha = fwd - beta * mkt
            for name, test in SCREENS.items():
                if i < nxt[name]:
                    continue
                try:
                    hit = test(f, i, q)
                except Exception:
                    hit = False
                if not hit:
                    continue
                pending = {"date": date, "ex": fwd - mkt,
                           "al": alpha, "beta": beta, "sym": sym}
                # The control was drawn uniformly over the name's whole history from a
                # single RNG shared across screens, so its dates depended on which
                # earlier screens had fired, and it `continue`d after the event was
                # already appended — desyncing the arms by up to 22 events and making
                # the control column irreproducible run to run. Now: per-screen RNG,
                # bounded retries, and the event is only kept if its control is valid.
                crng = controls_rng[name]
                ctl = None
                for _ in range(8):
                    ci_ = crng.randrange(WARMUP, n - args.hold)
                    if not f["ok"][ci_]:
                        continue
                    cd = f["dates"][ci_]
                    cbi = bidx.get(cd)
                    if cbi is None or cbi + args.hold >= len(bc):
                        continue
                    if f["dates"][ci_ + args.hold] != bdates[cbi + args.hold]:
                        continue
                    ctl = (ci_, cd, cbi)
                    break
                if ctl is None:
                    skipped["no control"] += 1
                    continue
                ci_, cd, cbi = ctl
                cfwd = f["close"][ci_ + args.hold] / f["close"][ci_] - 1.0
                cmkt = bc[cbi + args.hold] / bc[cbi] - 1.0
                cb = f["beta"][ci_]
                if cb is None:
                    skipped["no control"] += 1
                    continue
                cb = max(0.2, min(3.0, cb))
                # Both arms are appended together or neither is, so they can never
                # drift out of correspondence.
                results[name]["ev"].append(pending)
                results[name]["ct"].append({"date": cd, "ex": cfwd - cmkt,
                                            "al": cfwd - cb * cmkt})
                nxt[name] = i + args.hold          # one position per name at a time

    print("  bars discarded: " + ", ".join(f"{k} {v:,}" for k, v in skipped.items() if v))
    print()
    print(f"{'screen':<30}{'events':>7}{'beta':>6}{'raw ex':>8}{'ALPHA':>8}"
          f"{'95% CI on alpha':>20}{'control a':>10}{'win%':>6}")
    print("-" * 102)
    for name in SCREENS:
        ev, ct = results[name]["ev"], results[name]["ct"]
        if len(ev) < 30:
            print(f"{name:<30}{len(ev):>8}   too few events")
            continue
        ex = [e["ex"] for e in ev]
        al = [e["al"] for e in ev]
        cal = [c["al"] for c in ct]
        bt = st.mean([e["beta"] for e in ev])
        ci, nb = block_ci(ev, key="al")
        band = f"[{ci[0]:+.2%}, {ci[1]:+.2%}]" if ci else f"({nb} blocks, need 20)"
        win = sum(1 for v in al if v > 0) / len(al)
        print(f"{name:<30}{len(ev):>7}{bt:>6.2f}{st.mean(ex):>+8.2%}{st.mean(al):>+8.2%}"
              f"{band:>20}{st.mean(cal) if cal else 0:>+10.2%}{win:>6.0%}")
    # Comparing two screens by eyeballing their separate intervals is not a test, and it
    # is the error that produced the "G beats H" claim: two CIs that barely overlap can
    # still have a difference indistinguishable from zero. The difference needs its own
    # interval, bootstrapped on the *same* calendar blocks for both arms so the market
    # conditions cancel.
    print(f"\nPAIRED DIFFERENCES  (same blocks both arms, 4000 draws)")
    print(f"  {'comparison':<44}{'diff':>8}{'95% CI':>20}{'blocks won':>12}")
    pairs = [("G momentum winner + dip", "H momentum loser + dip"),
             ("B baseline + quality", "A baseline (your spec)"),
             ("F dip in uptrend + quality", "E dip in uptrend"),
             ("G momentum winner + dip", "A baseline (your spec)")]
    import datetime as _dt
    for a, b in pairs:
        ea, eb = results.get(a, {}).get("ev", []), results.get(b, {}).get("ev", [])
        if len(ea) < 50 or len(eb) < 50:
            continue
        def blocks(rows):
            out = {}
            for r in rows:
                d = r["date"]
                k = _dt.date(int(d[:4]), int(d[5:7]), int(d[8:10])).toordinal() // 36
                out.setdefault(k, []).append(r["al"])
            return out
        ba, bb = blocks(ea), blocks(eb)
        shared = sorted(set(ba) & set(bb))
        if len(shared) < 20:
            continue
        diffs = [st.mean(ba[k]) - st.mean(bb[k]) for k in shared]
        won = sum(1 for v in diffs if v > 0) / len(diffs)
        rr = random.Random(41)
        means = []
        for _ in range(4000):
            pool = [diffs[rr.randrange(len(diffs))] for _ in diffs]
            means.append(sum(pool) / len(pool))
        means.sort()
        lo, hi = means[100], means[3899]
        print(f"  {a.split()[0]} - {b.split()[0]:<38}"[:46]
              + f"{st.mean(diffs):>+7.2%}{f'[{lo:+.2%}, {hi:+.2%}]':>20}{won:>11.0%}")
    print(f"  blocks won is the share of shared calendar blocks the first screen led in.")
    print(f"  Near 50% means a coin flip whatever the point estimate says.")

    print("\n  ALPHA is the return after subtracting beta x market, using a trailing beta")
    print("  known at entry. RAW EX is the same figure before that subtraction; where the")
    print("  two diverge, the gap was leverage rather than selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
