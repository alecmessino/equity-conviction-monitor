#!/usr/bin/env python3
"""Event study: does the quality-gated dip actually pay, and pay for the stated reason.

Run against the per-symbol OHLCV the nightly build already writes:

    python -m equity_monitor.nightly            # populates ledger/history/
    python scripts/research_swing.py            # then this

Four things separate this from the event study the strategy is usually validated with.
Each one exists because the naive version reports a number that is too good and cannot
be traded.

**The exit is simulated along the path, not sampled at the horizon.** A naive study reads
close[t+20] and calls the difference the return. A real position with a stop does not
survive to t+20 — it is taken out on the day the low pierces the stop, which for a
mean-reversion entry is disproportionately early. Reading the horizon close credits the
strategy with recoveries it would have been stopped out of days before. Here every event
walks forward bar by bar, checking the stop against the low and the target against the
high, and exits on whichever comes first. Where both are touched in the same session the
**stop is assumed to fill first**, because daily bars cannot order two intrabar events and
the other assumption pays the strategy for a coin flip it did not win.

**Every return is reported against the benchmark over the identical window.** Buying a
name 25% off its high in a month the index rallies 6% is not evidence of mean reversion.
The market leg is the null this whole strategy has to beat, and it is a strong one.

**A matched control arm runs alongside every signal.** The same name, a random date,
the same holding rule. Without it a positive average return in a rising market reads as
signal, when it is drift the strategy did not have to work for. The number worth reading
is signal minus control, not signal.

**The interval is a block bootstrap over dates, not over events.** Dip events are
massively cross-correlated: a hundred names trigger on the same afternoon because the
index fell, and they are one observation about one afternoon, not a hundred. Resampling
events independently produces an interval several times too tight and is the single most
common way a dip-buying backtest convinces someone it has 4,000 observations when it has
about twenty. ``episodes`` is printed next to ``events`` for exactly this reason.

**Two biases run through everything below and neither can be removed with this data.**
Both inflate dip-buying, so the result is an upper bound rather than an estimate:

* *Survivorship.* The universe is today's index constituents. A name that fell 40% and
  kept falling until it was acquired at a discount, delisted, or dropped from the index is
  not in ``data/*.csv`` and therefore never enters the sample. Dip-buying studies are the
  single strategy most distorted by this, because the excluded names are excluded *for
  precisely the outcome being measured*.
* *Quality look-ahead.* Fundamentals are as-filed today, not as-of-date, which is the same
  leak ``rebuild.py`` refuses to write into ``ledger/snapshots/``. A company whose quality
  collapsed mid-sample carries today's score throughout. The ``--quality`` arms exist to
  bound it: run ``off`` and ``on`` and compare. The *gap* between them is edge plus leak,
  never edge alone, and it should be read as a ceiling.

Neither is a reason not to run this. They are a reason not to size a book on it until the
snapshot series in ``ledger/snapshots/`` is long enough to answer the same question
without a leak — which is the same standard ``edge.py`` already holds itself to.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equity_monitor import swing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "ledger")
HISTORY = os.path.join(LEDGER, "history")

WARMUP = 252          # bars needed before a 52-week high means anything
HORIZONS = (5, 10, 20, 25)
BENCHMARK = "SPY"
DD_BUCKETS = ((0.10, 0.15), (0.15, 0.20), (0.20, 0.25),
              (0.25, 0.30), (0.30, 0.40), (0.40, 0.60))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_series(limit: int | None = None, warmup: int = WARMUP) -> dict[str, dict]:
    """Per-symbol OHLCV from ledger/history/, which the nightly build regenerates."""
    if not os.path.isdir(HISTORY):
        sys.exit(f"no {HISTORY}/ — run `python -m equity_monitor.nightly` first, which "
                 f"writes the per-symbol OHLCV this study reads.")
    out: dict[str, dict] = {}
    names = sorted(f for f in os.listdir(HISTORY) if f.endswith(".json"))
    for fname in names:
        sym = fname[:-5]
        try:
            with open(os.path.join(HISTORY, fname)) as fh:
                d = json.load(fh)
        except Exception:
            continue
        if len(d.get("close") or []) < warmup + max(HORIZONS) + 10:
            continue
        out[sym] = d
        if limit and len(out) >= limit and BENCHMARK in out:
            break
    return out


def quality_ranks() -> dict[str, float]:
    """Today's quality percentile per symbol, from the published ledger.

    Today's, not as-of-date — see the module docstring. Returned separately from the
    price data so the caller must decide, explicitly, whether to use it.
    """
    path = os.path.join(LEDGER, "index.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        led = json.load(fh)
    rows = [r for r in led.get("all", []) if r.get("q_raw") is not None]
    rows.sort(key=lambda r: r["q_raw"])
    n = len(rows)
    return {r["symbol"]: (i / (n - 1) if n > 1 else 0.5) for i, r in enumerate(rows)}


# ---------------------------------------------------------------------------
# path-dependent exit
# ---------------------------------------------------------------------------
def simulate(highs: list[float], lows: list[float], opens: list[float],
             closes: list[float], i: int, entry: float, stop: float | None,
             target: float | None, max_hold: int) -> dict:
    """Walk the position forward bar by bar and exit on whatever is touched first.

    Three rules, each of which costs the strategy money and each of which is what an
    account would actually have experienced:

    **Ambiguous bars resolve against the strategy.** Where a bar's low pierces the stop
    and its high reaches the target in the same session, the stop is taken. A daily bar
    carries no intrabar sequence, and resolving it in the strategy's favour is how a
    backtest earns returns the account never saw.

    **A gap through the stop fills at the open, not at the stop.** Gapping down through
    a stop is the primary failure mode of a dip-buy — the name that fell on news falls
    again on news — and filling at the stop price pretends a resting order caught a
    price that never traded. The target branch keeps the limit price, which is already
    the conservative side: a gap *up* through a target fills better than modelled.

    **MFE and MAE stop at the exit.** Recorded past the fill they describe a position
    that was no longer open, so a −8% stop-out could report a −15% adverse excursion.
    """
    n = len(closes)
    mfe, mae = 0.0, 0.0
    for k in range(1, max_hold + 1):
        j = i + k
        if j >= n:
            break
        hit_stop = stop is not None and lows[j] <= stop
        hit_target = target is not None and highs[j] >= target
        if hit_stop:
            fill = min(stop, opens[j]) if j < len(opens) else stop
            ret = (fill - entry) / entry
            return {"exit": "stop", "days": k, "ret": ret,
                    "mfe": mfe, "mae": min(mae, ret)}
        if hit_target:
            ret = (target - entry) / entry
            return {"exit": "target", "days": k, "ret": ret,
                    "mfe": max(mfe, ret), "mae": mae}
        mfe = max(mfe, (highs[j] - entry) / entry)
        mae = min(mae, (lows[j] - entry) / entry)
    j = min(i + max_hold, n - 1)
    return {"exit": "time", "days": j - i, "ret": (closes[j] - entry) / entry,
            "mfe": mfe, "mae": mae}


def bench_return(bench_closes: list[float], bench_index: dict[str, int],
                 dates: list[str], i: int, days: int) -> float | None:
    """Benchmark return over the same calendar window, aligned by date not by offset."""
    if days <= 0 or i >= len(dates):
        return None
    a = bench_index.get(dates[i])
    end = min(i + days, len(dates) - 1)
    b = bench_index.get(dates[end])
    if a is None or b is None or bench_closes[a] <= 0:
        return None
    return bench_closes[b] / bench_closes[a] - 1.0


# ---------------------------------------------------------------------------
# the study
# ---------------------------------------------------------------------------
def run(series: dict[str, dict], ranks: dict[str, float], *, use_quality: bool,
        invert_quality: bool = False, max_hold: int = swing.MAX_HOLD_DAYS,
        stride: int = 1, seed: int = 7, warmup: int = WARMUP) -> dict:
    """Every signal event, with a date-matched random control for each."""
    rng = random.Random(seed)
    bench = series.get(BENCHMARK)
    if not bench:
        sys.exit(f"benchmark {BENCHMARK} missing from ledger/history/ — every return "
                 f"here is measured against it, and an absolute number would be the "
                 f"market leg wearing the strategy's name.")
    bench_closes = bench["close"]
    # Built rejecting empty and duplicate dates. A dict comprehension is last-write-wins,
    # so a single malformed "" date from the price chain becomes a live key pointing at an
    # arbitrary bar, and every lookup that misses lands on it.
    bench_index: dict[str, int] = {}
    for k, d in enumerate(bench["dates"]):
        if d and d not in bench_index:
            bench_index[d] = k

    events, controls = [], []
    for sym, d in series.items():
        if sym == BENCHMARK:
            continue
        closes, highs, lows = d["close"], d["high"], d["low"]
        opens = d.get("open") or closes
        dates = d.get("dates") or [""] * len(closes)
        q = ranks.get(sym)
        if use_quality:
            if q is None:
                continue
            q_eff = (1.0 - q) if invert_quality else q
        else:
            q_eff = 1.0            # gate open: every name passes on quality

        n = len(closes)
        # One position per name at a time. Without this a name that sits in the drawdown
        # band for six months emits an event every session — ~120 overlapping windows on
        # ~5 non-overlapping trades' worth of information, each counted as an independent
        # observation. It is the largest single source of false precision in the study,
        # and it compounds with the bootstrap rather than being caught by it.
        next_ok = warmup
        for i in range(warmup, n - max_hold, stride):
            if i < next_ok:
                continue
            c = closes[:i + 1]
            h, l = highs[:i + 1], lows[:i + 1]
            price = c[-1]
            hi52 = max(h[-min(252, warmup):])
            dd = (hi52 - price) / hi52 if hi52 > 0 else 0.0
            if dd < swing.MIN_DRAWDOWN or dd > swing.MAX_DRAWDOWN:
                continue
            bi = _bidx(bench_index, dates, i)
            if bi is None:
                continue          # unknown benchmark date — drop rather than guess
            ma50 = (sum(c[-50:]) / 50) if len(c) >= 50 else None
            atr = _atr(h, l, c)
            row = {"price": price, "ma50": ma50, "atr14": atr, "drawdown_52w": dd}
            s = swing.evaluate(row, c, h, l, bench_closes[:bi], quality_pctile=q_eff)
            if s["swing_signal"] not in {"PRIME", "SETUP"}:
                continue
            sim = simulate(highs, lows, opens, closes, i, price,
                           s["stop"], s["target_1"], max_hold)
            mkt = bench_return(bench_closes, bench_index, dates, i, sim["days"])
            events.append({
                "symbol": sym, "date": dates[i], "dd": dd,
                "rel_dd": s["rel_drawdown_52w"], "rr": s["reward_risk"],
                "rsi": s["rsi14"], "q": q, "signal": s["swing_signal"],
                **sim, "mkt": mkt,
                "excess": (sim["ret"] - mkt) if mkt is not None else None,
            })
            next_ok = i + max_hold

            # Matched control: same name, same exit *rule*, a date drawn at random.
            # The target is derived from the control date's own leg, not from the
            # signal's. Taking the signal's upside while taking the control's own ATR
            # stop builds a far target on a tight stop — a different strategy, which
            # stops out far more often and makes the control look worse than the signal
            # for a reason that has nothing to do with the signal.
            ci = rng.randrange(warmup, n - max_hold)
            cprice = closes[ci]
            catr = _atr(highs[:ci + 1], lows[:ci + 1], closes[:ci + 1])
            cstop = cprice - swing.STOP_ATR_MULT * catr if catr else None
            cleg = swing.selloff_leg(highs[:ci + 1], lows[:ci + 1])
            if cleg:
                ctarget = swing.targets(cleg, cprice, catr)["target_1"]
            else:
                # No leg at the control date: preserve the signal's reward-to-risk
                # multiple rather than its raw upside, so the exit rule matches.
                ctarget = (cprice + (s["reward_risk"] or 1.0) * (cprice - cstop)
                           if cstop else None)
            csim = simulate(highs, lows, opens, closes, ci, cprice, cstop, ctarget, max_hold)
            cmkt = bench_return(bench_closes, bench_index, dates, ci, csim["days"])
            controls.append({"symbol": sym, "date": dates[ci] if ci < len(dates) else "",
                             **csim, "mkt": cmkt,
                             "excess": (csim["ret"] - cmkt) if cmkt is not None else None,
                             "pair_date": dates[i]})
    return {"events": events, "controls": controls}


def _bidx(bench_index: dict[str, int], dates: list[str], i: int) -> int | None:
    """Benchmark bars available as of the event date, or None if the date is unknown.

    The previous version fell back to the *stock's* positional index when the date was
    missing from the benchmark. The two series need not be aligned, so that fallback
    could hand ``relative_drawdown`` a benchmark slice ending after the event date — a
    look-ahead leak that produced a plausible number rather than an error. Returning
    None and dropping the event is the only safe reading: a benchmark-relative measure
    computed against the wrong bar is worse than one not computed at all.
    """
    if i >= len(dates):
        return None
    j = bench_index.get(dates[i])
    return None if j is None else j + 1


def _atr(highs: list[float], lows: list[float], closes: list[float],
         window: int = 14) -> float | None:
    n = len(closes)
    if n < window + 1:
        return None
    trs = []
    for i in range(n - window, n):
        pc = closes[i - 1]
        trs.append(max(highs[i] - lows[i], abs(highs[i] - pc), abs(lows[i] - pc)))
    return sum(trs) / len(trs) if trs else None


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
# Contiguous blocks below this and the interval is not worth printing. Resampling a
# handful of blocks estimates its own width badly, and a confident-looking number is
# worse than none.
MIN_BLOCKS = 20


def block_bootstrap(events: list[dict], key: str = "excess", draws: int = 2000,
                    seed: int = 11, block: int = swing.MAX_HOLD_DAYS
                    ) -> tuple[float, float] | None:
    """95% interval by **circular block bootstrap over contiguous calendar runs**.

    Resampling individual dates — which is what this did before — is a *cluster*
    bootstrap. It removes the within-day correlation of a hundred names triggering on one
    afternoon, and nothing else. It leaves the correlation *between adjacent days*
    entirely intact, which for a strategy holding 25 sessions is most of the dependence
    there is: an event on Monday and one on Tuesday are the same trade seen twice.

    Contiguous blocks of at least the holding period keep neighbouring days together, so
    a resample cannot break a drawdown episode into independent pieces. Below
    ``MIN_BLOCKS`` distinct blocks the function returns None rather than an interval, and
    the caller prints the block count instead. On a two-episode sample that is the
    correct output: there is no interval to report.
    """
    by_date: dict[str, list[float]] = {}
    for e in events:
        v = e.get(key)
        if v is not None and e.get("date"):
            by_date.setdefault(e["date"], []).append(v)
    dates = sorted(by_date)                      # chronological, not insertion order
    if not dates:
        return None
    blocks = [dates[k:k + block] for k in range(0, len(dates), block)]
    if len(blocks) < MIN_BLOCKS:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        pool = [v for _ in blocks
                for d in rng.choice(blocks) for v in by_date[d]]
        if pool:
            means.append(sum(pool) / len(pool))
    if not means:
        return None
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def describe(events: list[dict], controls: list[dict], label: str) -> None:
    if not events:
        print(f"\n{label}: no events fired.")
        return
    ex = [e["excess"] for e in events if e["excess"] is not None]
    raw = [e["ret"] for e in events]
    dates = {e["date"] for e in events}
    months = {e["date"][:7] for e in events if e["date"]}
    cex = [c["excess"] for c in controls if c["excess"] is not None]

    span = f"{min(d for d in dates if d)} .. {max(d for d in dates if d)}" if any(dates) else "?"
    print(f"\n{label}")
    print(f"  events {len(events)}   distinct dates {len(dates)}   distinct months {len(months)}")
    print(f"  event window {span}")
    print(f"  -> {max(1, len(dates)//swing.MAX_HOLD_DAYS)} independent {swing.MAX_HOLD_DAYS}-session "
          f"blocks, not {len(events)} events. That is the sample size.")
    print(f"  raw return      mean {st.mean(raw):+7.2%}   median {st.median(raw):+7.2%}")
    if ex:
        ci = block_bootstrap(events)
        nblocks = max(1, len({e["date"] for e in events if e.get("date")}) // swing.MAX_HOLD_DAYS)
        band = (f"   95% CI [{ci[0]:+.2%}, {ci[1]:+.2%}]" if ci else
                f"   (no CI: {nblocks} independent blocks, need {MIN_BLOCKS})")
        print(f"  vs benchmark    mean {st.mean(ex):+7.2%}   median {st.median(ex):+7.2%}{band}")
    if cex:
        print(f"  control arm     mean {st.mean(cex):+7.2%}   "
              f"(random dates, same names, same exit rule)")
        if ex:
            print(f"  SIGNAL - CONTROL{st.mean(ex) - st.mean(cex):+8.2%}   "
                  f"<- the number worth reading")
    exits: dict[str, int] = {}
    for e in events:
        exits[e["exit"]] = exits.get(e["exit"], 0) + 1
    total = len(events)
    print("  exits           " + "  ".join(
        f"{k} {v/total:.0%}" for k, v in sorted(exits.items())))
    print(f"  holding days    median {st.median([e['days'] for e in events]):.0f}"
          f"   MFE {st.mean([e['mfe'] for e in events]):+.2%}"
          f"   MAE {st.mean([e['mae'] for e in events]):+.2%}")


def by_bucket(events: list[dict]) -> None:
    print(f"\n  {'drawdown':>13} {'n':>5} {'dates':>6} {'excess':>9} {'win':>6} "
          f"{'target':>7} {'stop':>6}")
    for lo, hi in DD_BUCKETS:
        sel = [e for e in events if lo <= e["dd"] < hi]
        if len(sel) < 5:
            continue
        ex = [e["excess"] for e in sel if e["excess"] is not None]
        if not ex:
            continue
        win = sum(1 for v in ex if v > 0) / len(ex)
        tgt = sum(1 for e in sel if e["exit"] == "target") / len(sel)
        stp = sum(1 for e in sel if e["exit"] == "stop") / len(sel)
        print(f"  {lo:>5.0%}-{hi:<7.0%} {len(sel):>5} {len({e['date'] for e in sel}):>6} "
              f"{st.mean(ex):>+8.2%} {win:>5.0%} {tgt:>6.0%} {stp:>5.0%}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None, help="cap the symbol count")
    ap.add_argument("--stride", type=int, default=1,
                    help="sample every Nth session; raises speed, lowers overlap")
    ap.add_argument("--max-hold", type=int, default=swing.MAX_HOLD_DAYS)
    ap.add_argument("--warmup", type=int, default=WARMUP,
                    help="bars required before the first event; 252 makes the 52-week "
                         "high a real 52-week high. Lower it only to exercise the "
                         "harness on a short series, never to report a result.")
    ap.add_argument("--json", default="", help="write the event table here")
    args = ap.parse_args()

    series = load_series(args.limit, args.warmup)
    ranks = quality_ranks()
    print(f"symbols {len(series)}   quality ranks {len(ranks)}   "
          f"warmup {args.warmup}   max hold {args.max_hold}")
    if args.warmup < 252:
        print(f"  WARNING: warmup {args.warmup} < 252, so 'drawdown from the 52-week "
              f"high' is really a drawdown from a {args.warmup}-bar high. Diagnostic only.")
    if args.warmup < 50:
        print(f"  WARNING: warmup {args.warmup} < 50 leaves ma50 undefined, so the z50 "
              f"oversold trigger cannot fire and entries degenerate to RSI-only.")

    # The burn-in is not free. The first `warmup` bars of every series can never be an
    # event date, so a short history does not merely give a small sample — it silently
    # removes the *oldest* market episodes, which on a two-year series is where the only
    # severe selloff usually sits. Stating the loss is the difference between a small
    # honest sample and a confident number about a period the study never looked at.
    depth = min((len(d["close"]) for d in series.values()), default=0)
    typical = max((len(d["close"]) for d in series.values()), default=0)
    usable = max(0, typical - args.warmup - args.max_hold)
    print(f"  history {typical} bars/name; burn-in consumes the first {args.warmup}, "
          f"leaving {usable} event dates per name (~{usable/21:.0f} months).")
    if usable < 500:
        print(f"  WARNING: {usable} event dates is roughly {max(1, usable//swing.MAX_HOLD_DAYS)} "
              f"independent {swing.MAX_HOLD_DAYS}-session blocks per name. Episodes older "
              f"than ~{typical/21:.0f} months are inside the burn-in and are NOT sampled. "
              f"Widen the price window before reading any number below as a result.")
    if not ranks:
        print("no ledger/index.json — the quality arms will be skipped.")

    arms = [("quality gate OFF  (price signal alone)", dict(use_quality=False))]
    if ranks:
        arms += [
            ("quality gate ON   (today's Q — carries look-ahead)", dict(use_quality=True)),
            ("quality INVERTED  (falsification arm)",
             dict(use_quality=True, invert_quality=True)),
        ]

    all_events = {}
    for label, kw in arms:
        out = run(series, ranks, max_hold=args.max_hold, stride=args.stride,
                  warmup=args.warmup, **kw)
        describe(out["events"], out["controls"], label)
        by_bucket(out["events"])
        all_events[label] = out["events"]

    print("\nRead the arms against each other, not on their own:")
    print("  * gate ON minus gate OFF is the quality gate's apparent value — and it is an")
    print("    upper bound, because today's Q leaks into every historical event.")
    print("  * the INVERTED arm is the falsification. If low quality dips pay as well as")
    print("    high quality ones, the gate is decoration and the edge is elsewhere.")
    print("  * every arm still carries survivorship: names that fell and never came back")
    print("    left the index and are not in this sample.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({k: v for k, v in all_events.items()}, fh, separators=(",", ":"))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
