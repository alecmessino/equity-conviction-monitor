#!/usr/bin/env python3
"""Measure the net sign of survivorship bias for a dip-buying strategy.

The usual claim is that survivorship inflates dip-buying: names that fell and kept falling
leave the index, so a current-constituent panel never sees them. For *this* strategy that
is only half of it. A large-cap 30% off its high rarely goes bankrupt — far more often it
is bought, and an opportunistic takeout at a premium to a depressed price is an excellent
outcome for a dip-buyer that the same panel also never sees. The two channels push in
opposite directions and the net sign is an empirical question, not a known constant.

**The central trap, and why this script is built the way it is.** Running the analysis on
the delisted cohort alone conditions on the outcome. At entry you do not know a name will
be acquired; selecting on it is the same class of error as scoring quality with today's
fundamentals. Numbers from that cohort describe a *mechanism* — they are not a strategy
result and cannot be traded.

So three panels are run and the comparison is what carries meaning:

    survivors   today's index members, the biased baseline every earlier run used
    delisted    names that later stopped trading — diagnostic only, outcome-conditioned
    combined    the universe as it actually stood, which is the tradeable measurement

**combined minus survivors is the survivorship effect.** Everything else here is
explanation for whichever way that number comes out.

One limitation stated plainly: the model's quality pillar cannot be computed for delisted
names, because it is built from filings this project reads for current constituents only.
The second axis of the difference-in-differences is therefore a price measure — trailing
three-year return over volatility — and it is **not** a quality proxy. Measured against the
605 survivors that carry both:

    price measure vs the model's fundamental Q   -0.165
    price measure vs 12-month momentum           +0.495
    fundamental Q vs 12-month momentum           -0.132

It is *negatively* related to quality and strongly related to momentum, so calling it a
quality proxy would report the momentum effect as a fundamental one. It is named `trend`
throughout for that reason.

That third correlation is weak, and it is worth being precise about what it does and does
not explain. It is driven almost entirely by the *yield* pillars, where price sits in the
denominator and the relationship is close to mechanical (fcf_yield -0.226, cfo_yield
-0.242); the operating pillars barely move (roic -0.016, gross_margin -0.056). So the
quality gate leans against momentum only through its valuation legs. That is far too
small to carry the recurring finding that gating on quality made screens worse — at 1,020
names the quality effect is not distinguishable from zero in either direction, and the
earlier readings in both directions were small-sample artifacts.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from equity_monitor import panel  # noqa: E402

HISTORY = os.path.join(ROOT, "ledger", "history")
BENCH = "RSP"          # equal weight: the average large-cap stock, not seven of them
FALLBACK_BENCH = "SPY"
WARMUP = 252
ETFS = {"SPY", "RSP", "IWB", "IWV", "IWM", "VTI"}


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


def rolling_max(vals, window):
    from collections import deque
    dq, out = deque(), [0.0] * len(vals)
    for i, v in enumerate(vals):
        while dq and vals[dq[-1]] <= v:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - window:
            dq.popleft()
        out[i] = vals[dq[0]]
    return out


def classify_ending(c, dates, last_trading_date):
    """Acquisition or collapse? The registry does not say, so read the price path.

    A takeout prints a jump into a flat line at the deal price; a collapse grinds down.
    Neither signature is certain — a deal can leak slowly, and a name can be delisted for
    a listing violation while healthy — so the classes are reported with their counts and
    never used as though they were a corporate-actions feed.
    """
    if len(c) < 90:
        return "unknown"
    tail = c[-60:]
    peak_1y = max(c[-252:]) if len(c) >= 252 else max(c)
    final = c[-1]
    ret60 = final / c[-60] - 1.0
    off_peak = final / peak_1y - 1.0
    flat_end = st.pstdev(c[-10:]) / final < 0.005 if final > 0 else False
    if ret60 > 0.10 or (flat_end and off_peak > -0.20):
        return "takeout"
    if off_peak < -0.50:
        return "collapse"
    return "other"


BAR_FIELDS = ("dates", "close", "high", "low", "open", "volume", "adjclose")


def load_panel(meta_path):
    """Series and cohort tag per symbol, with each name truncated at its own delisting.

    Cohort comes from the listing registry, not from where the price file happens to end.
    Reading the file was the earlier rule and it is exactly backwards on the names that
    matter: a recycled ticker carries a successor's bars right up to today, so the names
    this study exists to measure were being filed under *survivors*.

    Of 370 delisted names requested from the vendor, 325 returned data and 297 of those
    are the company that was asked for. The other 28 are successors holding the same
    symbol — S returns SentinelOne rather than Sprint, STI a 2024 relisting rather than
    SunTrust — and they are counted as the ordinary live companies they are, not as
    delisted ones. The delisted arm is 297 names, and saying it is 370 would be the
    survivorship error restated one level down.
    """
    listing = {}
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            listing = json.load(fh)
    series, tag = {}, {}
    dropped = {"recycled_whole": [], "short": 0, "etf": 0, "truncated": 0}
    for fn in sorted(os.listdir(HISTORY)):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        sym = fn[:-5]
        if panel.is_etf(sym) and sym not in (BENCH, FALLBACK_BENCH):
            dropped["etf"] += 1
            continue
        with open(os.path.join(HISTORY, fn)) as fh:
            d = json.load(fh)
        if len(d.get("close") or []) < WARMUP + 40:
            dropped["short"] += 1
            continue
        meta = listing.get(sym) or {}
        first, last, status = (meta.get("first_listed"), meta.get("last_listed"),
                               meta.get("status") or "unresolved")
        if status == "recycled":
            # No bar in the file belongs to the company that held this symbol: it is a
            # successor's history end to end and there is nothing here to measure.
            dropped["recycled_whole"].append(sym)
            continue
        keep = panel.episode_flags(d["dates"], first, last)
        if not all(keep):
            dropped["truncated"] += 1
            lo = keep.index(True)
            hi = len(keep) - keep[::-1].index(True)
            for f in BAR_FIELDS:
                if isinstance(d.get(f), list):
                    d[f] = d[f][lo:hi]
        if len(d["close"]) < WARMUP + 40:
            dropped["short"] += 1
            continue
        d["_ok"] = panel.bar_flags(d)
        series[sym] = d
        tag[sym] = ("delisted" if status != "unresolved" and last
                    and last < "2026-06-01" else "survivor")
    return series, tag, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--benchmark", default=BENCH)
    args = ap.parse_args()

    series, tag, dropped = load_panel(os.path.join(HISTORY, "_listing.json"))
    bsym = args.benchmark if args.benchmark in series else FALLBACK_BENCH
    if bsym not in series:
        sys.exit("no benchmark in the panel")
    bench = series[bsym]
    bidx = {d: i for i, d in enumerate(bench["dates"])}
    bc = bench["close"]
    for e in ETFS:
        series.pop(e, None)
        tag.pop(e, None)

    nd = sum(1 for v in tag.values() if v == "delisted")
    print(f"panel {len(series)} names   survivors {len(series)-nd}   delisted {nd}")
    rec = dropped["recycled_whole"]
    print(f"dropped {len(rec)} wholly-recycled tickers "
          f"({', '.join(rec[:8])}{'...' if len(rec) > 8 else ''}); "
          f"truncated {dropped['truncated']} at their delisting; "
          f"{dropped['etf']} ETFs and {dropped['short']} short series excluded")
    print(f"benchmark {bsym} (equal weight)   hold {args.hold} sessions\n")

    endings = {}
    for sym, d in series.items():
        if tag[sym] == "delisted":
            endings[sym] = classify_ending(d["close"], d["dates"], d["dates"][-1])
    if endings:
        cnt = {}
        for v in endings.values():
            cnt[v] = cnt.get(v, 0) + 1
        print("how the delisted names ended, inferred from the price path:")
        for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
            print(f"   {k:<10}{v:>5}  ({v/len(endings):.0%})")
        print()

    H = args.hold
    events = []
    for sym, d in series.items():
        c, h, l, dates = d["close"], d["high"], d["low"], d["dates"]
        n = len(c)
        rsi = rsi_series(c)
        hi252 = rolling_max(h, 252)
        # No event may span a suspect bar in either direction: not the 756-session trend
        # window behind it, nor the H-session hold in front of it.
        clean = panel.clean_windows(d["_ok"], 756, H)
        nxt = WARMUP
        for i in range(WARMUP, n - H):
            if i < nxt or not clean[i] or hi252[i] <= 0 or c[i - 252] <= 0:
                continue
            dd = (hi252[i] - c[i]) / hi252[i]
            if dd < 0.15:
                continue
            bi = bidx.get(dates[i])
            if bi is None or bi + H >= len(bc):
                continue
            nxt = i + H
            # Trailing 3-year return per unit of volatility. NOT quality — measured at
            # -0.165 against the model's Q and +0.495 against 12-month momentum. It is a
            # trend measure and is named one.
            w = c[max(0, i - 756):i + 1]
            rets = [math.log(w[k] / w[k - 1]) for k in range(1, len(w)) if w[k - 1] > 0]
            if len(rets) < 200:
                continue
            sd = st.pstdev(rets)
            trend = (st.mean(rets) / sd) if sd > 0 else 0.0
            events.append({
                "sym": sym, "date": dates[i], "cohort": tag[sym], "dd": dd,
                "mom": c[i - 21] / c[i - 252] - 1.0, "q": trend,
                "ex": (c[i + H] / c[i] - 1.0) - (bc[bi + H] / bc[bi] - 1.0),
                "ending": endings.get(sym, ""),
            })

    if not events:
        sys.exit("no events — is ledger/history populated?")

    qs = sorted(e["q"] for e in events)
    qcut = qs[int(0.65 * (len(qs) - 1))]

    def report(rows, label):
        if len(rows) < 50:
            print(f"{label:<34}{len(rows):>7}   too few events")
            return None
        ex = [r["ex"] for r in rows]
        ci = block_ci(rows)
        band = f"[{ci[0]:+.2%}, {ci[1]:+.2%}]" if ci else "n/a"
        print(f"{label:<34}{len(rows):>7}{st.mean(ex):>+9.2%}{band:>20}"
              f"{sum(1 for v in ex if v > 0)/len(ex):>7.0%}")
        return st.mean(ex)

    print(f"{'panel':<34}{'events':>7}{'excess':>9}{'95% CI':>20}{'win%':>7}")
    print("-" * 78)
    surv = [e for e in events if e["cohort"] == "survivor"]
    dele = [e for e in events if e["cohort"] == "delisted"]
    m_s = report(surv, "survivors only (the old baseline)")
    m_d = report(dele, "delisted only (DIAGNOSTIC ONLY)")
    m_c = report(events, "combined (the tradeable panel)")
    if m_s is not None and m_c is not None:
        print(f"\n  SURVIVORSHIP EFFECT = combined - survivors = {m_c - m_s:+.2%} per {H}-session hold")
        print(f"  Positive means the biased panel UNDERSTATED the strategy.")

    print(f"\nDIFFERENCE-IN-DIFFERENCES  (TREND x drawdown depth, excess return)")
    print(f"trend = 3y return/volatility, cut at the 65th percentile ({qcut:+.3f}).")
    print(f"NOT quality: -0.165 against the model's Q, +0.495 against 12-month momentum.")
    for cohort, rows in (("survivors", surv), ("delisted [diagnostic]", dele),
                         ("combined", events)):
        print(f"\n  {cohort}")
        print(f"    {'drawdown':<14}{'weak trend':>16}{'strong trend':>16}{'difference':>14}")
        for lo, hi_ in ((0.15, 0.30), (0.30, 0.50), (0.50, 1.01)):
            lq = [r for r in rows if lo <= r["dd"] < hi_ and r["q"] < qcut]
            hq = [r for r in rows if lo <= r["dd"] < hi_ and r["q"] >= qcut]
            if len(lq) < 25 or len(hq) < 25:
                continue
            a, b = st.mean([r["ex"] for r in lq]), st.mean([r["ex"] for r in hq])
            print(f"    {lo:>5.0%}-{hi_:<8.0%}{a:>+15.2%}{b:>+16.2%}{b-a:>+14.2%}")

    if endings:
        print(f"\nDELISTED COHORT BY ENDING  [diagnostic — conditioned on the outcome]")
        print(f"    {'ending':<12}{'events':>8}{'excess':>10}{'deep-dd excess':>18}")
        for kind in ("takeout", "collapse", "other"):
            rows = [e for e in dele if e["ending"] == kind]
            deep = [e for e in rows if e["dd"] >= 0.30]
            if len(rows) < 25:
                continue
            d_ex = f"{st.mean([r['ex'] for r in deep]):+.2%}" if len(deep) >= 25 else "n/a"
            print(f"    {kind:<12}{len(rows):>8}{st.mean([r['ex'] for r in rows]):>+10.2%}{d_ex:>18}")
    return 0


def block_ci(rows, key="ex", block_days=36, draws=3000, seed=7):
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


if __name__ == "__main__":
    raise SystemExit(main())
