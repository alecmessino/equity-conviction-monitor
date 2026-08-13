#!/usr/bin/env python3
"""Build a point-in-time universe that includes the names that died.

Every panel in this project so far was drawn from *today's* index membership, and that is
not a sample of the market — it is a sample of what survived. Measured: the 93-name panel
returned 20.8% a year equal-weighted against the equal-weighted S&P's 11.7%, a 6.5-point
annual tilt over even the cap-weighted S&P. Dip-buying is the strategy most distorted by
this, because a name that fell and kept falling until it was acquired at a discount or
delisted is excluded *for precisely the outcome being measured*.

Tiingo publishes the fix for free and without a key: ``supported_tickers.zip`` carries
every ticker it has ever served with a startDate and an endDate. A name whose endDate is
in the past stopped trading — acquired, taken private, merged, or bankrupt. Filtering to
US common stocks that were *alive on a given date* reconstructs the universe as it stood,
including the roughly 3,400 names that are no longer in it.

Two honest limits, both worth stating rather than papering over:

* **The file says when a ticker stopped, not why.** An acquisition at a premium and a
  bankruptcy both look like an endDate. They have opposite implications for a dip-buying
  study, and separating them needs a corporate-actions source this does not have. The
  price series itself is the best available proxy: a name that ends after a sharp rise was
  probably bought, one that ends after a long decline probably was not.
* **Size is not in the file.** "Large cap as of 2016" cannot be read off it directly, so
  liquidity has to be measured from the price history after fetching — which means paying
  the request before knowing whether the name qualifies. That cost is the price of an
  unbiased sample; skipping it is how the bias gets back in.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import urllib.request
import zipfile

TICKERS = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
US_EXCHANGES = {"NYSE", "NASDAQ", "NYSE ARCA", "AMEX", "NYSE MKT"}


def fetch_registry(cache: str) -> list[dict]:
    if not os.path.exists(cache):
        with urllib.request.urlopen(TICKERS, timeout=180) as r, open(cache, "wb") as fh:
            fh.write(r.read())
    z = zipfile.ZipFile(cache)
    with z.open(z.namelist()[0]) as fh:
        return list(csv.DictReader(io.TextIOWrapper(fh, "utf-8")))


def alive_on(rows: list[dict], date: str) -> list[dict]:
    """Tickers trading on `date` — the universe as it stood, not as it ended up."""
    out = []
    for r in rows:
        if r.get("exchange") not in US_EXCHANGES:
            continue
        if r.get("assetType") != "Stock" or r.get("priceCurrency") != "USD":
            continue
        s, e = r.get("startDate") or "", r.get("endDate") or ""
        if s and s <= date and (not e or e >= date):
            out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--as-of", default="2016-01-04",
                    help="reconstruct membership as of this date")
    ap.add_argument("--sample", type=int, default=300,
                    help="how many to draw; the request budget, not the universe size")
    ap.add_argument("--dead-share", type=float, default=0.35,
                    help="share of the draw reserved for names that later stopped "
                         "trading. Set from the real death rate, not from taste — "
                         "sampling survivors and non-survivors at the wrong ratio "
                         "reintroduces the bias in a subtler form.")
    ap.add_argument("--cache", default="/tmp/tiingo_tickers.zip")
    ap.add_argument("--out", default="universe_pit.json")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rows = fetch_registry(args.cache)
    live = alive_on(rows, args.as_of)
    dead = [r for r in live if (r.get("endDate") or "9999") < "2026-06-01"]
    survived = [r for r in live if (r.get("endDate") or "9999") >= "2026-06-01"]
    rate = len(dead) / max(1, len(live))
    print(f"universe as of {args.as_of}: {len(live):,} US common stocks")
    print(f"  still trading today : {len(survived):,}")
    print(f"  stopped trading     : {len(dead):,}   ({rate:.1%} of the cohort)")
    print(f"\n  A study drawn from today's members omits that {rate:.1%} entirely.")

    import random
    rng = random.Random(args.seed)
    n_dead = min(len(dead), int(args.sample * args.dead_share))
    n_live = min(len(survived), args.sample - n_dead)
    pick = rng.sample(dead, n_dead) + rng.sample(survived, n_live)
    rng.shuffle(pick)
    syms = [r["ticker"] for r in pick]
    with open(args.out, "w") as fh:
        json.dump(syms, fh)
    meta = {r["ticker"]: {"start": r.get("startDate"), "end": r.get("endDate")}
            for r in pick}
    with open(args.out.replace(".json", "_meta.json"), "w") as fh:
        json.dump(meta, fh)
    print(f"\nwrote {len(syms)} symbols to {args.out} "
          f"({n_dead} that later stopped trading, {n_live} that did not)")
    print(f"and their listing windows to {args.out.replace('.json','_meta.json')}, so a")
    print("study can tell a name that was delisted from one that simply has short history.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
