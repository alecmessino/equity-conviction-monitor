#!/usr/bin/env python3
"""Resolve every price file on disk to the listing episode its bars actually belong to.

A ticker is a slot on an exchange, and the vendor serves the slot, not the company. Left
unresolved, a delisted-cohort study reads SentinelOne as Sprint and a survivor panel reads
the company that held MRNA until 2018 as Moderna. This writes ``_listing.json`` beside the
history so every study shares one answer:

    {"AA": {"first_listed": ..., "last_listed": ..., "status": "truncated"}, ...}

Status is what a caller acts on — ``clean`` needs nothing, ``truncated`` bars outside the
window, ``recycled`` holds no bar of the company that was asked for and cannot be studied
at all, and ``unresolved`` means the registry and the file disagree structurally, so the
bounds are left open rather than deleting a name over a bad row.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from equity_monitor import panel  # noqa: E402

TICKERS = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
US_EXCHANGES = {"NYSE", "NASDAQ", "NYSE ARCA", "AMEX", "NYSE MKT"}
HISTORY = os.path.join(ROOT, "ledger", "history")


def registry(cache: str) -> list[dict]:
    if not os.path.exists(cache):
        with urllib.request.urlopen(TICKERS, timeout=180) as r, open(cache, "wb") as fh:
            fh.write(r.read())
    z = zipfile.ZipFile(cache)
    with z.open(z.namelist()[0]) as fh:
        return list(csv.DictReader(io.TextIOWrapper(fh, "utf-8")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--history", default=HISTORY)
    ap.add_argument("--cache", default="/tmp/tiingo_tickers.zip")
    args = ap.parse_args()

    rows = [r for r in registry(args.cache)
            if r.get("exchange") in US_EXCHANGES and r.get("assetType") == "Stock"
            and r.get("priceCurrency") == "USD"]
    eps = panel.episodes(rows)

    out, counts = {}, {}
    for fn in sorted(os.listdir(args.history)):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        sym = fn[:-5]
        with open(os.path.join(args.history, fn)) as fh:
            dates = (json.load(fh).get("dates") or [])
        if not dates:
            continue
        first, last, status = panel.resolve_episode(eps.get(sym), dates)
        out[sym] = {"first_listed": first, "last_listed": last, "status": status}
        counts[status] = counts.get(status, 0) + 1

    path = os.path.join(args.history, "_listing.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=0)
    print(f"resolved {len(out)} files -> {path}")
    for k in ("clean", "truncated", "recycled", "unresolved"):
        if counts.get(k):
            names = [s for s, v in out.items() if v["status"] == k]
            shown = ", ".join(names[:12]) + ("..." if len(names) > 12 else "")
            print(f"  {k:<11}{counts[k]:>5}   {shown if k != 'clean' else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
