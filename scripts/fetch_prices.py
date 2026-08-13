#!/usr/bin/env python3
"""Deep price history into ledger/history/, from whichever source you have access to.

The nightly build fetches ~2 years, which is the right window for scoring tonight's board
and the wrong one for measuring whether the board predicts anything: a 252-bar burn-in
eats the oldest episode, so a two-year file yields an event study over whatever selloff
happens to fall in the back half. This script exists to seed a deeper store once, after
which the nightly extends it.

Sources, in the order you should reach for them:

* ``yahoo``   — no key, 10 years, split- and dividend-adjusted. Rate-limits hard from
                datacenter IPs: one request at a time with real spacing, and a 429
                lockout lasts tens of minutes, so the pacing below is deliberately timid.
* ``tiingo``  — key, 30+ years, the cleanest adjustments of the lot. The free tier's
                binding limit is **500 unique symbols per month**, not per day, so it
                cannot cover a 1000-name universe on its own. Use it for a core subset.
* ``fmp``     — key, ~5 years of prices, and the only free source here that also exposes
                *historical* fundamentals. That matters more than the price depth: it is
                the one way to remove the quality look-ahead the research harness can
                otherwise only bound.
* ``parquet`` — no network. Point it at a file you dumped locally with yfinance. This is
                the right answer when a cloud IP is blocked and your laptop is not.

Adjusted bars, consistently. Yahoo returns raw OHLC alongside an adjusted close; using
the raw high with an adjusted close silently breaks every measure anchored to a peak —
the retracement leg, the 52-week band, ATR. Open, high and low are scaled by the same
``adjclose / close`` factor so a bar stays internally consistent.

    python scripts/fetch_prices.py --source yahoo --years 10
    python scripts/fetch_prices.py --source parquet --path ~/prices.parquet
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equity_monitor import universe as uni

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY = os.path.join(ROOT, "ledger", "history")

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range={r}&interval=1d"
TIINGO = "https://api.tiingo.com/tiingo/daily/{sym}/prices?startDate={start}&format=json"
FMP = "https://financialmodelingprep.com/api/v3/historical-price-full/{sym}?from={start}&apikey={key}"

MIN_BARS = 260


def _get(url: str, headers: dict | None = None, tries: int = 5) -> dict | list | None:
    """One request, with a long backoff on 429. A Yahoo lockout outlasts a short retry."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       "Accept": "application/json",
                                                       **(headers or {})})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503):
                time.sleep(45 + attempt * 45 + random.random() * 10)
                continue
            return None
        except Exception:
            time.sleep(3 + attempt * 3)
    return None


def _write(sym: str, rows: dict) -> bool:
    if len(rows.get("close") or []) < MIN_BARS:
        return False
    os.makedirs(HISTORY, exist_ok=True)
    with open(os.path.join(HISTORY, f"{sym}.json"), "w") as fh:
        json.dump(rows, fh, separators=(",", ":"))
    return True


def from_yahoo(sym: str, years: int) -> dict | None:
    """Yahoo, with OHLC rescaled onto the adjusted close.

    ``range=max`` silently switches to a monthly interval — 168 bars rather than 10,000 —
    so the range is always an explicit number of years.
    """
    for variant in (sym.replace(".", "-"),
                    *( [f"{sym[:-1]}-{sym[-1]}"] if len(sym) > 2 and sym[-1] in "ABCK"
                       and "." not in sym and "-" not in sym else [] )):
        payload = _get(YAHOO.format(sym=variant, r=f"{years}y"))
        res = ((payload or {}).get("chart") or {}).get("result") or []
        if not res:
            continue
        node = res[0]
        q = ((node.get("indicators") or {}).get("quote") or [{}])[0]
        adj = ((node.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose") or []
        ts = node.get("timestamp") or []
        closes = q.get("close") or []
        out = {"symbol": sym, "dates": [], "open": [], "high": [], "low": [],
               "close": [], "volume": [], "source": "yahoo"}
        for i, c in enumerate(closes):
            if c is None or i >= len(ts) or not c:
                continue
            a = adj[i] if i < len(adj) and adj[i] is not None else c
            k = a / c                       # cumulative split+dividend factor for this bar
            o = (q.get("open") or [None] * len(closes))[i]
            h = (q.get("high") or [None] * len(closes))[i]
            l = (q.get("low") or [None] * len(closes))[i]
            v = (q.get("volume") or [None] * len(closes))[i]
            out["dates"].append(dt.datetime.utcfromtimestamp(ts[i]).date().isoformat())
            out["close"].append(round(float(a), 6))
            out["open"].append(round(float(o) * k if o is not None else a, 6))
            out["high"].append(round(float(h) * k if h is not None else a, 6))
            out["low"].append(round(float(l) * k if l is not None else a, 6))
            out["volume"].append(float(v) if v is not None else 0.0)
        if len(out["close"]) >= MIN_BARS:
            return out
    return None


def from_tiingo(sym: str, years: int, key: str) -> dict | None:
    start = (dt.date.today() - dt.timedelta(days=365 * years + 10)).isoformat()
    data = _get(TIINGO.format(sym=sym.replace(".", "-"), start=start),
                headers={"Authorization": f"Token {key}",
                         "Content-Type": "application/json"})
    if not isinstance(data, list) or not data:
        return None
    out = {"symbol": sym, "dates": [], "open": [], "high": [], "low": [],
           "close": [], "volume": [], "source": "tiingo"}
    for row in data:
        # Tiingo publishes adjusted columns directly — no rescaling to do.
        out["dates"].append((row.get("date") or "")[:10])
        out["close"].append(float(row.get("adjClose") or row.get("close") or 0))
        out["open"].append(float(row.get("adjOpen") or row.get("open") or 0))
        out["high"].append(float(row.get("adjHigh") or row.get("high") or 0))
        out["low"].append(float(row.get("adjLow") or row.get("low") or 0))
        out["volume"].append(float(row.get("adjVolume") or row.get("volume") or 0))
    return out


def from_fmp(sym: str, years: int, key: str) -> dict | None:
    start = (dt.date.today() - dt.timedelta(days=365 * years + 10)).isoformat()
    data = _get(FMP.format(sym=sym, start=start, key=key))
    hist = (data or {}).get("historical") or []
    if not hist:
        return None
    hist = list(reversed(hist))            # FMP returns newest-first
    out = {"symbol": sym, "dates": [], "open": [], "high": [], "low": [],
           "close": [], "volume": [], "source": "fmp"}
    for row in hist:
        c, ac = row.get("close"), row.get("adjClose")
        if c is None:
            continue
        k = (ac / c) if (ac and c) else 1.0
        out["dates"].append(row.get("date", "")[:10])
        out["close"].append(float(ac if ac else c))
        out["open"].append(float(row.get("open", c)) * k)
        out["high"].append(float(row.get("high", c)) * k)
        out["low"].append(float(row.get("low", c)) * k)
        out["volume"].append(float(row.get("volume") or 0))
    return out


def from_parquet(path: str) -> int:
    """Ingest a local dump — the answer when this machine's IP is blocked and yours isn't.

    Accepts either a long frame (columns: date, symbol, open/high/low/close/volume) or a
    yfinance ``download(group_by='ticker')`` wide frame. Prices must already be adjusted.
    """
    import pandas as pd
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.stack(level=0).rename_axis(["date", "symbol"]).reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    if "adj_close" in df.columns and "close" in df.columns:
        for col in ("open", "high", "low"):
            if col in df.columns:
                df[col] = df[col] * (df["adj_close"] / df["close"])
        df["close"] = df["adj_close"]
    written = 0
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("date")
        rows = {"symbol": str(sym), "source": "local",
                "dates": [str(d)[:10] for d in g["date"]],
                "close": [float(x) for x in g["close"]],
                "open": [float(x) for x in g.get("open", g["close"])],
                "high": [float(x) for x in g.get("high", g["close"])],
                "low": [float(x) for x in g.get("low", g["close"])],
                "volume": [float(x) for x in g.get("volume", [0] * len(g))]}
        written += _write(str(sym), rows)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", choices=("yahoo", "tiingo", "fmp", "parquet"),
                    default="yahoo")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--key", default=os.environ.get("PRICE_API_KEY", ""))
    ap.add_argument("--path", default="", help="parquet/csv for --source parquet")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--symbols", default="",
                    help="comma-separated symbols, or a path to a JSON list. Spend the "
                         "rate limit on the names the study needs before the tail of "
                         "the universe.")
    ap.add_argument("--pace", type=float, default=1.3,
                    help="seconds between requests. Yahoo 429s aggressively from cloud "
                         "IPs and the lockout lasts tens of minutes, so going faster "
                         "than this usually finishes slower.")
    ap.add_argument("--refresh", action="store_true",
                    help="refetch symbols already on disk (default skips them, so an "
                         "interrupted run resumes where it stopped)")
    args = ap.parse_args()

    if args.source == "parquet":
        if not args.path:
            sys.exit("--source parquet needs --path")
        print(f"wrote {from_parquet(args.path)} symbols into {HISTORY}")
        return 0
    if args.source in ("tiingo", "fmp") and not args.key:
        sys.exit(f"--source {args.source} needs --key or PRICE_API_KEY")

    if args.symbols:
        if os.path.exists(args.symbols):
            with open(args.symbols) as fh:
                syms = json.load(fh)
        else:
            syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        members, _ = uni.load()
        syms = [m.symbol for m in members]
    if args.limit:
        syms = syms[:args.limit]
    os.makedirs(HISTORY, exist_ok=True)

    ok = skipped = failed = 0
    started = time.time()
    for i, sym in enumerate(syms, 1):
        path = os.path.join(HISTORY, f"{sym}.json")
        if os.path.exists(path) and not args.refresh:
            skipped += 1
            continue
        if args.source == "yahoo":
            rows = from_yahoo(sym, args.years)
        elif args.source == "tiingo":
            rows = from_tiingo(sym, args.years, args.key)
        else:
            rows = from_fmp(sym, args.years, args.key)
        if rows and _write(sym, rows):
            ok += 1
        else:
            failed += 1
        if i % 25 == 0:
            rate = (time.time() - started) / max(1, ok + failed)
            left = (len(syms) - i) * rate / 60
            print(f"{i}/{len(syms)}  ok={ok} failed={failed} skipped={skipped}  "
                  f"~{left:.0f} min left", flush=True)
        # Jittered: a metronome is easier to fingerprint than a person.
        time.sleep(args.pace * (0.75 + random.random() * 0.5))

    print(f"DONE ok={ok} failed={failed} skipped={skipped} -> {HISTORY}")
    return 0 if ok or skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
