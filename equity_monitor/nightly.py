"""Nightly builder — Equity Conviction Monitor.

Pulls features for the universe, enriches with RS, scores via model.score(),
and writes ledger/index.json. Requires FMP_API_KEY (env). If the key is absent
or a symbol fails, that symbol is skipped (never fabricated). The terminal
renders whatever the ledger contains — so a partial pull still shows real data.
"""
from __future__ import annotations
import json, os, sys, time
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import equity_monitor.data as data
import equity_monitor.rs as rs
from equity_monitor import model
from equity_monitor.model import score, signal

LEDGER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ledger")

def build():
    if not data.FMP_KEY:
        print("FMP_API_KEY not set — emitting empty ledger (terminal shows no rows).")
        write_ledger([], 0.0)
        return

    symbols = data.universe()
    feats, skipped = [], []
    for sym in symbols:
        try:
            f = data.build_features(sym)
            if f:
                feats.append(f)
            else:
                skipped.append(sym)
        except Exception as e:
            skipped.append(f"{sym}:{e}")
        time.sleep(0.05)  # gentle on rate limits

    feats = rs.enrich(feats)
    rows = []
    for f in feats:
        q, c, r, conv, sig, comp = score(f)
        rows.append({
            "symbol": f["symbol"],
            "price": round(f["price"], 2),
            "chg": round(f.get("chg24", 0.0), 2),
            "turnover": round(f.get("turnover", 0.0), 5),
            "adv": int(f.get("adv", 0)),
            "mcap": int(f.get("market_cap", 0)),
            "z_ath": None,
            "conviction": conv,
            "signal": sig,
            "factors": {
                "quality": comp["quality"],
                "confirmation": comp["confirmation"],
                "risk": comp["risk"],
                "rs_blend": comp["rs_blend"],
                "liquidity_fit": comp["liquidity_fit"],
                "val_zscore": comp["val_zscore"],
                "short_days": comp.get("short_days", 0.0),
            },
        })
    rows.sort(key=lambda x: x["conviction"], reverse=True)
    mcapsum = sum(r["mcap"] for r in rows)
    write_ledger(rows, mcapsum, skipped)
    print(f"built {len(rows)} rows, skipped {len(skipped)}: {skipped[:5]}")

def write_ledger(rows, mcapsum, skipped=None):
    os.makedirs(LEDGER, exist_ok=True)
    payload = {
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark": "SPY",
        "universe": len(rows),
        "mcapsum": float(mcapsum),
        "skipped": skipped or [],
        "all": rows,
        "top10": rows[:10],
    }
    with open(os.path.join(LEDGER, "index.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

if __name__ == "__main__":
    build()
