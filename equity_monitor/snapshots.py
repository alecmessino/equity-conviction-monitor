"""Nightly factor-level snapshots — the research dataset.

Saving only ``conviction = 82`` throws away almost everything that makes the number
worth having. Six months on it tells you Apple went 81 -> 84 and nothing else. Saving
the whole decomposition tells you the move was driven entirely by improving ROIC while
valuation deteriorated, which is a different and far more useful statement.

It is also the precondition for validating the model at all. Information Coefficient,
decile spreads, hit rates and factor decay are all regressions of *past factor values*
against *forward returns*. None of them can be run retroactively — the data either was
recorded each night or it was not.

Two properties this file is careful about:

* **Every snapshot records the spec version and hash that produced it.** An IC computed
  across a weight change is a number about two different models. Segmenting by
  ``spec_hash`` is what keeps the series honest when the specification eventually moves.

* **Attribution reports its own residual.** The decomposition below is exact for the
  pillars (which are linear in the percentiles) and first-order for the conviction
  score (which is not, being a cube root of their product). Rather than quietly
  distributing the approximation error across the factors, the leftover is returned as
  ``residual`` so the reader can see how much of the move the decomposition explains.
"""
from __future__ import annotations

import glob
import json
import os
from datetime import date

from . import model

# Column order for the columnar encoding. Append-only: adding a column at the end
# keeps every previously written snapshot readable, whereas inserting one in the
# middle would silently reinterpret history.
COLUMNS: list[str] = [
    "conviction", "q_raw", "c_raw", "r_raw", "mr_uplift",
    "p_roic", "p_fcf_yield", "p_gross_margin", "p_leverage", "p_earnings_stability",
    "p_rs", "p_trend", "p_liquidity", "p_value", "p_lowvol",
    "data_confidence", "price", "market_cap", "weight",
    # Appended when sector profiles landed (model v3.1.0). Appending rather than
    # inserting keeps every snapshot written before that date readable with the
    # same decoder.
    "p_roe", "p_capital", "p_cash_yield", "p_ffo_yield", "p_leverage_assets",
]

# Percentile columns per pillar. Quality depends on the row's sector profile, so it
# is resolved per row rather than fixed: attributing a bank's move to p_roic — an
# input its profile never reads — would be a confident and entirely fictitious answer.
PILLAR_FACTORS = {
    "quality": list(model.WEIGHTS["quality"]),
    "confirmation": list(model.WEIGHTS["confirmation"]),
    "risk": list(model.WEIGHTS["risk"]),
}


def pillar_factors_for(profile: str) -> dict[str, list[str]]:
    quality = model.QUALITY_PROFILES.get(profile or "default",
                                         model.QUALITY_PROFILES["default"])
    return {**PILLAR_FACTORS, "quality": list(quality)}
PILLAR_RANGE = {"quality": ("q_raw", model.Q_SPAN), "confirmation": ("c_raw", model.C_SPAN),
                "risk": ("r_raw", model.R_SPAN)}

_PRECISION = {"conviction": 0, "price": 4, "market_cap": 0, "weight": 3}
_DEFAULT_PRECISION = 4


def _encode(row: dict) -> list:
    out = []
    for col in COLUMNS:
        v = row.get(col)
        if v is None:
            out.append(None)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            places = _PRECISION.get(col, _DEFAULT_PRECISION)
            out.append(round(v, places) if places else round(v))
        else:
            out.append(v)
    return out


def snapshot_path(ledger_dir: str, on: date | str) -> str:
    stamp = on if isinstance(on, str) else on.isoformat()
    return os.path.join(ledger_dir, "snapshots", f"{stamp}.json")


def write(rows: list[dict], ledger_dir: str, on: date | str | None = None,
          as_of: str = "") -> str:
    """Persist one dated snapshot of the full factor decomposition.

    Columnar rather than a list of objects: at ~1,000 names the repeated key names
    would be roughly three quarters of the file, and these are committed to git every
    trading day.
    """
    stamp = (on or date.today())
    path = snapshot_path(ledger_dir, stamp)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scored = [r for r in rows if r.get("conviction") is not None]
    payload = {
        "date": stamp if isinstance(stamp, str) else stamp.isoformat(),
        "as_of": as_of,
        "model_version": model.MODEL_VERSION,
        "spec_hash": model.spec_hash(),
        "columns": COLUMNS,
        "sectors": {r["symbol"]: r.get("sector", "") for r in scored},
        "profiles": {r["symbol"]: r.get("profile", "default") for r in scored},
        "data": {r["symbol"]: _encode(r) for r in scored},
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    os.replace(tmp, path)
    return path


def read(path: str) -> dict:
    """A snapshot as {symbol: {column: value}}, plus its metadata."""
    with open(path) as fh:
        payload = json.load(fh)
    cols = payload.get("columns", COLUMNS)
    return {
        "date": payload.get("date"),
        "model_version": payload.get("model_version"),
        "spec_hash": payload.get("spec_hash"),
        "sectors": payload.get("sectors", {}),
        "profiles": payload.get("profiles", {}),
        "rows": {sym: dict(zip(cols, vals), profile=(payload.get("profiles") or {}).get(sym, "default"))
                 for sym, vals in (payload.get("data") or {}).items()},
    }


def available(ledger_dir: str) -> list[str]:
    """Snapshot dates present on disk, oldest first."""
    pattern = os.path.join(ledger_dir, "snapshots", "*.json")
    return sorted(os.path.basename(p)[:-5] for p in glob.glob(pattern))


def latest(ledger_dir: str, before: str | None = None) -> dict | None:
    """Most recent snapshot, optionally strictly before a given date."""
    dates = [d for d in available(ledger_dir) if not before or d < before]
    if not dates:
        return None
    return read(snapshot_path(ledger_dir, dates[-1]))


def series(ledger_dir: str, symbol: str, limit: int = 120) -> list[dict]:
    """Per-symbol factor time series, oldest first.

    Reads whole snapshots, so it is a research-side helper rather than something the
    browser should call. The terminal is served a pre-built trend file instead.
    """
    out = []
    for stamp in available(ledger_dir)[-limit:]:
        try:
            snap = read(snapshot_path(ledger_dir, stamp))
        except Exception:
            continue
        row = snap["rows"].get(symbol)
        if row:
            out.append({"date": snap["date"], "spec_hash": snap["spec_hash"], **row})
    return out


def build_trends(ledger_dir: str, keys: tuple[str, ...] = ("conviction",),
                 limit: int = 90) -> dict:
    """Compact per-symbol series for the terminal, assembled from the snapshots.

    Conviction only, and integers at that. Carrying q/c/r here as well would put the
    file around 2.5 MB at ninety days — as large as the whole ledger — for data the
    grid does not draw. The detail panel fetches one name's full decomposition from
    ``ledger/factors/SYM.json`` instead, which is the same lazy pattern the price
    history already uses.
    """
    dates = available(ledger_dir)[-limit:]
    trends: dict[str, dict[str, list]] = {}
    for stamp in dates:
        try:
            snap = read(snapshot_path(ledger_dir, stamp))
        except Exception:
            continue
        for sym, row in snap["rows"].items():
            slot = trends.setdefault(sym, {k: [] for k in keys})
            for k in keys:
                v = row.get(k)
                slot[k].append(round(v, 4) if isinstance(v, (int, float)) else None)
    return {"dates": dates, "keys": list(keys), "series": trends}


def write_symbol_factors(ledger_dir: str, limit: int = 250) -> int:
    """Per-symbol factor history, one file each, for lazy loading by the detail panel.

    Not committed — regenerated every run and shipped in the Pages artifact, exactly
    like the per-symbol price history.
    """
    dates = available(ledger_dir)[-limit:]
    if not dates:
        return 0
    per_symbol: dict[str, dict] = {}
    for stamp in dates:
        try:
            snap = read(snapshot_path(ledger_dir, stamp))
        except Exception:
            continue
        for sym, row in snap["rows"].items():
            slot = per_symbol.setdefault(sym, {"symbol": sym, "dates": [], "rows": []})
            slot["dates"].append(snap["date"])
            slot["rows"].append([row.get(c) for c in COLUMNS])

    out_dir = os.path.join(ledger_dir, "factors")
    os.makedirs(out_dir, exist_ok=True)
    for sym, payload in per_symbol.items():
        payload["columns"] = COLUMNS
        safe = sym.replace("/", "-")
        with open(os.path.join(out_dir, f"{safe}.json"), "w") as fh:
            json.dump(payload, fh, separators=(",", ":"))
    return len(per_symbol)


# ---------------------------------------------------------------------------
# score-change attribution
# ---------------------------------------------------------------------------
def attribute(previous: dict, current: dict) -> dict | None:
    """Decompose a conviction change into per-factor contributions.

    ``previous`` and ``current`` are single-symbol rows as returned by ``read()``.

    Method. Each pillar is linear in its percentiles:

        Q = Q_FLOOR + Q_SPAN * sum_k(w_k * p_k)

    so a factor's exact contribution to the pillar's move is ``Q_SPAN * w_k * dp_k``.
    Conviction is ``100 * (Q*C*R)^(1/3)``, which is not linear, so the pillar moves are
    mapped through a first-order expansion around the midpoint:

        d_conviction ~= conviction_mid * (1/3) * (dQ/Q_mid + dC/C_mid + dR/R_mid)

    Whatever that expansion fails to explain is returned as ``residual`` rather than
    being spread silently across the factors. On a normal overnight move the residual
    is a small fraction of a point; a large residual is a signal that something other
    than the factors moved, and it should be visible.
    """
    if not previous or not current:
        return None
    if previous.get("conviction") is None or current.get("conviction") is None:
        return None

    total = current["conviction"] - previous["conviction"]
    pillars: dict[str, dict] = {}
    factors: dict[str, float] = {}
    approx_total = 0.0

    profile = current.get("profile") or previous.get("profile") or "default"
    per_pillar = pillar_factors_for(profile)
    weights_by_pillar = model.weights_for(
        profile if profile in model.QUALITY_PROFILES else "")

    for pillar, keys in per_pillar.items():
        raw_key, span = PILLAR_RANGE[pillar]
        weights = weights_by_pillar[pillar]
        prev_raw, cur_raw = previous.get(raw_key), current.get(raw_key)
        if prev_raw is None or cur_raw is None:
            continue

        floor = {"quality": model.Q_FLOOR, "confirmation": model.C_FLOOR,
                 "risk": model.R_FLOOR}[pillar]
        prev_pillar = floor + span * prev_raw
        cur_pillar = floor + span * cur_raw
        mid = (prev_pillar + cur_pillar) / 2.0
        if mid <= 0:
            continue

        mid_conv = (previous["conviction"] + current["conviction"]) / 2.0
        scale = mid_conv * (1 / 3) / mid          # points per unit of pillar move
        pillar_points = scale * (cur_pillar - prev_pillar)
        approx_total += pillar_points

        # Each factor's contribution is computed directly, not split proportionally.
        # The pillar is linear in its percentiles, so span * w_k * dp_k is that
        # factor's exact share of the pillar's move, and the shares sum to it.
        # Splitting by |delta| instead would invert the sign of every factor whenever
        # the pillar fell, reporting a deteriorating input as a positive contribution.
        for k in keys:
            a, b = previous.get(k), current.get(k)
            if a is None or b is None:
                factors[k] = 0.0
                continue
            factors[k] = scale * span * weights.get(k, 0.0) * (b - a)

        pillars[pillar] = {
            "from": round(prev_raw, 4),
            "to": round(cur_raw, 4),
            "points": round(pillar_points, 3),
        }

    return {
        "total": total,
        "profile": profile,
        "from": previous["conviction"],
        "to": current["conviction"],
        "pillars": pillars,
        "factors": {k: round(v, 3) for k, v in sorted(
            factors.items(), key=lambda kv: -abs(kv[1]))},
        "residual": round(total - approx_total, 3),
    }


def attribute_all(ledger_dir: str, current_rows: list[dict],
                  before: str | None = None) -> dict:
    """Attribution for every name against the most recent prior snapshot."""
    prior = latest(ledger_dir, before=before)
    if not prior:
        return {}
    out = {}
    for row in current_rows:
        if row.get("conviction") is None:
            continue
        prev = prior["rows"].get(row["symbol"])
        if not prev:
            continue
        got = attribute(prev, row)
        if got and got["total"] != 0:
            out[row["symbol"]] = got
    return {"since": prior["date"], "spec_hash": prior["spec_hash"], "names": out}
