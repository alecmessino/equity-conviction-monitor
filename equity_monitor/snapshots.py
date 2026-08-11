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

* **Attribution is exact, and still reports a residual.** Conviction is a cube root of
  a product, so its logarithm is additive: ``d ln(conviction) = (d ln Q + d ln C +
  d ln R)/3``. Working in that space makes each pillar's share of a move exact rather
  than a linearisation, and within a pillar the percentile mapping is linear, so the
  factor shares are exact too. The residual is reported anyway — it should be zero, and
  a non-zero value means an assumption broke.
"""
from __future__ import annotations

import glob
import json
import math
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


PILLAR_RANGE = {"quality": ("q_raw", model.Q_SPAN),
                "confirmation": ("c_raw", model.C_SPAN),
                "risk": ("r_raw", model.R_SPAN)}


def pillar_factors_for(profile: str) -> dict[str, list[str]]:
    quality = model.QUALITY_PROFILES.get(profile or "default",
                                         model.QUALITY_PROFILES["default"])
    return {**PILLAR_FACTORS, "quality": list(quality)}

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
          as_of: str = "", benchmark: dict | None = None,
          session: dict | None = None) -> str:
    """Persist one dated snapshot of the full factor decomposition.

    Columnar rather than a list of objects: at ~1,000 names the repeated key names
    would be roughly three quarters of the file, and these are committed to git every
    trading day.

    ``benchmark`` records the reference index's close *on the night the board was
    published*. It is stored here rather than read back from the OHLCV cache because
    that cache lags — it held 2026-08-05 on the night of the 2026-08-08 run — and a
    return series that silently substitutes a nearby date for a missing one is
    comparing the book against a different holding period than the one it ran.

    ``session`` is :func:`prices.session_date` output — the trading day the closes in
    this snapshot actually come from. The filename records when the run happened, which
    is a different fact: a Friday run can be reading Thursday's bar, and a run triggered
    on a Saturday reads Friday's. Both are stored, because the file date is what the
    ledger is keyed on and the session date is what a return leg must be measured
    between. ``session_date`` is the flat field readers use; the rest is the evidence
    behind it, kept so a fragmented feed can be seen rather than inferred.
    """
    stamp = (on or date.today())
    path = snapshot_path(ledger_dir, stamp)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scored = [r for r in rows if r.get("conviction") is not None]
    payload = {
        "date": stamp if isinstance(stamp, str) else stamp.isoformat(),
        "as_of": as_of,
        "session_date": (session or {}).get("session"),
        "session": session or None,
        "model_version": model.MODEL_VERSION,
        "spec_hash": model.spec_hash(),
        "columns": COLUMNS,
        "sectors": {r["symbol"]: r.get("sector", "") for r in scored},
        "profiles": {r["symbol"]: r.get("profile", "default") for r in scored},
        "data": {r["symbol"]: _encode(r) for r in scored},
    }
    if benchmark and benchmark.get("price"):
        payload["benchmark"] = {"symbol": benchmark.get("symbol", ""),
                                "price": round(float(benchmark["price"]), 4)}
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
        # The trading day the closes came from. None on snapshots written before this
        # was recorded — callers must fall back to `date` and say that they did, rather
        # than treat the run date as if it were the session.
        "session_date": payload.get("session_date"),
        "session": payload.get("session"),
        "as_of": payload.get("as_of"),
        "model_version": payload.get("model_version"),
        "spec_hash": payload.get("spec_hash"),
        "sectors": payload.get("sectors", {}),
        "profiles": payload.get("profiles", {}),
        "benchmark": payload.get("benchmark"),
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
def exact_conviction(row: dict) -> float | None:
    """Unrounded conviction, reconstructed from the stored pillar values.

    Snapshots persist ``conviction`` as an integer, so differencing it mixes the real
    move with up to a full point of rounding. q_raw/c_raw/r_raw and the uplift are all
    stored, and score() is a pure function of them, so the exact value is recoverable
    and the residual can be made to measure only the non-linearity it is meant to.
    """
    q_raw, c_raw, r_raw = row.get("q_raw"), row.get("c_raw"), row.get("r_raw")
    if q_raw is None or c_raw is None or r_raw is None:
        return None
    Q = model.Q_FLOOR + model.Q_SPAN * q_raw
    R = model.R_FLOOR + model.R_SPAN * r_raw
    C = min(model.C_CEILING,
            (model.C_FLOOR + model.C_SPAN * c_raw) * (row.get("mr_uplift") or 1.0))
    return 100.0 * (Q * C * R) ** (1.0 / 3.0)


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
    exact_before, exact_after = exact_conviction(previous), exact_conviction(current)
    if exact_before is None or exact_after is None or exact_before <= 0 or exact_after <= 0:
        return None
    exact_total = exact_after - exact_before

    profile = current.get("profile") or previous.get("profile") or "default"
    per_pillar = pillar_factors_for(profile)
    weights_by_pillar = model.weights_for(
        profile if profile in model.QUALITY_PROFILES else "")

    # Exact decomposition, not a linearisation.
    #
    #   conviction = 100 * (Q*C*R)^(1/3)
    #   =>  d ln(conviction) = (1/3) * ( d ln Q + d ln C + d ln R )
    #
    # The log form is additive *exactly*, so each pillar's share of the total move is
    # its share of d ln(conviction) with no approximation. Converting those shares
    # into points against the real move makes the pillar contributions sum to the move
    # by construction. A first-order expansion about the midpoint was accurate for
    # small moves but drifted badly on large ones — median residual 2.2 points on a
    # 2-4 point move — and a decomposition that stops adding up exactly where the
    # interesting moves are is not much of a decomposition.
    pillars: dict[str, dict] = {}
    factors: dict[str, float] = {}

    floors = {"quality": model.Q_FLOOR, "confirmation": model.C_FLOOR,
              "risk": model.R_FLOOR}
    log_share: dict[str, float] = {}
    pillar_delta: dict[str, tuple[float, float, float]] = {}

    for pillar, keys in per_pillar.items():
        raw_key, span = PILLAR_RANGE[pillar]
        prev_raw, cur_raw = previous.get(raw_key), current.get(raw_key)
        if prev_raw is None or cur_raw is None:
            continue
        floor = floors[pillar]
        prev_p = floor + span * prev_raw
        cur_p = floor + span * cur_raw
        if pillar == "confirmation":
            prev_p = min(model.C_CEILING, prev_p * (previous.get("mr_uplift") or 1.0))
            cur_p = min(model.C_CEILING, cur_p * (current.get("mr_uplift") or 1.0))
        if prev_p <= 0 or cur_p <= 0:
            continue
        log_share[pillar] = math.log(cur_p / prev_p) / 3.0
        pillar_delta[pillar] = (prev_raw, cur_raw, cur_p - prev_p)

    total_log = sum(log_share.values())
    allocated = 0.0
    for pillar, share in log_share.items():
        prev_raw, cur_raw, _ = pillar_delta[pillar]
        points = exact_total * (share / total_log) if abs(total_log) > 1e-12 else 0.0
        allocated += points
        pillars[pillar] = {"from": round(prev_raw, 4), "to": round(cur_raw, 4),
                           "points": round(points, 3)}

        # Within a pillar the mapping is linear, so weight * delta_percentile is each
        # factor's exact share of the pillar's move and the shares sum to it.
        weights = weights_by_pillar[pillar]
        deltas = {}
        for k in per_pillar[pillar]:
            a, b = previous.get(k), current.get(k)
            deltas[k] = weights.get(k, 0.0) * (b - a) if (a is not None and b is not None) else 0.0
        net = sum(deltas.values())
        for k, dv in deltas.items():
            factors[k] = points * (dv / net) if abs(net) > 1e-12 else 0.0

    ordered = {k: round(v, 3) for k, v in
               sorted(factors.items(), key=lambda kv: -abs(kv[1]))}
    # The two entries that explain the move: the largest push in each direction.
    # Taking the top two by magnitude instead produced a label reading "+5, driven by
    # -0.4 trend and -0.4 relative strength" on 5% of names — two true numbers
    # arranged into a false summary.
    positives = [(k, v) for k, v in ordered.items() if v > 0.01]
    negatives = [(k, v) for k, v in ordered.items() if v < -0.01]
    headline = []
    if positives:
        headline.append(positives[0])
    if negatives:
        headline.append(negatives[0])
    if len(headline) == 1:
        pool = positives if positives else negatives
        headline = pool[:2]

    return {
        "total": total,
        "profile": profile,
        "from": previous["conviction"],
        "to": current["conviction"],
        "exact_total": round(exact_total, 3),
        "pillars": pillars,
        "factors": ordered,
        "headline": [[k, v] for k, v in headline],
        # Exact by construction, so this only ever reflects float noise. Measured
        # against the unrounded allocation — differencing the rounded, published
        # points would report the display precision as if it were model error.
        # Rounding of the displayed integer score is reported separately.
        "residual": round(exact_total - allocated, 6),
        "rounding": round(total - exact_total, 3),
    }


def attribute_all(ledger_dir: str, current_rows: list[dict],
                  before: str | None = None, today: str | None = None) -> dict:
    """Attribution for every name against the most recent snapshot from a PRIOR date.

    ``before`` defaults to today, which matters: the build overwrites the current
    date's snapshot, so comparing against ``latest()`` unguarded compares this run to
    an earlier run on the same day. That produces a real-looking "since 2026-08-07"
    attribution describing nothing but intraday rebuild noise — precisely the sort of
    plausible-but-meaningless output this project exists to avoid.
    """
    cutoff = before or today or date.today().isoformat()
    prior = latest(ledger_dir, before=cutoff)
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
