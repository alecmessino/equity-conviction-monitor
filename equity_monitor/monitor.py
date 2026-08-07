"""Model monitoring — the terminal's report on its own condition.

This measures whether the board is **operationally sound**, not whether it is right.
Those are different questions and conflating them would be the most damaging thing this
module could do. A model can be perfectly stable, fully covered, beautifully dispersed
and still predict nothing whatsoever; a green health panel is a statement that the
machinery is working, not that the signal has value. Whether the scores predict returns
is an Information Coefficient question, it needs months of accumulated snapshots, and
nothing here should be read as an early answer to it.

What it does measure:

* **Stability** — how much the ranking churns between runs. A model whose ordering
  reshuffles every night is fitting noise regardless of how good each night looks.
* **Coverage** — how much of the input set is actually observed rather than imputed,
  and whether that is drifting.
* **Regime** — the observable macro state, recorded so that factor behaviour can later
  be segmented by it. Observation only: nothing here feeds back into a score.
* **Health** — a set of pass/warn/fail checks over the above.

Every metric reports its sample size, and anything that cannot yet be computed says so
rather than returning a default that would render as a real reading.
"""
from __future__ import annotations

import math
import statistics as st
from datetime import datetime, timezone

from . import model, snapshots

# --- health thresholds. Each corresponds to a failure this project actually had. ---
FRESH_HOURS = 40.0            # a weekday build that has not run in 40h has missed one
MIN_DISPERSION = 5.0          # v2 shipped ~0.2 dispersion and nothing noticed
MIN_TIERS = 3
MIN_MEAN_CONFIDENCE = 0.70
MAX_MEAN_ABS_MOVE = 8.0       # a board that moves 8+ points a night is not measuring
MIN_RANK_CORRELATION = 0.90   # nor is one whose ordering reshuffles overnight
MAX_PRICE_FAILURE_RATE = 0.05


# ---------------------------------------------------------------------------
# stability
# ---------------------------------------------------------------------------
def _spearman(a: list[float], b: list[float]) -> float | None:
    """Rank correlation, implemented directly to avoid a scipy dependency.

    Ties are averaged, which matters here: conviction is an integer score over ~1,000
    names, so ties are common and ranking them arbitrarily would understate agreement.
    """
    n = len(a)
    if n < 3:
        return None

    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    ma, mb = st.fmean(ra), st.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return (num / den) if den > 0 else None


def stability(previous: dict, current: dict) -> dict | None:
    """How much the board changed between two snapshots."""
    if not previous or not current:
        return None
    shared = sorted(set(previous["rows"]) & set(current["rows"]))
    if len(shared) < 3:
        return None

    before = [previous["rows"][s]["conviction"] for s in shared]
    after = [current["rows"][s]["conviction"] for s in shared]
    moves = [abs(x - y) for x, y in zip(before, after)]

    # Tier migration. The counts matter more than the mean move: a board can have a
    # small average change while quietly reclassifying half its BUY list.
    tiers = ["STRONG", "BUY", "HOLD", "WATCH", "AVOID"]
    migration = {t: {u: 0 for u in tiers} for t in tiers}
    for s in shared:
        migration[model.signal(previous["rows"][s]["conviction"])][
            model.signal(current["rows"][s]["conviction"])] += 1
    changed = sum(migration[t][u] for t in tiers for u in tiers if t != u)

    return {
        "from": previous["date"],
        "to": current["date"],
        "names_compared": len(shared),
        "entered": sorted(set(current["rows"]) - set(previous["rows"]))[:20],
        "left": sorted(set(previous["rows"]) - set(current["rows"]))[:20],
        "rank_correlation": round(_spearman(before, after) or 0.0, 4),
        "mean_abs_move": round(st.fmean(moves), 3),
        "median_abs_move": round(st.median(moves), 3),
        "max_abs_move": max(moves),
        "unchanged": sum(1 for m in moves if m == 0),
        "tier_changes": changed,
        "tier_migration": migration,
        "spec_changed": previous.get("spec_hash") != current.get("spec_hash"),
    }


# ---------------------------------------------------------------------------
# coverage over time
# ---------------------------------------------------------------------------
def coverage_trend(ledger_dir: str, limit: int = 30) -> dict:
    """Share of names with each percentile observed rather than imputed, per date.

    Imputation is recorded per name as the percentile being absent before substitution,
    but the snapshot stores the substituted value, so absence is not directly visible
    after the fact. What is visible and equally diagnostic is ``data_confidence``,
    which is exactly the observed fraction at write time.
    """
    dates = snapshots.available(ledger_dir)[-limit:]
    series: list[dict] = []
    for stamp in dates:
        try:
            snap = snapshots.read(snapshots.snapshot_path(ledger_dir, stamp))
        except Exception:
            continue
        confs = [r.get("data_confidence") for r in snap["rows"].values()
                 if r.get("data_confidence") is not None]
        if not confs:
            continue
        by_sector: dict[str, list[float]] = {}
        for sym, r in snap["rows"].items():
            c = r.get("data_confidence")
            if c is not None:
                by_sector.setdefault(snap["sectors"].get(sym, "") or "?", []).append(c)
        series.append({
            "date": snap["date"],
            "names": len(snap["rows"]),
            "mean_confidence": round(st.fmean(confs), 4),
            "fully_measured": round(sum(1 for c in confs if c >= 0.999) / len(confs), 4),
            "spec_hash": snap.get("spec_hash"),
            "by_sector": {k: round(st.fmean(v), 4) for k, v in sorted(by_sector.items())},
        })
    return {"observations": len(series), "series": series}


def spec_consistency(ledger_dir: str, limit: int = 250) -> dict:
    """Which specification produced each stretch of history.

    A series spanning two spec hashes cannot be regressed as one dataset. Surfacing the
    boundaries now means the Phase 3 analysis starts from a segmented series rather than
    discovering the discontinuity in its results.
    """
    dates = snapshots.available(ledger_dir)[-limit:]
    runs: list[dict] = []
    for stamp in dates:
        try:
            snap = snapshots.read(snapshots.snapshot_path(ledger_dir, stamp))
        except Exception:
            continue
        h = snap.get("spec_hash")
        if runs and runs[-1]["spec_hash"] == h:
            runs[-1]["to"] = snap["date"]
            runs[-1]["days"] += 1
        else:
            runs.append({"spec_hash": h, "model_version": snap.get("model_version"),
                         "from": snap["date"], "to": snap["date"], "days": 1})
    return {"segments": runs, "current": runs[-1]["spec_hash"] if runs else None}


# ---------------------------------------------------------------------------
# regime observation
# ---------------------------------------------------------------------------
def regime(macro: dict | None) -> dict:
    """Describe the observable macro state. Observation, never prediction.

    Deliberately reports four independent readings rather than one cycle label.
    "Late-cycle" is an interpretation that would be wrong often enough to matter, and
    once printed it would be read as a forecast. These four are measurements.

    Nothing here feeds back into a score. It is recorded so that, once enough history
    exists, factor behaviour can be segmented by the conditions it occurred in.
    """
    if not macro:
        return {"available": False}

    def level(series_id: str) -> float | None:
        node = macro.get(series_id)
        return node.get("value") if node else None

    def change(series_id: str) -> float | None:
        node = macro.get(series_id)
        return node.get("change_1m") if node else None

    ten_year, curve = level("DGS10"), level("T10Y2Y")
    vix, credit = level("VIXCLS"), level("BAMLH0A0HYM2")
    rate_move = change("DGS10")

    states = {}
    if ten_year is not None and rate_move is not None:
        states["rates"] = ("rising" if rate_move > 0.10 else
                           "falling" if rate_move < -0.10 else "stable")
    if curve is not None:
        states["curve"] = ("inverted" if curve < 0 else
                           "flat" if curve < 0.5 else "steep")
    if vix is not None:
        states["volatility"] = ("calm" if vix < 15 else
                                "normal" if vix < 25 else "stressed")
    if credit is not None:
        states["credit"] = ("tight" if credit < 3.0 else
                            "normal" if credit < 5.0 else "wide")

    # A single composite, and only from the two risk-appetite readings, which are the
    # two that genuinely co-move. Labelled as a description of current conditions.
    #
    # Credit carries the most weight: high-yield spreads are the cleaner appetite gauge,
    # and requiring VIX to be outright "calm" would read a 15.8 VIX against 2.75% spreads
    # as merely "mixed", which understates an obviously accommodating tape.
    appetite = None
    if "volatility" in states and "credit" in states:
        stressed = states["volatility"] == "stressed" or states["credit"] == "wide"
        seeking = (states["credit"] == "tight"
                   and states["volatility"] in ("calm", "normal"))
        appetite = "risk-averse" if stressed else "risk-seeking" if seeking else "mixed"

    return {
        "available": True,
        "states": states,
        "risk_appetite": appetite,
        "levels": {"ten_year": ten_year, "curve": curve, "vix": vix, "credit": credit},
        "note": "Observed conditions, not a forecast, and not an input to any score.",
    }


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------
def _check(name: str, status: str, detail: str, value=None) -> dict:
    return {"name": name, "status": status, "detail": detail, "value": value}


def health(payload: dict, stab: dict | None, cov: dict,
           now: datetime | None = None) -> list[dict]:
    """Pass/warn/fail checks over operational condition.

    Not a judgement on whether the scores are any good — see the module docstring.
    """
    checks: list[dict] = []
    rows = payload.get("all") or []
    convictions = [r["conviction"] for r in rows if r.get("conviction") is not None]

    # freshness
    try:
        built = datetime.strptime(payload["as_of"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc)
        age = ((now or datetime.now(timezone.utc)) - built).total_seconds() / 3600.0
        checks.append(_check(
            "Data freshness",
            "pass" if age <= FRESH_HOURS else "fail",
            f"built {age:.0f}h ago" + ("" if age <= FRESH_HOURS
                                       else f"; a scheduled run has been missed"),
            round(age, 1)))
    except Exception:
        checks.append(_check("Data freshness", "fail", "no readable as_of timestamp"))

    # dispersion
    if convictions:
        sd = st.pstdev(convictions)
        checks.append(_check(
            "Score dispersion", "pass" if sd >= MIN_DISPERSION else "fail",
            f"σ = {sd:.1f} across {len(convictions)} names"
            + ("" if sd >= MIN_DISPERSION else "; the model is not discriminating"),
            round(sd, 2)))

        tiers = {r.get("signal") for r in rows if r.get("signal")}
        checks.append(_check(
            "Signal tiers populated", "pass" if len(tiers) >= MIN_TIERS else "fail",
            f"{len(tiers)} of 5 tiers in use", len(tiers)))

    # coverage
    confs = [r.get("data_confidence") for r in rows if r.get("data_confidence") is not None]
    if confs:
        mean_conf = st.fmean(confs)
        checks.append(_check(
            "Input coverage", "pass" if mean_conf >= MIN_MEAN_CONFIDENCE else "warn",
            f"{mean_conf:.0%} of scored inputs observed rather than imputed",
            round(mean_conf, 4)))

    # price sourcing
    fails = payload.get("price_failures") or []
    total = (payload.get("universe") or len(rows)) + len(fails)
    if total:
        rate = len(fails) / total
        checks.append(_check(
            "Price sourcing", "pass" if rate <= MAX_PRICE_FAILURE_RATE else "warn",
            f"{len(fails)} of {total} symbols unavailable", round(rate, 4)))

    # stability — only once there is something to compare against
    if stab:
        rho = stab["rank_correlation"]
        checks.append(_check(
            "Ranking stability", "pass" if rho >= MIN_RANK_CORRELATION else "warn",
            f"rank correlation {rho:.3f} vs {stab['from']}"
            + ("" if rho >= MIN_RANK_CORRELATION
               else "; the ordering is reshuffling faster than the inputs should allow"),
            rho))
        checks.append(_check(
            "Overnight move size", "pass" if stab["mean_abs_move"] <= MAX_MEAN_ABS_MOVE else "warn",
            f"mean |Δ| = {stab['mean_abs_move']:.1f} points, "
            f"{stab['tier_changes']} names changed tier",
            stab["mean_abs_move"]))
        if stab["spec_changed"]:
            checks.append(_check(
                "Specification continuity", "warn",
                "the scoring specification changed between these two runs, so the "
                "comparison spans two different models"))
    else:
        checks.append(_check(
            "Ranking stability", "pending",
            "needs a second snapshot; nothing to compare against yet"))

    # spec consistency across the whole recorded history
    segments = (cov.get("spec_segments") or {}).get("segments") or []
    if len(segments) > 1:
        checks.append(_check(
            "Specification history", "warn",
            f"history spans {len(segments)} specifications — analysis must segment by "
            f"spec_hash rather than treat it as one series", len(segments)))
    elif segments:
        checks.append(_check(
            "Specification history", "pass",
            f"one specification across {segments[0]['days']} recorded day(s)",
            segments[0]["days"]))

    return checks


def build(ledger_dir: str, payload: dict, macro: dict | None = None,
          today: str | None = None) -> dict:
    """Assemble the full monitoring report written alongside the ledger."""
    dates = snapshots.available(ledger_dir)
    current = snapshots.latest(ledger_dir)
    previous = snapshots.latest(ledger_dir, before=(current or {}).get("date"))
    stab = stability(previous, current) if (previous and current) else None

    cov = coverage_trend(ledger_dir)
    cov["spec_segments"] = spec_consistency(ledger_dir)

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": model.MODEL_VERSION,
        "spec_hash": model.spec_hash(),
        "observations": len(dates),
        "first_observation": dates[0] if dates else None,
        "stability": stab,
        "coverage": cov,
        "regime": regime(macro),
        "health": health(payload, stab, cov),
        "scope": ("Operational condition only. Whether these scores predict returns is "
                  "a separate question requiring months of accumulated history."),
    }
