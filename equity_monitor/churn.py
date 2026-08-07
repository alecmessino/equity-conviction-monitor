"""Churn diagnosis: did the board move because the data moved, or because the model
amplified a small change?

``monitor.stability`` answers *how much* the ranking moved. It cannot answer *why*, and
the why is what determines the response. A rank correlation of 0.85 caused by genuine
input movement is the model working; the same 0.85 produced from inputs that barely
changed is the model manufacturing turnover, and acting on it would be trading noise.

**The method is counterfactual, not correlational.** The obvious approach — rank-correlate
each input and compare the aggregate against conviction's — does not survive contact with
real data. Inputs move in opposing directions, so a weighted mean of signed correlations
collapses toward zero and stops meaning anything; and an input with heavy model weight but
little cross-sectional dispersion barely steers the ranking despite dominating the average.
Tested against a case where the momentum block inverts — unambiguously information — that
aggregate reported amplification.

So instead the model itself is asked. For each input in turn, last night's board is
re-scored with *only that input* advanced to tonight's values, and the resulting
reordering is measured. That is an exact statement about the frozen specification, made
by evaluating it, and it decomposes cleanly:

* ``contribution[k]`` — reordering caused by input *k* alone.
* ``sum(contributions)`` — the reordering the inputs account for one at a time.
* ``total`` — the reordering that actually happened.
* ``amplification = total / sum(contributions)`` — near 1 the churn is the sum of its
  parts and the movement is information; well above 1 the inputs are interacting through
  the geometric mean to produce turnover no single input explains, which is sensitivity.

Two supporting tests run alongside it:

* **Boundary flapping.** Tiers are hard thresholds, so a one-point move across 70 turns a
  HOLD into a BUY without any meaningful change of view. Counting tier changes that came
  from marginal moves separates real reclassification from label noise.

* **Elasticity, with its mechanism.** ``d ln(conviction) = (d ln Q + d ln C + d ln R)/3``,
  so a pillar near its floor moves the score far harder per unit of input change than one
  near the middle. Q floors at 0.02, so a name pinned there amplifies about an order of
  magnitude more than a median name. Reporting conviction change per unit of input
  movement alongside each name's weakest pillar makes the mechanism nameable.

This module reports evidence. It does not recommend a specification change, and a single
overnight reading should not be treated as one — the same measurement sustained over
weeks is what would distinguish a structural property from one unusual night.
"""
from __future__ import annotations

import statistics as st

from . import model, snapshots
from .monitor import _spearman

# The board's ordering moving this many times further than its inputs did is the
# specification amplifying rather than transmitting. Deliberately generous: a percentile
# ranking is a compressed readout, so some gain over the raw input movement is expected
# even from a faithful model, and the question is whether it dominates.
AMPLIFICATION_RATIO = 3.0

# A tier change produced by a move this small is a threshold artifact, not a changed
# view. Two points, because conviction is published as an integer.
MARGINAL_MOVE = 2.0

# Below this, a name's inputs are unchanged and elasticity is not defined.
INPUT_EPSILON = 1e-9

# A board whose inputs barely moved has no churn to explain, and a ratio computed from
# two near-zero numbers is noise dressed as a finding.
MIN_INPUT_MOVEMENT = 1e-4


def _effective_weights(profile: str) -> dict[str, float]:
    """Each factor's share of ``d ln(conviction)``.

    Each pillar contributes exactly one third in log space, and within a pillar the
    factor shares are its weights, so a factor's effective weight is ``w/3``. Used to
    enumerate the inputs a given sector profile actually reads — attributing a bank's
    move to ``p_roic``, a column its profile never touches, would be a confident and
    entirely fictitious answer.
    """
    factors = snapshots.pillar_factors_for(profile)
    out: dict[str, float] = {}
    for pillar, keys in factors.items():
        weights = (model.QUALITY_PROFILES.get(profile or "default",
                                              model.QUALITY_PROFILES["default"])
                   if pillar == "quality" else model.WEIGHTS[pillar])
        for k in keys:
            out[k] = weights.get(k, 0.0) / 3.0
    return out


def _percentiles(row: dict, keys) -> dict:
    return {k: row.get(k) for k in keys if row.get(k) is not None}


def _conviction(percentiles: dict, profile: str, uplift: float | None) -> float | None:
    """Conviction from a percentile vector, at a stored mean-reversion uplift.

    ``model.score()`` derives the uplift from raw 52-week drawdown, which snapshots do
    not carry — they store the resulting multiplier instead. So the pillars come from
    ``score()`` (its weights, floors and spans, not a copy of them) and only the final
    published line is re-evaluated with the stored uplift applied.

    Holding the uplift fixed at its recorded value is deliberate: it is a function of
    price history rather than of any percentile, so varying it here would attribute a
    drawdown move to whichever input happened to be under test.
    """
    if not percentiles:
        return None
    weights = {**model.WEIGHTS,
               "quality": model.QUALITY_PROFILES.get(profile or "default",
                                                     model.QUALITY_PROFILES["default"])}
    s = model.score(percentiles, weights)
    c = min(model.C_CEILING, s["c"] * (uplift if uplift is not None else 1.0))
    return 100.0 * (s["q"] * c * s["r"]) ** (1.0 / 3.0)


def _rank01(values: list[float]) -> list[float]:
    """Values mapped to their percentile position in [0, 1], ties averaged."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        pos = ((i + j) / 2.0) / max(1, n - 1)
        for k in range(i, j + 1):
            out[order[k]] = pos
        i = j + 1
    return out


def movement(previous: dict, current: dict, shared: list[str]) -> dict:
    """How far the inputs moved, and how far the ordering moved, in the same units.

    Both sides are expressed in percentile points, which is what makes the comparison
    mean anything. Input movement is each name's weighted mean absolute percentile
    change, weighted by the factor's share of ``d ln(conviction)``. Output movement is
    the change in the name's own percentile position within the conviction ranking.

    Under a model that transmits, a one-point input move produces about a one-point move
    in standing. A ratio well above one is the specification moving the board further
    than its data did — which is the definition of the thing being tested for.

    Deliberately not a decomposition of rank correlation. Rank statistics respond
    non-linearly to how much *weight* moves rather than how far it moves, so a
    correlation-based ratio reports superadditivity whenever several correlated inputs
    move together, which is a property of the statistic and not of the model.
    """
    prev_rows, curr_rows = previous["rows"], current["rows"]
    usable, input_moves = [], []
    for sym in shared:
        a, b = prev_rows[sym], curr_rows[sym]
        weights = _effective_weights(b.get("profile") or "default")
        moved = 0.0
        for k, w in weights.items():
            x, y = a.get(k), b.get(k)
            if x is None or y is None:
                break
            moved += w * abs(x - y)
        else:
            usable.append(sym)
            input_moves.append(moved)

    if len(usable) < 10:
        return {"names": len(usable), "input": None, "output": None, "ratio": None}

    before = _rank01([prev_rows[s]["conviction"] for s in usable])
    after = _rank01([curr_rows[s]["conviction"] for s in usable])
    output_moves = [abs(x - y) for x, y in zip(before, after)]

    mean_in = st.fmean(input_moves)
    mean_out = st.fmean(output_moves)
    return {
        "names": len(usable),
        "input": round(mean_in, 5),
        "output": round(mean_out, 5),
        "ratio": round(mean_out / mean_in, 3) if mean_in > INPUT_EPSILON else None,
        "median_input": round(st.median(input_moves), 5),
        "median_output": round(st.median(output_moves), 5),
    }


def counterfactuals(previous: dict, current: dict, shared: list[str]) -> dict:
    """Reordering attributable to each input, by advancing one input at a time.

    Every name is re-scored under the frozen specification with last night's inputs
    except for the factor under test. Nothing about the model changes; it is evaluated,
    not modified.
    """
    prev_rows, curr_rows = previous["rows"], current["rows"]

    baseline: list[float] = []
    actual: list[float] = []
    per_factor: dict[str, list[float]] = {}
    factor_names: dict[str, int] = {}
    usable: list[str] = []

    for sym in shared:
        a, b = prev_rows[sym], curr_rows[sym]
        profile = b.get("profile") or "default"
        keys = list(_effective_weights(profile))
        p_prev = _percentiles(a, keys)
        p_curr = _percentiles(b, keys)
        if len(p_prev) != len(keys) or len(p_curr) != len(keys):
            continue          # an input the snapshot never recorded for this name
        base = _conviction(p_prev, profile, a.get("mr_uplift"))
        now = _conviction(p_curr, profile, b.get("mr_uplift"))
        if base is None or now is None:
            continue
        usable.append(sym)
        baseline.append(base)
        actual.append(now)
        for k in keys:
            one = dict(p_prev)
            one[k] = p_curr[k]
            per_factor.setdefault(k, []).append(
                _conviction(one, profile, a.get("mr_uplift")))
            factor_names[k] = factor_names.get(k, 0) + 1

    if len(usable) < 10:
        return {"names": len(usable), "total": None, "contributions": {}}

    def instability(after: list[float]) -> float | None:
        rho = _spearman(baseline, after)
        return None if rho is None else 1.0 - rho

    total = instability(actual)
    contributions: dict[str, dict] = {}
    for k, vals in sorted(per_factor.items()):
        if len(vals) != len(baseline):
            continue          # measured on a different name set; not comparable
        inst = instability(vals)
        if inst is None:
            continue
        contributions[k] = {"instability": round(inst, 5), "names": factor_names[k]}

    return {"names": len(usable), "total": round(total, 5) if total is not None else None,
            "contributions": contributions}


def boundary_flapping(previous: dict, current: dict, shared: list[str]) -> dict:
    """How many tier changes came from a move too small to mean anything."""
    prev_rows, curr_rows = previous["rows"], current["rows"]
    changed, marginal, examples = 0, 0, []
    for sym in shared:
        before, after = prev_rows[sym]["conviction"], curr_rows[sym]["conviction"]
        if model.signal(before) == model.signal(after):
            continue
        changed += 1
        move = abs(after - before)
        if move <= MARGINAL_MOVE:
            marginal += 1
            if len(examples) < 12:
                examples.append({
                    "symbol": sym, "from": before, "to": after,
                    "from_tier": model.signal(before), "to_tier": model.signal(after),
                })
    return {
        "tier_changes": changed,
        "marginal": marginal,
        "marginal_share": round(marginal / changed, 4) if changed else None,
        "threshold": MARGINAL_MOVE,
        "examples": examples,
    }


def elasticity(previous: dict, current: dict, shared: list[str],
               limit: int = 15) -> dict:
    """Conviction points moved per unit of total input movement, per name.

    High elasticity is not a defect on its own — it is what the geometric mean is *for*,
    since a collapsing pillar should move the score hard. It becomes a finding when the
    board's churn is concentrated in these names while the inputs behind them barely
    moved.
    """
    prev_rows, curr_rows = previous["rows"], current["rows"]
    scored: list[dict] = []
    inert = []  # conviction moved while every input held: not possible from percentiles

    for sym in shared:
        a, b = prev_rows[sym], curr_rows[sym]
        profile = b.get("profile") or "default"
        keys = _effective_weights(profile)
        move = 0.0
        for k in keys:
            x, y = a.get(k), b.get(k)
            if x is not None and y is not None:
                move += abs(x - y)
        d_conv = abs(b["conviction"] - a["conviction"])
        pillars = {"Q": b.get("q_raw"), "C": b.get("c_raw"), "R": b.get("r_raw")}
        named = [(v, k) for k, v in pillars.items() if v is not None]
        weakest = min(named) if named else (None, None)

        if move < INPUT_EPSILON:
            if d_conv >= 1 and len(inert) < 12:
                # The only remaining route is the mean-reversion uplift, which keys off
                # raw drawdown rather than a percentile. Recording it keeps the
                # accounting closed instead of leaving an unexplained residue.
                inert.append({
                    "symbol": sym, "move": d_conv,
                    "uplift_before": a.get("mr_uplift"),
                    "uplift_after": b.get("mr_uplift"),
                })
            continue

        scored.append({
            "symbol": sym,
            "elasticity": d_conv / move,
            "conviction_move": b["conviction"] - a["conviction"],
            "input_move": round(move, 4),
            "weakest_pillar": weakest[1],
            "weakest_pillar_value": round(weakest[0], 4) if weakest[0] is not None else None,
        })

    if not scored:
        return {"names": 0, "median": None, "amplifiers": [], "inert_movers": inert}

    values = sorted(x["elasticity"] for x in scored)
    median = st.median(values)
    top = sorted(scored, key=lambda x: -x["elasticity"])[:limit]
    for x in top:
        x["elasticity"] = round(x["elasticity"], 2)
        x["vs_median"] = round(x["elasticity"] / median, 1) if median > 0 else None

    return {
        "names": len(scored),
        "median": round(median, 3),
        "p90": round(values[int(0.9 * (len(values) - 1))], 3),
        "amplifiers": top,
        "inert_movers": inert,
    }


def diagnose(previous: dict, current: dict) -> dict | None:
    """Assemble the tests into evidence about the cause of overnight churn."""
    if not previous or not current:
        return None
    shared = sorted(set(previous["rows"]) & set(current["rows"]))
    if len(shared) < 10:
        return None

    before = [previous["rows"][s]["conviction"] for s in shared]
    after = [current["rows"][s]["conviction"] for s in shared]
    published_rho = _spearman(before, after)

    move = movement(previous, current, shared)
    ratio = move.get("ratio")

    # Which input drove the reordering, by re-scoring with one input advanced at a time.
    # Reported as shares of the reordering the inputs account for — the absolute scale is
    # not comparable across inputs, but the ordering of the shares is.
    cf = counterfactuals(previous, current, shared)
    contributions = cf.get("contributions") or {}
    explained = sum(v["instability"] for v in contributions.values())
    for v in contributions.values():
        v["share"] = round(v["instability"] / explained, 4) if explained > 0 else None
    # With nothing to explain, the top of a list of zeros is not a leading contributor —
    # naming one would be reporting an artifact of dict ordering as a finding.
    ranked = sorted(contributions.items(), key=lambda kv: -kv[1]["instability"])
    largest = ({"factor": ranked[0][0], **ranked[0][1]}
               if ranked and explained > 0 else None)

    if move.get("input") is None:
        assessment, basis = None, "not enough recorded inputs to compare movement"
    elif move["input"] < MIN_INPUT_MOVEMENT and move["output"] < MIN_INPUT_MOVEMENT:
        assessment = "quiet"
        basis = "neither the inputs nor the ordering moved; there is no churn to explain"
    elif ratio is None:
        assessment = "quiet"
        basis = ("the inputs did not move at all, so no ratio is defined"
                 + ("; the ordering did move, which should not be possible from "
                    "percentiles alone" if move["output"] > MIN_INPUT_MOVEMENT else ""))
    elif ratio >= AMPLIFICATION_RATIO:
        assessment = "amplified"
        basis = (f"standing in the ranking moved {ratio:.1f}x further than the inputs "
                 f"behind it — the board is being reordered by the specification rather "
                 f"than by the data")
    else:
        assessment = "tracks-inputs"
        basis = (f"standing moved {ratio:.1f}x the input movement, within what a "
                 f"percentile readout produces from a faithful model — the churn is "
                 f"information")

    return {
        "from": previous["date"],
        "to": current["date"],
        "names_compared": len(shared),
        "names_measured": move.get("names", 0),
        "published_rank_correlation": round(published_rho, 4) if published_rho else None,
        "input_movement": move.get("input"),
        "output_movement": move.get("output"),
        "median_input_movement": move.get("median_input"),
        "median_output_movement": move.get("median_output"),
        "amplification": ratio,
        "threshold": AMPLIFICATION_RATIO,
        "contributions": contributions,
        "largest_contributor": largest,
        "assessment": assessment,
        "basis": basis,
        "boundary": boundary_flapping(previous, current, shared),
        "elasticity": elasticity(previous, current, shared),
        "scope": ("Evidence about the cause of overnight churn, not a recommendation. "
                  "One night is not a finding; the same reading sustained over weeks "
                  "would be."),
    }


def from_ledger(ledger_dir: str) -> dict | None:
    """Diagnose the two most recent snapshots on disk.

    Lives here rather than in ``monitor.build`` because this module already imports
    ``monitor`` for the rank correlation, and the reverse import would close the loop.
    The nightly attaches the result to the monitoring report.
    """
    current = snapshots.latest(ledger_dir)
    if not current:
        return None
    previous = snapshots.latest(ledger_dir, before=current.get("date"))
    return diagnose(previous, current) if previous else None
