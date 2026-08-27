"""Descriptions of an already-scored board. Nothing here can move a score.

Every function takes rows that ``model.score_rows`` has already written to, and returns
a description of them. No function writes ``conviction``, ``signal``, ``q``, ``c``,
``r`` or any ``p_*`` key back onto a row, and none is consulted by the scorer. That
separation is the point: the board is a measurement, and this module measures the
measurement without disturbing it. ``tests/test_diagnostics.py`` asserts the invariant
directly — it scores a universe, runs every function here, and requires the rows to be
byte-identical afterwards.

Three things are described.

**Pillar influence.** Conviction is ``100·(Q·C·R)^(1/3)``, which weights the three
pillars equally. Ranking orders on ``ln Q + ln C + ln R`` — a sum in which each pillar
carries a coefficient of 1/3 — so a pillar's effective contribution to the ordering is
set by how much it *varies* across the universe, not by its coefficient. A pillar taking
nearly the same value for every name separates none of them, whatever weight it carries.
``pillar_influence`` reports the variance decomposition of that sum, which sums to
exactly 1, alongside two observable quantities associated with the split: how much spread
each pillar's blend retains, and the floor-and-span range it is mapped onto.

Both of those are deliberate model transforms — the floors are non-zero on purpose and
are hashed into ``model.spec()``, and the blend weights are the published specification.
Neither is an explicit *top-level* pillar weight, which is why the split cannot be read
off the weights alone; what the decomposition reports is the implicit influence produced
by the chosen transforms and the covariance observed on this cross-section. It is
descriptive, not causal, and says nothing about predictive value.

**Sector tilt.** Percentiles for margins, leverage and valuation are ranked within GICS
sector, so the model takes no deliberate sector view — but the pillars that are ranked
universe-wide (relative strength, trend, liquidity, volatility) can still concentrate
the top of the board in a handful of sectors. ``sector_tilt`` measures that
concentration. It is descriptive: nothing here neutralises it.

**Capped by missing data.** ``model.prepare`` substitutes the group median for any input
it could not observe, which is the conservative choice and the right one — but it means
a good business with a gap in its disclosure is scored partly on a median it did not
earn. ``capped_by_data`` asks a counterfactual: if this name's unobserved inputs looked
like the inputs we *can* see for it, what would it have scored? The gap is explanatory
only. It is never added to a score, and the names it surfaces are not re-ranked.
"""
from __future__ import annotations

import math

from . import model

# A share is a share of current cross-sectional log-score variance, so it needs a
# population big enough for a variance to mean anything. Below this the decomposition is
# suppressed rather than published with a number nobody should read.
MIN_POPULATION = 30

# "Otherwise strong" for the capped-by-data list: a name has to be plausibly investable
# before a missing input is worth explaining. Below this the gap is real but moot.
CAPPED_MIN_CONVICTION = 50
CAPPED_MIN_GAP = 2  # conviction points; smaller than this is rounding, not a story


def _mean(v: list[float]) -> float:
    return sum(v) / len(v) if v else 0.0


def _pstdev(v: list[float]) -> float:
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def _cov(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return 0.0
    ma, mb = _mean(a), _mean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a)


def _corr(a: list[float], b: list[float]) -> float:
    sa, sb = _pstdev(a), _pstdev(b)
    return (_cov(a, b) / (sa * sb)) if sa > 0 and sb > 0 else 0.0


def _spearman(a: list[float], b: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            tie = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = tie
            i = j + 1
        return out

    return _corr(ranks(a), ranks(b))


PILLARS = (
    ("quality", "q", "q_raw", model.Q_FLOOR, model.Q_SPAN),
    ("confirmation", "c", "c_raw", model.C_FLOOR, model.C_SPAN),
    ("risk", "r", "r_raw", model.R_FLOOR, model.R_SPAN),
)


def _scored(rows: list[dict]) -> list[dict]:
    return [r for r in rows
            if r.get("conviction") is not None
            and all(isinstance(r.get(k), (int, float)) and r[k] > 0
                    for k in ("q", "c", "r"))]


def _component_detail(rows: list[dict], pillar: str) -> dict:
    """How much of its components' dispersion a pillar's blend retains.

    A weighted mean of near-independent inputs averages much of their dispersion away;
    a weighted mean of strongly overlapping inputs preserves more of it. This is one of
    the two observable quantities reported alongside an uneven split.

    It is not an unchosen property. The retention computed here is a function of the
    blend weights the specification assigns (``model.WEIGHTS[pillar]``, or the default
    quality profile), the number of inputs, and the covariance observed on this
    cross-section. The first two are deliberate model choices; only the third is not.
    """
    if pillar == "quality":
        # Quality's inputs vary by profile, so the correlation is only meaningful
        # within one. The default profile is reported: it is the large majority and
        # the only population on which all five of its inputs are defined.
        pop = [r for r in rows if r.get("profile") == "default"]
        weights = model.QUALITY_PROFILES["default"]
        basis = "default profile"
    else:
        pop = rows
        weights = model.WEIGHTS[pillar]
        basis = "all scored names"

    keys = list(weights)
    series = {k: [r[k] for r in pop if isinstance(r.get(k), (int, float))] for k in keys}
    series = {k: v for k, v in series.items() if len(v) == len(pop) and len(v) >= 2}
    if len(series) < 2 or len(pop) < MIN_POPULATION:
        return {"components": len(keys), "basis": basis, "population": len(pop)}

    pairs = [(a, b) for i, a in enumerate(series) for b in list(series)[i + 1:]]
    corrs = [_corr(series[a], series[b]) for a, b in pairs]
    strongest = max(pairs, key=lambda ab: abs(_corr(series[ab[0]], series[ab[1]])))

    # What a weighted mean of these components does to their dispersion. 1.0 means the
    # blend cancelled nothing; 0.45 means it averaged more than half of it away.
    blended = [
        sum(weights[k] * series[k][i] for k in series) / sum(weights[k] for k in series)
        for i in range(len(pop))
    ]
    avg_component_sd = _mean([_pstdev(series[k]) for k in series])
    retention = (_pstdev(blended) / avg_component_sd) if avg_component_sd > 0 else 0.0

    return {
        "components": len(keys),
        "basis": basis,
        "population": len(pop),
        "mean_abs_correlation": round(_mean([abs(c) for c in corrs]), 4),
        "strongest_pair": list(strongest),
        "strongest_correlation": round(_corr(series[strongest[0]], series[strongest[1]]), 4),
        "component_sd": round(avg_component_sd, 4),
        "blended_sd": round(_pstdev(blended), 4),
        "dispersion_retained": round(retention, 4),
    }


def pillar_influence(rows: list[dict]) -> dict:
    """Nominal weight against effective contribution to cross-sectional score variance.

    **The estimator, exactly.** For pillar ``j`` in ``{Q, C, R}``::

        share_j = Cov(ln P_j, ln Q + ln C + ln R) / Var(ln Q + ln C + ln R)

    Implemented as ``Cov(ln P_j / 3, S) / Var(S)`` with ``S = (ln Q + ln C + ln R) / 3``,
    which is the same quantity: scaling the summed score by 1/3 divides the covariance
    by 3 and the variance by 9, and the 1/3 on the term restores it. The two forms agree
    numerically; ``test_influence_matches_a_hand_rolled_decomposition`` checks the
    implemented form against the summed form computed independently.

    **What goes into it.** The final transformed pillars ``q``, ``c``, ``r`` — the exact
    values the geometric mean consumes — so ``c`` is post-uplift and post-ceiling, and
    ``r`` sits on its floor. Read before ``nightly.round_row`` serialises them to four
    places; at the precision these shares are reported to the two are indistinguishable,
    but the estimator is defined on the finals and is computed on the finals.

    **What it is.** A descriptive decomposition of the current cross-section. It reports
    each pillar's effective contribution to the variance of the log score the ranking
    orders on, and the shares sum to 1 by construction while accounting for covariance
    between pillars. It is a property of *this board*: the split moves as the
    cross-section moves, and it is recomputed nightly for that reason.

    **What it is not.** Not causal, and not evidence about predictive value. It says
    where the ordering's variance currently comes from, not which pillar is right, nor
    what would happen to returns under a different set of transforms. A share can in
    principle fall outside [0, 1] if a pillar covaries negatively with the total; the
    number is reported as computed rather than clipped, and the renderer is written to
    survive it.
    """
    scored = _scored(rows)
    n = len(scored)
    if n < MIN_POPULATION:
        return {"population": n, "sufficient": False,
                "basis": f"needs at least {MIN_POPULATION} scored names"}

    logs = {name: [math.log(r[key]) for r in scored] for name, key, *_ in PILLARS}
    total = [sum(logs[name][i] for name, *_ in PILLARS) / 3.0 for i in range(n)]
    var_total = _cov(total, total)
    conviction = [float(r["conviction"]) for r in scored]

    out = []
    for name, key, raw_key, floor, span in PILLARS:
        term = [x / 3.0 for x in logs[name]]
        level = [r[key] for r in scored]
        raw = [r[raw_key] for r in scored if isinstance(r.get(raw_key), (int, float))]
        share = (_cov(term, total) / var_total) if var_total > 0 else 0.0
        out.append({
            "pillar": name,
            "nominal": round(1 / 3, 4),
            "influence": round(share, 4),
            "vs_nominal": round(share * 3, 3),
            "sd": round(_pstdev(level), 4),
            "sd_log": round(_pstdev(logs[name]), 4),
            "sd_blend": round(_pstdev(raw), 4) if len(raw) == n else None,
            "mean": round(_mean(level), 4),
            "floor": floor,
            "span": span,
            "rank_correlation": round(_spearman(level, conviction), 4),
            "blend": _component_detail(scored, name),
        })

    ranked = sorted(out, key=lambda d: -d["influence"])
    lead, tail = ranked[0], ranked[-1]
    return {
        "population": n,
        "sufficient": True,
        "model_version": model.MODEL_VERSION,
        "pillars": out,
        "leader": lead["pillar"],
        "leader_influence": lead["influence"],
        "spread": (round(lead["influence"] / tail["influence"], 2)
                   if tail["influence"] > 0 else None),
        "estimator": ("share_j = Cov(ln P_j, ln Q + ln C + ln R) / "
                      "Var(ln Q + ln C + ln R), on the final transformed pillars"),
        "basis": (
            "Conviction weights the three pillars equally. Ranking orders on "
            "ln Q + ln C + ln R, in which each pillar carries the same coefficient, so a "
            "pillar's effective contribution to the ordering is set by how much it varies "
            "across the universe. Shares are "
            "Cov(ln P_j, ln Q + ln C + ln R) / Var(ln Q + ln C + ln R), computed on the "
            "final transformed pillars: they account for covariance between pillars and "
            "sum to 1. This is a description of the current cross-section, recomputed "
            "nightly — not a constant of the model, not a causal decomposition, and not "
            "evidence about predictive value."
        ),
    }


def sector_tilt(rows: list[dict], share: float = 0.10) -> dict:
    """Sector mix of the top decile against the universe it was drawn from.

    Quality, valuation and leverage are ranked within sector, so the model takes no
    deliberate sector view. Relative strength, trend, liquidity and volatility are
    ranked universe-wide, and those are free to concentrate the top of the board.
    Descriptive only: nothing here neutralises the ranking.
    """
    scored = [r for r in rows if r.get("conviction") is not None]
    n = len(scored)
    if n < MIN_POPULATION:
        return {"population": n, "sufficient": False}

    scored = sorted(scored, key=lambda r: -r["conviction"])
    k = max(1, int(round(n * share)))
    top = scored[:k]

    def mix(pop: list[dict]) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in pop:
            out[r.get("sector") or "—"] = out.get(r.get("sector") or "—", 0) + 1
        return out

    tm, um = mix(top), mix(scored)
    sectors = []
    for sec, count in sorted(um.items()):
        t_share = tm.get(sec, 0) / k
        u_share = count / n
        sectors.append({
            "sector": sec,
            "top": tm.get(sec, 0),
            "universe": count,
            "top_share": round(t_share, 4),
            "universe_share": round(u_share, 4),
            "tilt_pp": round((t_share - u_share) * 100, 2),
            "multiple": round(t_share / u_share, 2) if u_share else None,
        })
    sectors.sort(key=lambda d: -d["tilt_pp"])
    lead = sectors[0] if sectors else None
    return {
        "population": n,
        "sufficient": True,
        "share": share,
        "cutoff": top[-1]["conviction"] if top else None,
        "top_n": k,
        "sectors": sectors,
        "most_over": lead["sector"] if lead else None,
        "most_over_multiple": lead["multiple"] if lead else None,
        "basis": (
            f"Top {int(share * 100)}% by conviction ({k} of {n}) against the same "
            "universe. Margins, leverage and valuation are ranked within GICS sector, "
            "so this tilt comes from the universe-ranked inputs — relative strength, "
            "trend, liquidity, volatility — not from a sector view. Descriptive: the "
            "ranking is not neutralised."
        ),
    }


def _counterfactual(row: dict) -> dict | None:
    """Score this name again with its unobserved inputs set to its own observed level.

    Not a correction and not an alternative score. It answers one question — how much
    of this name's conviction is the median substitution, rather than the company —
    and the answer is only reported, never ranked on.
    """
    imputed = row.get("imputed") or []
    if not imputed:
        return None
    weights = model.weights_for(row.get("sector") or "")
    used = model.used_percentiles(row.get("sector") or "")
    observed = [k for k in used if k not in imputed]
    if not observed:
        return None

    p = {k: row.get(k) for k in used if isinstance(row.get(k), (int, float))}
    p["drawdown_52w"] = row.get("drawdown_52w")

    # Per pillar, so a name strong on quality and weak on risk is not handed its
    # quality average in place of a missing risk input.
    for pillar, keys in (("quality", weights["quality"]),
                         ("confirmation", weights["confirmation"]),
                         ("risk", weights["risk"])):
        seen = [row[k] for k in keys if k in observed and isinstance(row.get(k), (int, float))]
        if not seen:
            continue
        level = _mean(seen)
        for k in keys:
            if k in imputed:
                p[k] = level
    return model.score(p, weights)


def capped_by_data(rows: list[dict], limit: int = 25) -> dict:
    """Otherwise-strong names whose score is materially held down by missing inputs.

    Explanatory only. No score, ordering, signal or weight anywhere in the ledger is
    adjusted by anything in this function.
    """
    scored = [r for r in rows if r.get("conviction") is not None]
    names = []
    for r in scored:
        alt = _counterfactual(r)
        if not alt:
            continue
        gap = alt["conviction"] - r["conviction"]
        if gap < CAPPED_MIN_GAP or alt["conviction"] < CAPPED_MIN_CONVICTION:
            continue
        names.append({
            "symbol": r.get("symbol"),
            "name": r.get("name"),
            "sector": r.get("sector"),
            "profile": r.get("profile"),
            "conviction": r["conviction"],
            "signal": r.get("signal"),
            "would_be": alt["conviction"],
            "would_be_signal": alt["signal"],
            "gap": gap,
            "tier_change": alt["signal"] != r.get("signal"),
            "imputed": list(r.get("imputed") or []),
            "data_confidence": r.get("data_confidence"),
        })
    names.sort(key=lambda d: (-d["gap"], -d["would_be"]))
    with_gap = len(names)
    return {
        "population": len(scored),
        "constrained": with_gap,
        "tier_changes": sum(1 for d in names if d["tier_change"]),
        "min_gap": CAPPED_MIN_GAP,
        "min_conviction": CAPPED_MIN_CONVICTION,
        "names": names[:limit],
        "truncated": max(0, with_gap - limit),
        "basis": (
            "Each name re-scored with its unobserved inputs set to the average of what "
            "it does report in the same pillar, instead of the group median the model "
            "substitutes. The gap is what the median substitution costs the name. "
            "Explanatory only: no published score, signal, ordering or weight is "
            f"adjusted by it. Listed when the gap is at least {CAPPED_MIN_GAP} points "
            f"and the name would reach {CAPPED_MIN_CONVICTION}."
        ),
    }


def build(rows: list[dict]) -> dict:
    """Every diagnostic, in the shape the ledger and the terminal consume."""
    return {
        "pillar_influence": pillar_influence(rows),
        "sector_tilt": sector_tilt(rows),
        "capped_by_data": capped_by_data(rows),
    }
