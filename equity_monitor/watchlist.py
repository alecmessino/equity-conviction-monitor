"""The overnight diff: what changed on the board since the last run, and why.

The obvious version of this — list every name whose tier changed — is actively
misleading, and the churn diagnostic is what showed it. On an ordinary night, **39 of 39
tier changes came from moves of two points or less across a hard threshold**. A morning
list of 39 "new BUYs" and "new WATCHes" built from those would be 39 pieces of noise
presented as 39 decisions, and it would be wrong in the specific way that erodes trust in
a tool: confidently, in detail, every single day.

So the diff sorts changes by whether the *view* changed, not by whether a label did:

* **Upgrades / downgrades** — the tier moved *and* conviction moved more than
  :data:`churn.MARGINAL_MOVE`. Something actually happened.
* **Boundary crossings** — the tier moved on a hair. Reported, separately and quietly,
  because a name sitting on 69/70 is worth knowing about precisely so it can be ignored
  rather than traded.
* **Material moves without a reclassification** — conviction moved hard while staying
  inside its tier. A tier-only diff misses these entirely, and they are frequently the
  most informative rows on the page: a name going 58 → 68 has not become a BUY, but it is
  doing something.
* **Universe changes** — names that entered or left, which are index mechanics rather
  than opinion and must never be mistaken for a rating action.

Every row carries its driver: the largest factor pushing in the direction of the move,
taken from the exact log-space attribution already computed nightly. Nothing is inferred
or re-derived here.
"""
from __future__ import annotations

from . import churn, model, snapshots

# Conviction moving at least this far is a change of view rather than a rounding wobble.
# A name drifting 58 -> 68 without crossing a tier boundary is real news that a
# tier-only diff would silently drop.
MATERIAL_MOVE = 5.0

# Longest list any one section will publish. A morning view that needs scrolling is a
# morning view that gets skipped.
SECTION_LIMIT = 25


def _tier_rank(tier: str) -> int:
    """Position in the tier ladder, ascending, so direction is comparable."""
    order = ["AVOID", "WATCH", "HOLD", "BUY", "STRONG"]
    return order.index(tier) if tier in order else -1


def _driver(attr: dict | None, direction: int) -> dict | None:
    """The largest factor pushing the same way the score moved.

    Deliberately not the largest factor by magnitude. A name that rose five points is
    explained by what lifted it, and the biggest number in the decomposition is
    sometimes a drag pulling the other way — reporting that as the reason produces a
    row reading "+5, driven by deteriorating trend", which is two true numbers arranged
    into a false sentence.
    """
    if not attr:
        return None
    same = [(k, v) for k, v in (attr.get("factors") or {}).items()
            if (v > 0) == (direction > 0) and abs(v) > 0.01]
    if not same:
        return None
    k, v = max(same, key=lambda kv: abs(kv[1]))
    return {"factor": k, "points": round(v, 2)}


def _drag(attr: dict | None, direction: int) -> dict | None:
    """The largest factor pushing against the move, when there is one worth naming."""
    if not attr:
        return None
    against = [(k, v) for k, v in (attr.get("factors") or {}).items()
               if (v > 0) != (direction > 0) and abs(v) > 0.01]
    if not against:
        return None
    k, v = max(against, key=lambda kv: abs(kv[1]))
    return {"factor": k, "points": round(v, 2)}


def _row(sym: str, previous: dict, current: dict, attribution: dict,
         meta: dict) -> dict:
    before = previous["rows"][sym]["conviction"]
    after = current["rows"][sym]["conviction"]
    delta = after - before
    direction = 1 if delta >= 0 else -1
    attr = (attribution or {}).get(sym)
    info = meta.get(sym) or {}
    return {
        "symbol": sym,
        "name": info.get("name"),
        "sector": info.get("sector") or current.get("sectors", {}).get(sym),
        "from": before,
        "to": after,
        "delta": delta,
        "from_tier": model.signal(before),
        "to_tier": model.signal(after),
        "marginal": abs(delta) <= churn.MARGINAL_MOVE,
        "driver": _driver(attr, direction),
        "drag": _drag(attr, direction),
        "data_confidence": current["rows"][sym].get("data_confidence"),
    }


def diff(previous: dict, current: dict, attribution: dict | None = None,
         meta: dict | None = None) -> dict | None:
    """What changed between two snapshots, sorted by whether it means anything."""
    if not previous or not current:
        return None

    prev_syms, curr_syms = set(previous["rows"]), set(current["rows"])
    shared = sorted(prev_syms & curr_syms)
    # An empty overlap is not an empty diff: a reconstitution that replaced the universe
    # wholesale is exactly the morning a person needs to be told something happened.
    if not prev_syms and not curr_syms:
        return None

    meta = meta or {}
    upgrades, downgrades, boundary, movers = [], [], [], []

    for sym in shared:
        row = _row(sym, previous, current, attribution or {}, meta)
        moved_tier = row["from_tier"] != row["to_tier"]
        if moved_tier and row["marginal"]:
            boundary.append(row)
        elif moved_tier:
            (upgrades if _tier_rank(row["to_tier"]) > _tier_rank(row["from_tier"])
             else downgrades).append(row)
        elif abs(row["delta"]) >= MATERIAL_MOVE:
            movers.append(row)

    upgrades.sort(key=lambda r: -r["delta"])
    downgrades.sort(key=lambda r: r["delta"])
    movers.sort(key=lambda r: -abs(r["delta"]))
    boundary.sort(key=lambda r: (-_tier_rank(r["to_tier"]), r["symbol"]))

    def entry(sym: str, snap: dict) -> dict:
        info = meta.get(sym) or {}
        row = snap["rows"].get(sym) or {}
        conviction = row.get("conviction")
        return {"symbol": sym, "name": info.get("name"),
                "sector": info.get("sector") or snap.get("sectors", {}).get(sym),
                "conviction": conviction,
                "tier": model.signal(conviction) if conviction is not None else None}

    entered = [entry(s, current) for s in sorted(curr_syms - prev_syms)]
    left = [entry(s, previous) for s in sorted(prev_syms - curr_syms)]

    return {
        "from": previous["date"],
        "to": current["date"],
        "names_compared": len(shared),
        "upgrades": upgrades[:SECTION_LIMIT],
        "downgrades": downgrades[:SECTION_LIMIT],
        "movers": movers[:SECTION_LIMIT],
        "boundary": boundary[:SECTION_LIMIT],
        "entered": entered[:SECTION_LIMIT],
        "left": left[:SECTION_LIMIT],
        "counts": {
            "upgrades": len(upgrades), "downgrades": len(downgrades),
            "movers": len(movers), "boundary": len(boundary),
            "entered": len(entered), "left": len(left),
            "unchanged": len(shared) - len(upgrades) - len(downgrades)
                         - len(movers) - len(boundary),
        },
        "material_move": MATERIAL_MOVE,
        "marginal_move": churn.MARGINAL_MOVE,
        "scope": ("Changes since the previous run. Tier changes driven by a move of "
                  f"{churn.MARGINAL_MOVE:.0f} points or less are listed separately as "
                  "boundary crossings, because a name at 69 and a name at 71 are not "
                  "different holdings."),
    }


def from_ledger(ledger_dir: str, payload: dict | None = None) -> dict | None:
    """Diff the two most recent snapshots on disk.

    Company names and sectors come from the published ledger rather than the snapshot,
    which stores only what the model reads. Passing the payload is optional so the diff
    still builds — with tickers alone — from a bare snapshot directory.
    """
    current = snapshots.latest(ledger_dir)
    if not current:
        return None
    prior = snapshots.latest(ledger_dir, before=current.get("date"))
    if not prior:
        return None

    meta = {r["symbol"]: {"name": r.get("name"), "sector": r.get("sector")}
            for r in ((payload or {}).get("all") or []) if r.get("symbol")}
    attribution = ((payload or {}).get("attribution") or {}).get("names") or {}
    return diff(prior, current, attribution, meta)
