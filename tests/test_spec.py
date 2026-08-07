"""The scoring specification is frozen. This test is the lock.

Why a hash rather than a comment saying "do not change these":

The nightly run persists a factor-level snapshot of every name. Those snapshots are
the research dataset — months from now they get regressed against forward returns to
measure whether the model predicts anything at all. That measurement is only valid
*within a single specification*. If someone nudges the ROIC weight from 0.30 to 0.35
and nothing records it, the resulting Information Coefficient is computed across two
different models and means nothing, while looking entirely respectable.

So: any change to a weight, a pillar floor, a tier boundary, or the rank spec breaks
this test. The fix is never to update the pin quietly. It is to bump MODEL_VERSION,
update the pin, and note the change in docs/methodology.html — so that every snapshot
written before and after can be told apart by the spec_hash it carries.
"""
from __future__ import annotations

import json

import pytest

from equity_monitor import model

# Bump deliberately, together with MODEL_VERSION, when the specification changes.
FROZEN_SPEC_VERSION = "v3.1.0"
FROZEN_SPEC_HASH = "516fd7e5a032"


def test_spec_hash_is_pinned():
    assert model.MODEL_VERSION == FROZEN_SPEC_VERSION, (
        f"MODEL_VERSION changed to {model.MODEL_VERSION} without updating this pin."
    )
    assert model.spec_hash() == FROZEN_SPEC_HASH, (
        "The scoring specification changed.\n\n"
        f"  expected {FROZEN_SPEC_HASH}\n  got      {model.spec_hash()}\n\n"
        "If this was deliberate: bump MODEL_VERSION, update FROZEN_SPEC_HASH, and "
        "record the change in docs/methodology.html. Historical snapshots carry the "
        "old hash, so analysis can still segment by model version.\n\n"
        f"current spec:\n{json.dumps(model.spec(), indent=2, sort_keys=True)}"
    )


def test_spec_hash_is_stable_across_calls():
    """A hash that varies between calls could not pin anything."""
    assert len({model.spec_hash() for _ in range(10)}) == 1


def test_spec_hash_actually_changes_when_the_spec_changes():
    """Guard the guard: a hash insensitive to the weights would pass while asleep."""
    original = model.WEIGHTS["quality"]["p_roic"]
    baseline = model.spec_hash()
    model.WEIGHTS["quality"]["p_roic"] = original + 0.01
    try:
        assert model.spec_hash() != baseline, (
            "spec_hash did not change when a quality weight changed — the freeze is "
            "not actually locking the thing it claims to lock"
        )
    finally:
        model.WEIGHTS["quality"]["p_roic"] = original
    assert model.spec_hash() == baseline, "spec_hash failed to restore"


def test_spec_covers_the_sector_profiles():
    """A profile change alters what a whole sector is scored on. If it did not move
    the hash, an entire sector could be silently re-based mid-history."""
    spec = model.spec()
    assert set(spec["quality_profiles"]) == set(model.QUALITY_PROFILES)

    original = model.QUALITY_PROFILES["Financials"]["p_roe"]
    baseline = model.spec_hash()
    model.QUALITY_PROFILES["Financials"]["p_roe"] = original + 0.01
    try:
        assert model.spec_hash() != baseline, "a profile weight change did not move the hash"
    finally:
        model.QUALITY_PROFILES["Financials"]["p_roe"] = original


def test_every_profile_is_a_valid_probability_weighting():
    """Weights that do not sum to 1 silently rescale a sector's whole quality pillar."""
    known = {k for k, _, _ in model.RANK_SPEC} | {"p_value"}
    for name, weights in model.QUALITY_PROFILES.items():
        assert abs(sum(weights.values()) - 1.0) < 1e-9, f"{name} sums to {sum(weights.values())}"
        assert all(w > 0 for w in weights.values()), f"{name} has a non-positive weight"
        for key in weights:
            assert key in known, f"{name} references {key}, which nothing ranks"


def test_profile_sectors_route_correctly():
    assert model.quality_profile("Financials") == "Financials"
    assert model.quality_profile("Real Estate") == "Real Estate"
    for other in ("Industrials", "Information Technology", "", "Nonsense Sector"):
        assert model.quality_profile(other) == "default", other


def test_spec_covers_every_score_moving_constant():
    """Anything that can move a score must be inside the hashed spec.

    Listed explicitly so that adding a new constant to the model without adding it to
    spec() is caught here rather than discovered later as unexplained score drift.
    """
    spec = model.spec()
    flat = json.dumps(spec, sort_keys=True)

    for constant in (model.Q_FLOOR, model.Q_SPAN, model.C_FLOOR, model.C_SPAN,
                     model.R_FLOOR, model.R_SPAN, model.C_CEILING,
                     model.MR_QUALITY_GATE, model.MR_DRAWDOWN_GATE,
                     model.MR_DRAWDOWN_SPAN, model.MR_MAX_UPLIFT,
                     model.MIN_SECTOR_GROUP, model.NEUTRAL):
        assert str(constant) in flat, f"{constant} is not represented in spec()"

    assert set(spec["weights"]) == set(model.WEIGHTS)
    for pillar, weights in model.WEIGHTS.items():
        assert spec["weights"][pillar] == pytest.approx(weights)

    assert [tuple(t) for t in spec["signal_tiers"]] == model.SIGNAL_TIERS
    assert [tuple(r) for r in spec["rank_spec"]] == model.RANK_SPEC
    assert set(spec["sector_relative"]) == model.SECTOR_RELATIVE


def test_spec_is_json_serialisable():
    """It is written into the ledger and into every snapshot."""
    round_tripped = json.loads(json.dumps(model.spec()))
    assert round_tripped["version"] == model.MODEL_VERSION
