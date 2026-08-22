"""Descriptions of the board that must never become changes to it.

The whole point of this module is that it measures without touching. Every scoring
constant, every published conviction and every ordering has to survive it byte for
byte, and the first test here is the one that matters: if it ever fails, the
diagnostics have stopped being diagnostics.

The rest check that the numbers mean what the panel says they mean — an influence share
that does not sum to 1 is not a decomposition, and a counterfactual that can lower a
score is not measuring what "held down by missing data" claims.
"""
from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import pytest

from equity_monitor import diagnostics, model

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = ROOT / "web" / "terminal.html"
LEDGER = ROOT / "ledger" / "index.json"


def _universe(n: int = 240, seed: int = 11) -> list[dict]:
    """A synthetic cross-section wide enough for a variance to mean something."""
    rng = random.Random(seed)
    sectors = ["Information Technology", "Financials", "Real Estate", "Health Care",
               "Industrials", "Energy"]
    rows = []
    for i in range(n):
        sector = sectors[i % len(sectors)]
        rows.append({
            "symbol": f"S{i:03d}",
            "name": f"Company {i}",
            "sector": sector,
            "asset_class": "EQUITY",
            "roic": rng.uniform(0.01, 0.4) if rng.random() > 0.25 else None,
            "fcf_yield": rng.uniform(-0.02, 0.12) if rng.random() > 0.2 else None,
            "gross_margin": rng.uniform(0.1, 0.8) if rng.random() > 0.3 else None,
            "net_debt_ebitda": rng.uniform(-1, 6) if rng.random() > 0.35 else None,
            "earnings_stability": rng.uniform(0, 1),
            "roe": rng.uniform(0.01, 0.35) if rng.random() > 0.1 else None,
            "equity_to_assets": rng.uniform(0.05, 0.5) if rng.random() > 0.1 else None,
            "cfo_yield": rng.uniform(0, 0.2) if rng.random() > 0.3 else None,
            "ffo_yield": rng.uniform(0, 0.15) if rng.random() > 0.2 else None,
            "debt_to_assets": rng.uniform(0.1, 0.7) if rng.random() > 0.2 else None,
            "efficiency_ratio": rng.uniform(0.4, 0.8) if sector == "Financials"
                                and rng.random() > 0.6 else None,
            "rs_blend": rng.uniform(-1, 1),
            "trend": rng.uniform(0, 1),
            "adv_usd": rng.uniform(1e6, 1e9),
            "vol_1y": rng.uniform(0.1, 0.9),
            "drawdown_52w": rng.uniform(0, 0.6),
            "market_cap": rng.uniform(1e9, 1e12),
            "earnings_yield": rng.uniform(-0.05, 0.12),
            "ebitda_yield": rng.uniform(0, 0.2),
        })
    model.score_rows(rows)
    return rows


# ---------------------------------------------------------------------------
# the invariant
# ---------------------------------------------------------------------------
def test_no_diagnostic_changes_a_single_published_number():
    """The reason this module exists in its own file with no write path.

    Pillar influence was measured because the board's behaviour was worth
    understanding, NOT because the weights should move. If running the description
    ever alters the thing described, the description is worthless and the model has
    been silently re-based.
    """
    rows = _universe()
    before = copy.deepcopy(rows)

    diagnostics.build(rows)
    diagnostics.pillar_influence(rows)
    diagnostics.sector_tilt(rows)
    diagnostics.capped_by_data(rows)

    assert rows == before, "a diagnostic mutated the rows it was handed"


def test_ordering_and_tiers_are_untouched():
    rows = _universe()
    order_before = [r["symbol"] for r in sorted(rows, key=lambda r: (-r["conviction"], r["symbol"]))]
    tiers_before = {r["symbol"]: r["signal"] for r in rows}
    convs_before = {r["symbol"]: r["conviction"] for r in rows}

    diagnostics.build(rows)

    assert [r["symbol"] for r in sorted(rows, key=lambda r: (-r["conviction"], r["symbol"]))] \
        == order_before
    assert {r["symbol"]: r["signal"] for r in rows} == tiers_before
    assert {r["symbol"]: r["conviction"] for r in rows} == convs_before


def test_the_scoring_spec_hash_is_unaffected():
    """Diagnostics must not add anything spec() hashes; the frozen model stays frozen."""
    before = model.spec_hash()
    diagnostics.build(_universe())
    assert model.spec_hash() == before


# ---------------------------------------------------------------------------
# pillar influence
# ---------------------------------------------------------------------------
def test_influence_shares_sum_to_one():
    """It is a variance decomposition or it is nothing. Shares that do not add up are
    not shares, and a panel presenting them as such would be inventing a denominator."""
    d = diagnostics.pillar_influence(_universe())
    assert d["sufficient"]
    assert sum(p["influence"] for p in d["pillars"]) == pytest.approx(1.0, abs=1e-3)


def test_influence_is_reported_against_an_equal_nominal_weight():
    d = diagnostics.pillar_influence(_universe())
    for p in d["pillars"]:
        assert p["nominal"] == pytest.approx(1 / 3, abs=1e-3)
        assert p["vs_nominal"] == pytest.approx(p["influence"] * 3, abs=1e-2)


def test_a_pillar_with_no_dispersion_gets_no_influence():
    """The claim the panel makes, tested directly: influence follows variation, not
    weight. A pillar identical for every name cannot separate any two of them."""
    rows = _universe()
    for r in rows:
        r["c"] = 0.5          # flatten confirmation entirely
        r["c_raw"] = 0.47
    d = diagnostics.pillar_influence(rows)
    conf = next(p for p in d["pillars"] if p["pillar"] == "confirmation")
    assert conf["sd"] == pytest.approx(0.0, abs=1e-9)
    assert conf["influence"] == pytest.approx(0.0, abs=1e-6), \
        "a constant pillar still carries a third of the nominal weight and none of the say"
    assert d["leader"] != "confirmation"


def test_influence_tracks_dispersion_not_the_published_weight():
    rows = _universe()
    d = diagnostics.pillar_influence(rows)
    by_sd = sorted(d["pillars"], key=lambda p: -p["sd_log"])
    by_influence = sorted(d["pillars"], key=lambda p: -p["influence"])
    assert [p["pillar"] for p in by_sd] == [p["pillar"] for p in by_influence]


def test_the_blend_detail_names_the_closest_pair_of_inputs():
    """Confirmation's two inputs measure nearly the same thing, which is why its blend
    cancels almost nothing. The panel has to be able to say so with a number."""
    d = diagnostics.pillar_influence(_universe())
    conf = next(p for p in d["pillars"] if p["pillar"] == "confirmation")
    assert conf["blend"]["components"] == 2
    assert set(conf["blend"]["strongest_pair"]) == {"p_rs", "p_trend"}
    assert 0.0 <= conf["blend"]["dispersion_retained"] <= 1.5

    qual = next(p for p in d["pillars"] if p["pillar"] == "quality")
    assert qual["blend"]["components"] == 5
    assert qual["blend"]["basis"] == "default profile", \
        "quality's inputs vary by profile; a correlation across profiles is meaningless"


def test_a_board_too_small_to_decompose_says_so_rather_than_guessing():
    d = diagnostics.pillar_influence(_universe(n=6))
    assert d["sufficient"] is False
    assert "pillars" not in d


def test_influence_matches_a_hand_rolled_decomposition():
    """Recomputed independently of the module, so an error in the helper cannot agree
    with itself."""
    rows = [r for r in _universe() if r.get("conviction") is not None]
    d = diagnostics.pillar_influence(rows)

    logs = {k: [math.log(r[k]) for r in rows] for k in ("q", "c", "r")}
    total = [(logs["q"][i] + logs["c"][i] + logs["r"][i]) / 3 for i in range(len(rows))]
    mt = sum(total) / len(total)
    var = sum((x - mt) ** 2 for x in total) / len(total)
    for key, name in (("q", "quality"), ("c", "confirmation"), ("r", "risk")):
        term = [x / 3 for x in logs[key]]
        m = sum(term) / len(term)
        cov = sum((a - m) * (b - mt) for a, b in zip(term, total)) / len(term)
        got = next(p for p in d["pillars"] if p["pillar"] == name)["influence"]
        assert got == pytest.approx(cov / var, abs=1e-3)


# ---------------------------------------------------------------------------
# sector tilt
# ---------------------------------------------------------------------------
def test_sector_shares_are_measured_against_the_same_universe():
    d = diagnostics.sector_tilt(_universe())
    assert d["sufficient"]
    assert sum(s["top"] for s in d["sectors"]) == d["top_n"]
    assert sum(s["universe"] for s in d["sectors"]) == d["population"]
    assert sum(s["top_share"] for s in d["sectors"]) == pytest.approx(1.0, abs=1e-3)


def test_tilt_multiple_is_the_ratio_of_the_two_shares():
    for s in diagnostics.sector_tilt(_universe())["sectors"]:
        if s["universe_share"]:
            assert s["multiple"] == pytest.approx(
                s["top_share"] / s["universe_share"], abs=1e-2)


def test_sector_tilt_states_that_it_does_not_neutralise_anything():
    assert "not neutralised" in diagnostics.sector_tilt(_universe())["basis"]


# ---------------------------------------------------------------------------
# capped by missing data
# ---------------------------------------------------------------------------
def test_a_name_with_nothing_imputed_has_no_counterfactual():
    rows = _universe()
    complete = next(r for r in rows if not r.get("imputed"))
    assert diagnostics._counterfactual(complete) is None


def test_the_counterfactual_only_ever_reports_a_gain():
    """The list is 'held down by missing data'. A name whose observed inputs are worse
    than the median it was given is not being held down, and listing it would invert
    the claim the heading makes."""
    d = diagnostics.capped_by_data(_universe())
    for n in d["names"]:
        assert n["gap"] >= d["min_gap"]
        assert n["would_be"] > n["conviction"]
        assert n["would_be"] >= d["min_conviction"]


def test_the_counterfactual_substitutes_within_the_pillar_not_across_it():
    """A name strong on quality and weak on risk must not be handed its quality average
    for a missing risk input — that would import a signal from the wrong place."""
    row = {
        "sector": "Information Technology",
        "p_roic": 0.9, "p_fcf_yield": 0.9, "p_gross_margin": 0.9,
        "p_leverage": 0.9, "p_earnings_stability": 0.9,
        "p_rs": 0.5, "p_trend": 0.5,
        "p_liquidity": 0.1, "p_value": 0.1, "p_lowvol": model.NEUTRAL,
        "drawdown_52w": 0.0,
        "imputed": ["p_lowvol"],
    }
    alt = diagnostics._counterfactual(row)
    # p_lowvol should be filled from the risk pillar's own observed level (0.1), which is
    # BELOW the median it was given, so the counterfactual is worse — and therefore never
    # reaches the published list.
    assert alt["r"] < model.R_FLOOR + model.R_SPAN * model.NEUTRAL


def test_capped_names_carry_the_inputs_that_were_not_observed():
    d = diagnostics.capped_by_data(_universe())
    for n in d["names"]:
        assert n["imputed"], "a constrained name must name what it is missing"
        assert all(k.startswith("p_") for k in n["imputed"])


def test_capped_by_data_says_it_adjusts_nothing():
    assert "Explanatory only" in diagnostics.capped_by_data(_universe())["basis"]


# ---------------------------------------------------------------------------
# the shipped ledger, and the terminal that reads it
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not LEDGER.exists(), reason="no ledger committed")
def test_the_published_ledger_carries_the_diagnostics_the_terminal_reads():
    payload = json.loads(LEDGER.read_text())
    diag = payload.get("diagnostics") or {}
    assert set(diag) >= {"pillar_influence", "sector_tilt", "capped_by_data"}
    assert payload.get("coverage_detail"), "scoped denominators must be published"
    for field, rep in payload["coverage_detail"].items():
        assert rep["population"] > 0
        assert 0.0 <= rep["share"] <= 1.0
        assert rep["observed"] <= rep["population"]


@pytest.mark.skipif(not LEDGER.exists(), reason="no ledger committed")
def test_profile_scoped_inputs_are_not_measured_against_the_whole_universe():
    """efficiency_ratio read 4.5% for months because 853 non-banks were counted as
    missing a bank metric. The denominator is the bug this test pins."""
    payload = json.loads(LEDGER.read_text())
    detail = payload["coverage_detail"]
    scored = len([r for r in payload["all"] if r.get("conviction") is not None])

    assert detail["efficiency_ratio"]["scope"] == ["Financials"]
    assert detail["efficiency_ratio"]["population"] < scored
    assert detail["ffo_yield"]["scope"] == ["Real Estate"]
    assert detail["roic"]["scope"] == ["default"]
    # confirmation and risk inputs are scored on every name, so they stay universe-wide
    assert detail["rs_blend"]["scope"] == []
    assert detail["adv_usd"]["scope"] == []
    assert detail["rs_blend"]["population"] == scored


def test_the_terminal_renders_influence_without_claiming_to_correct_it():
    html = TERMINAL.read_text(encoding="utf-8")
    body = html.split("function renderPillars(", 1)[1].split("\nfunction ", 1)[0]
    assert "pillar_influence" in body
    assert "nominal" in body.lower()
    assert "Descriptive, not a correction" in body, \
        "the panel must say it changes nothing, because it changes nothing"


def test_the_terminal_labels_coverage_with_its_scope():
    html = TERMINAL.read_text(encoding="utf-8")
    body = html.split("function renderData(", 1)[1].split("\nfunction ", 1)[0]
    assert "coverage_detail" in body
    assert "scope" in body
    assert "population" in body


def test_the_terminal_reads_the_tilt_and_the_capped_list():
    html = TERMINAL.read_text(encoding="utf-8")
    assert "sector_tilt" in html
    assert "capped_by_data" in html
