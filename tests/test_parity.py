"""Frontend/backend parity gate.

The terminal recomputes conviction in the browser so the detail panel can show a live
decomposition. That means the scoring function exists twice — once in Python, once in
JavaScript — and two implementations of one formula drift silently by default. The
drift is invisible precisely because both sides keep producing plausible numbers.

This gate extracts the JS model port straight out of ``web/terminal.html``, runs it in
node, and asserts it agrees with ``equity_monitor.model.score`` on hundreds of
randomised inputs plus the edge cases that matter.

Splitting cross-sectional ranking (Python only) from a pure ``score(percentiles)`` is
what makes this tractable: there is exactly one pure function to keep in sync, and it
takes plain numbers.
"""
from __future__ import annotations

import json
import pathlib
import random
import re
import shutil
import subprocess
import tempfile

import pytest

from equity_monitor import model

ROOT = pathlib.Path(__file__).resolve().parents[1]
TERMINAL = ROOT / "web" / "terminal.html"

MARKER_START = "MODEL PORT"
MARKER_END = "END MODEL PORT"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node is required to execute the JS port")


def extract_port() -> str:
    """Pull the delimited model port out of the terminal's inline script.

    The markers sit inside comment blocks, so we slice from the *end* of the opening
    comment to the *start* of the closing one — anything else hands node a fragment
    of prose and fails with a syntax error rather than a parity result.
    """
    html = TERMINAL.read_text()
    script = re.search(r"<script>(.*?)</script>", html, re.S)
    assert script, "terminal.html has no inline <script> block"
    body = script.group(1)
    start = body.find(MARKER_START)
    end = body.find(MARKER_END)
    assert start != -1 and end != -1, (
        f"could not find the {MARKER_START}/{MARKER_END} markers in terminal.html — "
        "the parity gate cannot verify a port it cannot locate"
    )
    open_close = body.find("*/", start)
    assert open_close != -1, "the opening MODEL PORT marker is not inside a /* */ comment"
    close_open = body.rfind("/*", start, end)
    assert close_open != -1, "the END MODEL PORT marker is not inside a /* */ comment"
    return body[open_close + 2:close_open]


def test_port_declares_the_v3_structure():
    """Guard the shape, so a rewrite cannot quietly reintroduce a v2 formula."""
    js = extract_port()
    assert "Math.cbrt" in js, "conviction must use the geometric mean, not a raw product"
    assert "MR_QUALITY_GATE" in js, "the mean-reversion uplift must be quality-gated"
    assert "p_roic" in js and "p_value" in js, "port must consume percentile inputs"
    # v2 artefacts that must not come back.
    assert "cmRaw" not in js
    assert "0.20) / 0.60" not in js and "(raw-0.20)/0.60" not in js


def run_js(cases: list[dict]) -> list[dict]:
    """Execute the extracted port against `cases` in node and return its output."""
    js = extract_port()
    with tempfile.TemporaryDirectory() as tmp:
        script = pathlib.Path(tmp) / "port.mjs"
        script.write_text(
            js
            + "\nconst cases = " + json.dumps(cases) + ";\n"
            + "console.log(JSON.stringify(cases.map(c => score(c))));\n"
        )
        proc = subprocess.run(["node", str(script)], capture_output=True, text=True,
                              timeout=60)
    if proc.returncode != 0:
        raise AssertionError(f"node failed executing the port:\n{proc.stderr}")
    return json.loads(proc.stdout)


PROFILES = ["default", "Financials", "Real Estate", "", None, "Industrials"]


def random_case(rng: random.Random) -> dict:
    case = {k: round(rng.random(), 6) for k in model.ALL_PERCENTILES}
    case["drawdown_52w"] = round(rng.random() * 0.6, 6)
    # Exercise the sector profiles. Without this the gate would run only the default
    # weighting and stay silent on whichever port got the profile routing wrong.
    profile = rng.choice(PROFILES)
    if profile is not None:
        case["profile"] = profile
    # A quarter of the time, drop some inputs entirely — missing values take a
    # different branch in both implementations and are where drift would hide.
    if rng.random() < 0.25:
        for key in rng.sample(model.ALL_PERCENTILES, rng.randint(1, 4)):
            case[key] = None
    return case


EDGE_CASES: list[dict] = [
    {k: 0.0 for k in model.ALL_PERCENTILES},
    {k: 1.0 for k in model.ALL_PERCENTILES},
    {k: 0.5 for k in model.ALL_PERCENTILES},
    # exactly on the mean-reversion gates
    {**{k: 0.55 for k in model.ALL_PERCENTILES}, "drawdown_52w": 0.15},
    {**{k: 0.55 for k in model.ALL_PERCENTILES}, "drawdown_52w": 0.1500001},
    {**{k: 0.5499 for k in model.ALL_PERCENTILES}, "drawdown_52w": 0.4},
    # drawdown beyond the span, so the uplift saturates
    {**{k: 0.9 for k in model.ALL_PERCENTILES}, "drawdown_52w": 0.95},
    # nothing observed at all
    {k: None for k in model.ALL_PERCENTILES},
    {},
    # every profile at its extremes, plus an unknown sector that must fall back
    *[{**{k: v for k in model.ALL_PERCENTILES}, "profile": prof, "drawdown_52w": 0.3}
      for prof in ("default", "Financials", "Real Estate", "Nonsense Sector")
      for v in (0.0, 0.5, 1.0)],
]


def test_js_and_python_agree():
    rng = random.Random(20260806)
    cases = EDGE_CASES + [random_case(rng) for _ in range(400)]
    js_results = run_js(cases)
    assert len(js_results) == len(cases)

    mismatches = []
    for case, got in zip(cases, js_results):
        sector = case.get("profile")
        want = model.score(dict(case),
                           model.weights_for(sector) if sector is not None else None)
        for key in ("conviction", "signal"):
            if want[key] != got[key]:
                mismatches.append((case, key, want[key], got[key]))
        for key in ("q", "c", "r", "q_raw", "c_raw", "r_raw", "mr_uplift"):
            if abs(want[key] - got[key]) > 1e-9:
                mismatches.append((case, key, want[key], got[key]))

    assert not mismatches, (
        f"{len(mismatches)} JS/Python disagreements; first three:\n"
        + "\n".join(f"  {k}: python={w!r} js={g!r}  case={c}"
                    for c, k, w, g in mismatches[:3])
    )


def test_sector_profiles_match_between_implementations():
    """A profile present in one port and not the other reweights a whole sector."""
    js = extract_port()
    for name, weights in model.QUALITY_PROFILES.items():
        if name == "default":
            continue
        block = re.search(rf'"{re.escape(name)}":\s*\{{(.*?)\}}', js, re.S)
        assert block, f"JS port has no quality profile for {name!r}"
        found = {m.group(1): float(m.group(2))
                 for m in re.finditer(r"(\w+)\s*:\s*([0-9.]+)", block.group(1))}
        assert found == pytest.approx(weights), (
            f"{name} profile differs — python={weights} js={found}")


def test_weights_match_between_implementations():
    """Weights live in both files; a change to one alone silently reweights the board."""
    js = extract_port()
    for pillar, weights in model.WEIGHTS.items():
        block = re.search(rf"{pillar}:\s*\{{(.*?)\}}", js, re.S)
        assert block, f"JS port has no '{pillar}' weight block"
        found = {m.group(1): float(m.group(2))
                 for m in re.finditer(r"(\w+)\s*:\s*([0-9.]+)", block.group(1))}
        assert found == pytest.approx(weights), (
            f"{pillar} weights differ — python={weights} js={found}"
        )


def test_signal_thresholds_match():
    js = extract_port()
    found = [(int(a), b) for a, b in re.findall(r"\[(\d+),\s*'([A-Z]+)'\]", js)]
    assert found == model.SIGNAL_TIERS


# ---------------------------------------------------------------------------
# hover explanations
#
# The terminal explains every factor and pillar on hover, from one registry. These
# tests exist so a new input cannot ship undocumented: the failure mode is silent —
# the metric renders fine and simply has no explanation behind it, which nobody
# notices until someone hovers it looking for the formula and gets nothing.
# ---------------------------------------------------------------------------
def _js_object(html: str, name: str) -> list[str]:
    """Top-level keys of a `const NAME={...}` literal in the terminal."""
    start = html.index(f"const {name}={{") + len(f"const {name}=")
    depth, i = 0, start
    while i < len(html):
        if html[i] == "{":
            depth += 1
        elif html[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    body = html[start:i + 1]
    keys, depth = [], 0
    for line in body.splitlines():
        stripped = line.strip()
        if depth == 1 and ":" in stripped and stripped[0].isalpha():
            keys.append(stripped.split(":", 1)[0].strip())
        depth += line.count("{") - line.count("}")
    return keys


def test_every_scored_input_has_a_hover_definition():
    html = TERMINAL.read_text(encoding="utf-8")
    documented = set(_js_object(html, "FACTOR_DEFS"))
    scored = set(model.ALL_PERCENTILES)
    for profile in model.QUALITY_PROFILES.values():
        scored |= set(profile)
    missing = scored - documented
    assert not missing, f"scored inputs with no hover explanation: {sorted(missing)}"


def test_no_hover_definition_describes_an_input_the_model_does_not_use():
    """A definition for a retired factor is documentation that quietly became fiction."""
    html = TERMINAL.read_text(encoding="utf-8")
    documented = set(_js_object(html, "FACTOR_DEFS"))
    scored = set(model.ALL_PERCENTILES)
    for profile in model.QUALITY_PROFILES.values():
        scored |= set(profile)
    assert not documented - scored, \
        f"explanations for inputs the model never reads: {sorted(documented - scored)}"


def test_every_pillar_has_a_hover_definition():
    html = TERMINAL.read_text(encoding="utf-8")
    assert set(_js_object(html, "PILLAR_DEFS")) == set(model.WEIGHTS)


def test_every_factor_definition_carries_a_formula():
    """A description without the arithmetic still sends the reader to the docs."""
    html = TERMINAL.read_text(encoding="utf-8")
    block = html[html.index("const FACTOR_DEFS={"):html.index("const PILLAR_DEFS={")]
    for key in _js_object(html, "FACTOR_DEFS"):
        chunk = block[block.index(key + ":"):]
        chunk = chunk[:chunk.index("},") + 1]
        assert "formula:" in chunk, f"{key} has no formula"
        assert "note:" in chunk, f"{key} has no plain-language note"
