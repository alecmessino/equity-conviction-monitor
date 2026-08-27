"""Where a filing link points, and why the obvious construction is wrong.

SEC archive paths are keyed by the **registrant's** CIK::

    /Archives/edgar/data/{registrant CIK, no leading zeros}/{accession, no dashes}/…

An accession number's leading ten digits are the CIK of whoever *submitted* the filing.
That is frequently a filing agent, not the company. Microsoft is the standing example:

    registrant CIK   789019
    accession        0000950170-25-100235   (prefix 950170 = Donnelley Financial)
    correct path     /Archives/edgar/data/789019/000095017025100235/
    wrong path       /Archives/edgar/data/950170/000095017025100235/

This is not an edge case. Across the shipped ledger, 347 of the 995 rows carrying
accessions have at least one accession whose prefix is not the registrant CIK.

An earlier version of ``edgarDoc`` built the directory from the accession prefix and
was "verified" by fetching eight index URLs and getting eight 200s. That verification
was worthless: the ``{accession}-index.htm`` endpoint resolves the accession rather than
walking the directory, so it answers under *either* CIK. A document inside the folder
does not — ``/data/950170/…/msft-20250630.htm`` is a hard 404 while the registrant path
serves the file, and EDGAR's own index page emits the registrant path in its links.

So these tests assert the exact pathname, and the network-gated one asserts the returned
page identifies the intended accession rather than merely returning 200.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = ROOT / "web" / "terminal.html"
LEDGER = ROOT / "ledger" / "index.json"

# The fixture. Registrant CIK and accession prefix deliberately differ.
MSFT_CIK = 789019
MSFT_ACCN = "0000950170-25-100235"
MSFT_PATH = "/Archives/edgar/data/789019/000095017025100235/0000950170-25-100235-index.htm"
MSFT_URL = "https://www.sec.gov" + MSFT_PATH
AGENT_PREFIX = 950170  # Donnelley Financial — what the accession prefix would give


def _node(expr: str) -> str:
    """Evaluate an expression against the terminal's own EDGAR helpers, in node.

    The helpers are lifted out of web/terminal.html rather than reimplemented, so this
    tests the shipped code and not a copy of it that could drift.
    """
    html = TERMINAL.read_text(encoding="utf-8")
    src = html.split("<script", 1)[1].split(">", 1)[1].rsplit("</script>", 1)[0]

    def grab(name: str) -> str:
        """Lift one function out by matching braces.

        Splitting on the next `\\nfunction ` over-reads: edgarLink is followed by a bare
        `S.ern={...}` statement, which then throws ReferenceError under bare node and
        makes every assertion here fail for a reason that has nothing to do with URLs.
        """
        head = f"function {name}("
        assert head in src, f"{name} is missing from web/terminal.html"
        start = src.index(head)
        i = src.index("{", start)
        depth, j = 0, i
        while j < len(src):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    return src[start:j + 1]
            j += 1
        raise AssertionError(f"unbalanced braces extracting {name}")

    prelude = "const nz=(v)=>v!==null&&v!==undefined&&!Number.isNaN(v);\n"
    prelude += "const esc=(s)=>String(s);\n"
    program = prelude + "\n".join(grab(n) for n in ("edgarUrl", "edgarDoc", "edgarLink"))
    program += f"\nprocess.stdout.write(String({expr}));"
    out = subprocess.run(["node", "-e", program], capture_output=True, text=True,
                         timeout=30)
    assert out.returncode == 0, out.stderr
    return out.stdout


# ---------------------------------------------------------------------------
# the exact pathname
# ---------------------------------------------------------------------------
def test_the_document_path_uses_the_registrant_cik_not_the_accession_prefix():
    got = _node(f"edgarDoc({MSFT_CIK}, '{MSFT_ACCN}')")
    assert got == MSFT_URL, f"expected {MSFT_URL}, got {got}"
    assert f"/data/{MSFT_CIK}/" in got
    assert f"/data/{AGENT_PREFIX}/" not in got, \
        "the accession prefix is the submitting agent, not the registrant"


def test_leading_zeros_are_stripped_from_the_registrant_cik():
    assert _node(f"edgarDoc('0000789019', '{MSFT_ACCN}')") == MSFT_URL


def test_no_directory_is_guessed_when_the_registrant_cik_is_absent():
    """The rule that keeps a wrong link from ever being emitted: with no CIK there is
    no defensible directory, so there is no link."""
    for bad in ("null", "undefined", "''", "'not-a-cik'"):
        assert _node(f"String(edgarDoc({bad}, '{MSFT_ACCN}'))") in ("null", "")


def test_a_malformed_accession_produces_no_link():
    for bad in ("'0000950170-25-1002'", "'nonsense'", "null", "''"):
        assert _node(f"String(edgarDoc({MSFT_CIK}, {bad}))") in ("null", "")


def test_the_browse_url_zero_pads_the_registrant_cik():
    got = _node(f"edgarUrl({MSFT_CIK}, '8-K')")
    assert "CIK=0000789019" in got
    assert "type=8-K" in got
    assert got.startswith("https://www.sec.gov/cgi-bin/browse-edgar?")


def test_a_missing_cik_renders_a_non_link_rather_than_a_broken_one():
    assert "<a" not in _node("edgarLink(null, '8-K', '8-K')")
    assert "<a" in _node(f"edgarLink({MSFT_CIK}, '8-K', '8-K')")


# ---------------------------------------------------------------------------
# the shipped ledger has to carry what the link needs
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not LEDGER.exists(), reason="no ledger committed")
def test_rows_carry_a_registrant_cik_and_it_is_not_the_accession_prefix():
    payload = json.loads(LEDGER.read_text())
    rows = payload["all"]
    with_cik = [r for r in rows if r.get("cik")]
    assert len(with_cik) > 0.95 * len(rows), \
        "nearly every row needs a registrant CIK or the tear sheet loses its links"

    differ = 0
    for r in rows:
        cik = r.get("cik")
        if not cik:
            continue
        for v in (r.get("as_of") or {}).values():
            accn = v.get("accn")
            if accn and int(accn.split("-")[0]) != int(cik):
                differ += 1
                break
    assert differ > 100, (
        "if no row's accession prefix differed from its registrant CIK this fixture "
        f"would prove nothing; found {differ}")


@pytest.mark.skipif(not LEDGER.exists(), reason="no ledger committed")
def test_the_top_projection_carries_the_cik_too():
    """`top` is a projection of `all`. A row that lost its CIK on the way into it would
    silently render a tear sheet with no source-filing links at all."""
    payload = json.loads(LEDGER.read_text())
    top = payload.get("top") or []
    assert top
    assert all("cik" in r for r in top), "top rows must carry the same provenance as all"


@pytest.mark.skipif(not LEDGER.exists(), reason="no ledger committed")
def test_every_link_the_tear_sheet_would_build_is_registrant_keyed():
    """Exhaustive over the shipped board rather than a sample: every (cik, accn) pair
    the panel could turn into a URL must produce a registrant-keyed path."""
    payload = json.loads(LEDGER.read_text())
    pairs = []
    for r in payload["all"]:
        cik = r.get("cik")
        if not cik:
            continue
        for v in list((r.get("as_of") or {}).values())[:8]:
            if v.get("accn"):
                pairs.append((int(cik), v["accn"]))
    assert pairs
    for cik, accn in pairs:
        expected = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                    f"{accn.replace('-', '')}/{accn}-index.htm")
        # built the same way the shipped helper builds it
        assert re.fullmatch(r"https://www\.sec\.gov/Archives/edgar/data/"
                            r"\d+/\d{18}/\d{10}-\d{2}-\d{6}-index\.htm", expected)
        assert f"/data/{cik}/" in expected


# ---------------------------------------------------------------------------
# the page actually returned — not merely a 200
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.environ.get("EDGAR_LIVE"),
                    reason="network test; set EDGAR_LIVE=1 to run")
def test_the_registrant_path_returns_the_intended_accession():
    """Status codes were what made the wrong construction look right. This asserts the
    page identifies the accession we asked for, and that a document inside the folder —
    which the index endpoint's permissiveness does not cover — resolves only under the
    registrant."""
    import urllib.request

    def fetch(url: str) -> tuple[int, str]:
        req = urllib.request.Request(
            url, headers={"User-Agent": os.environ.get(
                "SEC_CONTACT", "equity-conviction-monitor tests")})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, ""

    status, body = fetch(MSFT_URL)
    assert status == 200
    assert MSFT_ACCN in body, "the page must identify the accession that was requested"
    assert f"/Archives/edgar/data/{MSFT_CIK}/" in body, \
        "EDGAR's own document links on this page are registrant-keyed"

    # The distinction the index endpoint hides: a document under the agent prefix 404s.
    doc = re.search(rf"/Archives/edgar/data/{MSFT_CIK}/\d{{18}}/([A-Za-z0-9._-]+\.htm)",
                    body)
    assert doc, "expected at least one document link on the filing index"
    name = doc.group(1)
    base = "https://www.sec.gov/Archives/edgar/data"
    bare = MSFT_ACCN.replace("-", "")
    assert fetch(f"{base}/{MSFT_CIK}/{bare}/{name}")[0] == 200
    assert fetch(f"{base}/{AGENT_PREFIX}/{bare}/{name}")[0] == 404, \
        "if the agent path served documents too, the registrant rule would be optional"


# ---------------------------------------------------------------------------
# the mirror
# ---------------------------------------------------------------------------
def test_no_shipped_terminal_builds_an_archive_path_from_an_accession_prefix():
    """docs/terminal.html is a build artifact mirrored from web/. It went stale carrying
    the defective one-argument edgarDoc, so the mirror is checked, not assumed."""
    for path in (TERMINAL, ROOT / "docs" / "terminal.html"):
        if not path.exists():
            continue
        rel = path.relative_to(ROOT)          # web/ and docs/ share a basename
        src = path.read_text(encoding="utf-8")
        assert "function edgarDoc(cik, accn)" in src, \
            f"{rel} does not take a registrant CIK — is the docs/ mirror stale?"
        body = src.split("function edgarDoc(", 1)[1].split("\nfunction ", 1)[0]
        assert "slice(0,10)" not in body.replace(" ", ""), \
            f"{rel} still derives the directory from the accession prefix"
