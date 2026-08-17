"""Everything the terminal fetches must actually get published.

``ledger/performance.json`` was written by the nightly, committed by nobody, and copied
into the Pages artifact by nobody. The terminal fetched it inside a bare ``catch{}``, so
on the deployed site the request 404'd in silence and the alpha curve sat in its
"awaiting historical baseline" state indefinitely — indistinguishable, to anyone
looking at it, from a curve honestly waiting for its fifth day. The panel was correct,
the pipeline was correct, and the file was simply never shipped.

That is the same failure shape as the all-zero dashboard and the fabricated alpha
chart: a build that is green because nothing threw. So the coupling is pinned here
rather than left to be noticed. A new fetch in the terminal fails this test until the
workflow publishes the file it names.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = ROOT / "web" / "terminal.html"
WORKFLOW = ROOT / ".github" / "workflows" / "pages.yml"

# Directories the terminal reads under ledger/. Copied wholesale by the workflow, so
# they are checked as directories rather than by filename.
BULK = ("history", "factors", "snapshots")


def fetched() -> set:
    """Ledger files the terminal asks for by name."""
    html = TERMINAL.read_text(encoding="utf-8")
    return set(re.findall(r"fetch\(\s*base\s*\+\s*'([A-Za-z0-9_.\-]+\.json)'", html))


@pytest.fixture(scope="module")
def workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_the_terminal_fetches_something():
    """A guard on the guard: if the fetch pattern stops matching, every assertion below
    passes vacuously and the check quietly stops checking."""
    assert len(fetched()) >= 5


@pytest.mark.parametrize("name", sorted(fetched()))
def test_each_fetched_file_is_committed_and_copied(name, workflow):
    add = re.search(r"git add (.+?)\n\s*if git diff", workflow, re.S)
    assert add, "could not find the git add block in pages.yml"
    assert f"ledger/{name}" in add.group(1), (
        f"the terminal fetches {name} but the nightly never commits it — it exists on "
        f"the runner and nowhere else")
    assert re.search(rf"cp\s+ledger/{re.escape(name)}\s+site/ledger/", workflow), (
        f"the terminal fetches {name} but it is not copied into the Pages artifact, so "
        f"the deployed page requests a file that is not there")


@pytest.mark.parametrize("directory", BULK)
def test_the_bulk_directories_are_copied(directory, workflow):
    assert re.search(rf"cp -r\s+ledger/{directory}\s+site/ledger/", workflow)


def test_a_missing_ledger_file_is_not_swallowed_without_trace():
    """Every one of these fetches sits in a catch that returns null, which is right —
    one absent optional file must not blank the whole terminal. But the panels that
    consume them have to distinguish "not published" from "not enough data yet", or a
    plumbing fault reads as a data-availability message. The alpha curve is the one
    that got this wrong, so it is the one pinned."""
    html = TERMINAL.read_text(encoding="utf-8")
    assert "S.perf=await fetch" in html
    # renderAlpha's empty state must key off the payload's own fields rather than
    # treating "no payload at all" as "too few days".
    assert re.search(r"renderAlpha\s*\(\s*\)\s*\{", html)


# ---------------------------------------------------------------------------
# name collisions in a single-file terminal
# ---------------------------------------------------------------------------
def test_no_top_level_function_is_defined_twice():
    """The whole terminal is one script, so a duplicated name silently overrides the
    earlier definition and the caller gets a different function with the same signature
    shape. renderSizing() called sizeBook() and reached a pre-existing
    sizeBook(rows,{min,max,rule}) defined 350 lines later — it returned an object with
    no `rows`, and the panel rendered empty with a null-property error buried in a
    handler."""
    import collections
    script = re.search(r"<script>(.*?)</script>", TERMINAL.read_text(encoding="utf-8"), re.S).group(1)
    names = re.findall(r"^function\s+([A-Za-z_$][\w$]*)\s*\(", script, re.M)
    dupes = {n: c for n, c in collections.Counter(names).items() if c > 1}
    assert not dupes, f"defined more than once: {dupes}"
