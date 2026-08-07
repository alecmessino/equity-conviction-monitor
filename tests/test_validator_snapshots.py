"""Integrity of the accumulated factor history.

The snapshot series is the one artifact in this project that cannot be rebuilt. A ledger
can be regenerated from today's filings and prices; a night recorded wrong is wrong
forever, and a night never recorded is gone. These checks run before the history is
committed, which is the only moment they can still prevent damage.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from equity_monitor import snapshots
from tests.test_snapshots import row

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "validate_ledger", ROOT / "scripts" / "validate_ledger.py")
validator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validator)


def write(tmp_path, stamp, rows):
    snapshots.write(rows, str(tmp_path), on=stamp, as_of=stamp + "T23:00:00Z")
    return snapshots.snapshot_path(str(tmp_path), stamp)


def mutate(path, **changes):
    payload = json.loads(Path(path).read_text())
    payload.update(changes)
    Path(path).write_text(json.dumps(payload))


def test_a_clean_history_passes(tmp_path):
    for stamp in ("2026-08-05", "2026-08-06", "2026-08-07"):
        write(tmp_path, stamp, [row("AAPL"), row("MSFT", p_roic=0.4)])
    assert validator.check_snapshots(str(tmp_path)) == []


def test_no_snapshots_is_not_a_failure(tmp_path):
    """A first run has no history yet, and that is not a defect."""
    assert validator.check_snapshots(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# the append-only column invariant
# ---------------------------------------------------------------------------
def test_a_column_inserted_mid_list_is_caught(tmp_path):
    """The failure this check exists for.

    Snapshots are columnar, so values are positional. Inserting a column mid-list leaves
    every prior file parsing without error while reporting one factor's values under
    another factor's name — no exception, no empty column, no visible symptom, in the
    dataset the return analysis will run against.
    """
    path = write(tmp_path, "2026-08-07", [row("AAPL")])
    shifted = ["conviction", "INSERTED"] + snapshots.COLUMNS[1:]
    mutate(path, columns=shifted)
    problems = validator.check_snapshots(str(tmp_path))
    assert problems, "a mid-list column insertion passed silently"
    assert "column 1" in problems[0]
    assert "append-only" in problems[0]


def test_a_shorter_prefix_of_columns_is_accepted(tmp_path):
    """Appending is the supported change; older files carry a genuine prefix."""
    path = write(tmp_path, "2026-08-07", [row("AAPL")])
    mutate(path, columns=snapshots.COLUMNS[:-3])
    assert validator.check_snapshots(str(tmp_path)) == []


def test_a_snapshot_with_no_column_order_is_caught(tmp_path):
    path = write(tmp_path, "2026-08-07", [row("AAPL")])
    mutate(path, columns=[])
    problems = validator.check_snapshots(str(tmp_path))
    assert any("no column order" in p for p in problems)


# ---------------------------------------------------------------------------
# identity and segmentation
# ---------------------------------------------------------------------------
def test_a_snapshot_whose_inner_date_disagrees_with_its_filename_is_caught(tmp_path):
    path = write(tmp_path, "2026-08-07", [row("AAPL")])
    mutate(path, date="2026-08-01")
    problems = validator.check_snapshots(str(tmp_path))
    assert any("disagree with itself" in p for p in problems)


def test_a_snapshot_without_a_spec_hash_is_caught(tmp_path):
    """Unsegmentable history is not history; it is two models averaged together."""
    path = write(tmp_path, "2026-08-07", [row("AAPL")])
    mutate(path, spec_hash=None)
    problems = validator.check_snapshots(str(tmp_path))
    assert any("spec_hash" in p for p in problems)


def test_an_empty_snapshot_is_caught(tmp_path):
    path = write(tmp_path, "2026-08-07", [row("AAPL")])
    mutate(path, data={})
    problems = validator.check_snapshots(str(tmp_path))
    assert any("no rows" in p for p in problems)


def test_an_unreadable_snapshot_is_caught_rather_than_crashing(tmp_path):
    write(tmp_path, "2026-08-07", [row("AAPL")])
    (tmp_path / "snapshots" / "2026-08-06.json").write_text("{not json")
    problems = validator.check_snapshots(str(tmp_path))
    assert any("unreadable" in p for p in problems)


# ---------------------------------------------------------------------------
# truncated runs
# ---------------------------------------------------------------------------
def test_a_collapsed_row_count_is_caught(tmp_path):
    """A partial run that still exits zero is the shape of the original v2 failure."""
    full = [row(f"S{i:03d}") for i in range(40)]
    for stamp in ("2026-08-04", "2026-08-05", "2026-08-06"):
        write(tmp_path, stamp, full)
    write(tmp_path, "2026-08-07", full[:5])
    problems = validator.check_snapshots(str(tmp_path))
    assert any("truncated night" in p for p in problems)


def test_a_modest_drop_in_row_count_is_not_flagged(tmp_path):
    """Names legitimately drop out. The check is for collapse, not for drift."""
    full = [row(f"S{i:03d}") for i in range(40)]
    for stamp in ("2026-08-04", "2026-08-05", "2026-08-06"):
        write(tmp_path, stamp, full)
    write(tmp_path, "2026-08-07", full[:34])
    assert validator.check_snapshots(str(tmp_path)) == []


def test_row_count_collapse_needs_history_before_it_fires(tmp_path):
    """Two nights is not a baseline, and a false alarm on night two blocks a good build."""
    write(tmp_path, "2026-08-06", [row(f"S{i:03d}") for i in range(40)])
    write(tmp_path, "2026-08-07", [row("AAPL")])
    assert validator.check_snapshots(str(tmp_path)) == []
