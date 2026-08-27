"""Whether each published ledger artifact actually refreshed, and says so if it did not.

The failure this exists to stop has already happened once. The earnings step raised a
TypeError that its own ``except`` swallowed; the run went green, every other artifact
refreshed, and ``earnings.json`` silently kept the previous night's file. A calendar of
plausible dates is indistinguishable from a current one, so nothing looked wrong — the
only tell was that one filename was missing from the refresh commit. 597 of 1,015
entries carried convictions from a board three days old.

``earnings.mark_stale`` was the narrow fix. This module is the general one: every
artifact can be flagged the same way, and every artifact's age is measured and
published so the terminal can show it without being told which files to care about.

**Age is measured against the board, not the clock.** The nightly runs at 23:00 UTC on
weekdays, so wall-clock age says "22 hours" on a Tuesday morning and "70 hours" on a
Sunday for a ledger that is behaving perfectly. Comparing each artifact to
``index.json``'s ``as_of`` instead asks the only question that has a stable answer: did
this file refresh on the same run as the board? A file that did lags by seconds. A file
that did not lags by a day or more, on a weekend as surely as on a Tuesday.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

# (filename, key its timestamp lives under, prose label)
#
# The stamp key is not uniform and cannot be made so without breaking readers:
# monitor.json has always used `generated`, and earnings.json publishes a date rather
# than a timestamp because a calendar is a thing about days. Both are declared here
# rather than guessed at read time.
#
# history.json, macro.json and the per-symbol files are keyed by symbol at the top
# level and have nowhere to put a stamp without changing the shape their consumers
# iterate over. They are deliberately absent: an unmeasurable file is better left out
# of the audit than reported as unstamped every single night.
ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    ("index.json", "as_of", "the scored board"),
    ("watchlist.json", "as_of", "overnight tier changes"),
    ("monitor.json", "generated", "model condition checks"),
    ("performance.json", "as_of", "book versus benchmark"),
    ("edge.json", "as_of", "factor edge decomposition"),
    ("health.json", "as_of", "conviction stickiness"),
    ("earnings.json", "as_of", "the earnings calendar"),
    ("trends.json", "as_of", "coverage and dispersion history"),
)

PRIMARY = "index.json"

# How far an artifact may sit behind the board before the terminal calls it lagging.
# Generous on purpose: the artifacts are written minutes apart within one run, so
# anything inside a day refreshed on the same night. A file that missed a run lands
# at 24h or more and is unambiguous.
LAG_WARN_HOURS = 24.0


def read_stamp(payload: dict, key: str) -> datetime | None:
    """Parse an artifact's timestamp, accepting both shapes the ledger publishes.

    A date-only stamp is read as the end of that day in UTC, not the start. The
    calendar is built during the 23:00 run, so anchoring '2026-08-21' to midnight
    would invent 23 hours of age on a file that is current.
    """
    raw = payload.get(key)
    if not isinstance(raw, str) or not raw:
        return None
    for fmt, end_of_day in (("%Y-%m-%dT%H:%M:%SZ", False), ("%Y-%m-%d", True)):
        try:
            got = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        return got.replace(hour=23, minute=59, second=59) if end_of_day else got
    return None


def mark_stale(path: str, *, reason: str, as_of: str) -> bool:
    """Flag an already-published artifact as stale, in place.

    A build step that fails leaves the previous file on disk, where it is
    indistinguishable from a current one. Rather than delete it — which empties the
    view and reads as "nothing to report" — or leave it silent, the payload records
    that it did not refresh and why, and the terminal says so above the data.

    The original ``as_of`` is preserved under ``built_as_of`` so the file still knows
    when it was actually built. Returns False when there is nothing to flag; never
    raises for an absent or unreadable file.
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path) as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    payload["stale"] = True
    payload["stale_reason"] = reason
    payload["stale_since"] = as_of
    payload.setdefault("built_as_of", payload.get("as_of") or payload.get("generated"))
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    os.replace(tmp, path)
    return True


def stamp(path: str, as_of: str) -> bool:
    """Record when an artifact was written, on artifacts whose writer does not.

    watchlist, performance, edge and trends each publish a dict with no timestamp in
    it. That is survivable while a human is reading them one at a time and fatal the
    moment anything tries to answer "did this refresh tonight?" — an unstamped file is
    not fresh, it is unknowable, and unknowable reads as fine. Stamped here at the call
    site rather than inside each writer so the four of them keep one obvious owner.

    Overwrites any previous stamp: this is called immediately after a successful write,
    so the value being replaced belongs to the copy that was just superseded.
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path) as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    payload["as_of"] = as_of
    # A successful rewrite is the definition of not-stale. Clearing the markers here
    # means a file that failed last night and recovered tonight stops warning, without
    # anything having to remember that it was ever flagged.
    for key in ("stale", "stale_reason", "stale_since"):
        payload.pop(key, None)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    os.replace(tmp, path)
    return True


def audit(ledger_dir: str, now: str | None = None) -> dict:
    """Age of every measurable artifact relative to the board, plus any stale flags.

    Returns a payload the terminal renders directly. ``max_lag_hours`` is the single
    number a health chip needs; ``files`` carries the detail behind it.
    """
    files = []
    primary_at: datetime | None = None

    loaded: dict[str, dict] = {}
    for name, key, label in ARTIFACTS:
        path = os.path.join(ledger_dir, name)
        payload: dict = {}
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    got = json.load(fh)
                payload = got if isinstance(got, dict) else {}
            except (OSError, ValueError):
                payload = {}
        loaded[name] = payload
        if name == PRIMARY:
            primary_at = read_stamp(payload, key)

    for name, key, label in ARTIFACTS:
        payload = loaded[name]
        path = os.path.join(ledger_dir, name)
        stamp = read_stamp(payload, key)
        lag = None
        if stamp and primary_at:
            lag = round((primary_at - stamp).total_seconds() / 3600.0, 2)
        files.append({
            "file": name,
            "label": label,
            "present": os.path.exists(path),
            "as_of": payload.get(key),
            "stamp_key": key,
            "lag_hours": lag,
            "lagging": bool(lag is not None and lag > LAG_WARN_HOURS),
            "stale": bool(payload.get("stale")),
            "stale_reason": payload.get("stale_reason"),
            "stale_since": payload.get("stale_since"),
            "built_as_of": payload.get("built_as_of"),
        })

    missing = [f["file"] for f in files if not f["present"]]
    unstamped = [f["file"] for f in files if f["present"] and f["as_of"] is None]
    stale = [f["file"] for f in files if f["stale"]]
    lagging = [f["file"] for f in files if f["lagging"]]
    lags = [f["lag_hours"] for f in files if f["lag_hours"] is not None]
    worst = max(files, key=lambda f: (f["lag_hours"] or 0)) if lags else None

    return {
        "checked_at": now,
        "primary": PRIMARY,
        "primary_as_of": loaded.get(PRIMARY, {}).get("as_of"),
        "files": files,
        "max_lag_hours": round(max(lags), 2) if lags else None,
        "worst": worst["file"] if worst and (worst["lag_hours"] or 0) > 0 else None,
        "stale": stale,
        "lagging": lagging,
        "missing": missing,
        "unstamped": unstamped,
        "ok": not stale and not lagging and not missing,
        "lag_warn_hours": LAG_WARN_HOURS,
        "basis": (
            "Each artifact's own timestamp compared against the board's. The nightly "
            "runs on weekdays, so wall-clock age reads as a failure every weekend for a "
            "ledger behaving normally; measuring against the board asks instead whether "
            "a file refreshed on the same run, which has the same answer on a Sunday as "
            "on a Tuesday. A file flagged stale did not rebuild and says why. Files "
            "keyed by symbol — history, macro, per-name factors — have nowhere to carry "
            "a timestamp and are not audited."
        ),
    }


def write_audit(ledger_dir: str, now: str | None = None) -> dict:
    """Run the audit and record it inside index.json, which the terminal always has.

    Deliberately a read-modify-write at the end of the run rather than a field set when
    index.json is first written: most artifacts are produced *after* the board, so an
    audit taken at that moment would be describing last night's copies of half of them.
    """
    report = audit(ledger_dir, now)
    path = os.path.join(ledger_dir, PRIMARY)
    if not os.path.exists(path):
        return report
    with open(path) as fh:
        payload = json.load(fh)
    payload["data_health"] = report
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    os.replace(tmp, path)
    return report
