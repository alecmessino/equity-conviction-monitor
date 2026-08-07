"""Shared HTTP plumbing: retries, gzip, polite rate limiting, on-disk caching.

Centralised so every source gets identical backoff behaviour and so a single
place governs how hard we hit third-party endpoints. SEC asks for <10 req/s and
a declaring User-Agent; we honour both.
"""
from __future__ import annotations

import gzip
import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(ROOT, ".cache")

# SEC's fair-access policy requires a User-Agent carrying a reachable contact address.
# This is enforced, not advisory: a URL-only User-Agent and a @users.noreply.github.com
# address are both rejected with 403, so there is no way to satisfy SEC without a real
# mailbox. We take it from the environment rather than hardcoding a personal address
# into source that ships in a public repository.
SEC_CONTACT_ENV = "SEC_CONTACT"
_SEC_UA_TEMPLATE = "equity-conviction-monitor/3.0 ({contact})"


def sec_user_agent() -> str:
    contact = os.environ.get(SEC_CONTACT_ENV, "").strip()
    if not contact:
        raise RuntimeError(
            f"{SEC_CONTACT_ENV} is not set. SEC's fair-access policy requires a "
            "User-Agent containing a working contact email, and returns 403 without "
            "one. Set it to an address you monitor, e.g.\n"
            f"    export {SEC_CONTACT_ENV}='you@example.com'\n"
            "In CI, set it as a repository secret of the same name."
        )
    return contact if "(" in contact else _SEC_UA_TEMPLATE.format(contact=contact)
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

_throttle_lock = threading.Lock()
_last_call: dict[str, float] = {}


def _throttle(host: str, min_interval: float) -> None:
    """Sleep just enough that consecutive calls to `host` stay under the rate cap."""
    if min_interval <= 0:
        return
    with _throttle_lock:
        prev = _last_call.get(host, 0.0)
        wait = min_interval - (time.monotonic() - prev)
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.monotonic()


def fetch(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    retries: int = 3,
    backoff: float = 1.5,
    throttle_key: str = "",
    min_interval: float = 0.0,
) -> bytes:
    """GET `url` with retries and transparent gzip decoding.

    Raises the final exception if every attempt fails — callers decide whether a
    failed source is fatal or should fall through to the next one in the chain.
    """
    hdrs = {"User-Agent": BROWSER_UA, "Accept-Encoding": "gzip"}
    hdrs.update(headers or {})
    last: Exception = RuntimeError("no attempt made")
    for attempt in range(retries):
        _throttle(throttle_key or url.split("/")[2], min_interval)
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                return body
        except urllib.error.HTTPError as exc:
            last = exc
            # 404 means "this concept/period genuinely has no data" — not worth retrying.
            if exc.code == 404:
                raise
        except Exception as exc:  # network hiccup, timeout, malformed chunk
            last = exc
        if attempt < retries - 1:
            time.sleep(backoff ** (attempt + 1))
    raise last


def fetch_json(url: str, **kw: Any) -> Any:
    return json.loads(fetch(url, **kw).decode("utf-8", "replace"))


def cached_json(cache_key: str, url: str, *, ttl_hours: float = 20.0, **kw: Any) -> Any:
    """`fetch_json` with an on-disk cache.

    The nightly job runs on a cold runner so the cache is mostly a local-development
    convenience, but it also protects against re-fetching the same SEC frame when
    several concepts share a period.
    """
    path = os.path.join(CACHE_DIR, cache_key + ".json")
    if os.path.exists(path) and ttl_hours > 0:
        age_h = (time.time() - os.path.getmtime(path)) / 3600.0
        if age_h < ttl_hours:
            try:
                with open(path) as fh:
                    return json.load(fh)
            except Exception:
                pass  # corrupt cache entry — fall through and re-fetch
    data = fetch_json(url, **kw)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)
    return data
