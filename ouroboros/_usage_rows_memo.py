"""In-process read acceleration for the usage ledger, beside the substrate.

Extracted from ``usage_accounting.py`` at the perf2 P1 render-cache round for
the same reason ``_usage_rows.py`` exists: the accounting module sits at the
hard module-size gate and this layer is a self-contained seam. It carries the
ledger's two in-process caches — the validated-rows memo + render cache for
display projections, and the in-lock warm read cache the monetary write paths
use (razzant/ouroboros#129) — and it deliberately lives beside, not inside,
``usage_ledger.py``: the substrate stays cache-ignorant. In every case the
full ``_read_records_locked`` replay remains the authority and the sole owner
of quarantine; a cache can only ever change the COST of a read, never its
result, because ``_read_new_records_locked`` re-stats the file under the held
lock and refuses to resume on any doubt.

``usage_accounting`` re-binds every name here, and the implementation resolves
the substrate (``_locked``, ``_read_records_locked``, ...) through the
``usage_accounting`` namespace at call time, so the historical monkeypatch
sites (``ua._locked``, ``ua._read_records_locked``) keep governing these reads
exactly as when the code was inline.
"""
from __future__ import annotations

import collections
import copy
import logging
import pathlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from ouroboros.runtime_limits import (
    USAGE_DISPLAY_LOCK_TIMEOUT_SEC,
    USAGE_DISPLAY_REVALIDATE_AFTER_SEC,
)
from ouroboros.usage_ledger import QUARANTINE_REL, LedgerResumeState, UsageLockUnavailable

log = logging.getLogger(__name__)


def _ua():
    """The accounting namespace, resolved lazily (import cycle + test pins)."""
    from ouroboros import usage_accounting

    return usage_accounting


@dataclass
class _LedgerRowsMemo:
    """In-process cache of one drive root's per-attempt FINAL rows.

    Holds only the ``_final_rows`` dict (one row per attempt, first-occurrence
    order) plus the resume fingerprint — O(final rows), not O(ledger rows);
    superseded transition rows are not retained.

    ``renders`` is the fingerprint-keyed cache of finished display renders
    (``usage_projection``/``usage_breakdown`` bodies) computed over these rows:
    valid exactly while the rows are, so it is cleared on refold and on every
    non-empty advance, never by TTL. ``generation`` increments on those same
    two events and guards the clear-then-publish race (see ``_render_cached``).
    """

    resume: LedgerResumeState
    final_rows: Dict[str, Dict[str, Any]]
    generation: int = 0
    renders: Dict[Tuple[Any, ...], Dict[str, Any]] = field(default_factory=dict)


# Read-side memo per RESOLVED drive root. Populated and advanced only under the
# cross-process ledger lock; the module lock guards the dict itself. Write paths
# (reserve/_transition/settle/import) never touch it — they read through their
# own in-lock cache below (full ordered records, which seq assignment and
# whole-history append validation need; this memo keeps only final rows), and
# the stat + seq-continuity check on the next read is what makes a stale memo
# impossible to serve, so correctness never depends on any writer remembering
# to invalidate.
_ROWS_MEMO: Dict[str, _LedgerRowsMemo] = {}
_ROWS_MEMO_LOCK = threading.Lock()
# Display readers only (``allow_stale``): the monotonic instant before which a
# root that just lost the lock race is served from its memo without another
# attempt. Guarded by ``_ROWS_MEMO_LOCK``; the exact path never reads it.
_STALE_BACKOFF: Dict[str, float] = {}


def _stale_memo_rows(key: str) -> "Optional[Tuple[list, bool, _LedgerRowsMemo, int]]":
    """The last validated snapshot for ``key`` without the ledger lock, or None when cold.

    Copied under ``_ROWS_MEMO_LOCK``, the lock an advancing reader applies its
    batch under, so the copy is a ledger position that was really validated —
    never half of an append — paired with the generation that renders it."""
    with _ROWS_MEMO_LOCK:
        memo = _ROWS_MEMO.get(key)
        if memo is None:
            return None
        return list(memo.final_rows.values()), memo.resume.st_ino != -2, memo, memo.generation


def _memoized_final_rows(
    root: pathlib.Path,
    *,
    allow_stale: bool = False,
) -> Tuple[list, bool, "_LedgerRowsMemo", int]:
    """Validated final rows for display projections, resumed incrementally.

    Cold (or whenever the resume fingerprint is rejected — file replacement,
    size shrink, same-size rewrite, seq discontinuity, a non-row-aligned tail,
    structural corruption) this is one full ``_read_records_locked`` replay,
    which owns quarantine. Warm, it parses only the bytes appended since the
    previous read. Locks and substrate reads resolve through the
    ``usage_accounting`` namespace at call time (tests monkeypatch those
    names). Returned row dicts are shared read-only snapshots; row ORDER
    matches a from-scratch ``_final_rows`` exactly (first-occurrence order,
    updates in place), so aggregation over them is bit-identical to a fresh
    replay.

    Returns ``(rows, cacheable, memo, generation)`` — the render-cache
    transport: ``cacheable`` is False for the deliberately NON-RESUMABLE
    crash-tail fingerprint (``st_ino == -2``), whose every read stays a full
    replay, so caching a render of it would hide exactly the reads that must
    keep re-checking the torn tail. ``memo``/``generation`` let
    ``_render_cached`` publish a render computed OUTSIDE the lock only if the
    rows have not moved since.

    ``allow_stale`` is the DISPLAY-reader contract, for a thread that must not
    wait on money (the supervisor loop, a gateway request): the lock is tried
    for ``USAGE_DISPLAY_LOCK_TIMEOUT_SEC`` only, and a contended lock serves the
    last validated snapshot instead of raising — the same tuple, whose rows
    carry their own ledger position (``_marker_from_final``), which is how the
    one consumer that can act on a lagging answer, the ``state.json`` writer,
    refuses to regress. Losing the race arms ``USAGE_DISPLAY_REVALIDATE_AFTER_SEC``
    during which the lock is not tried at all. A cold memo has nothing validated
    to serve and still raises ``UsageLockUnavailable``: unknown, never zero. No
    monetary decision may pass ``allow_stale``; ``reserve_attempt`` never reads
    this memo at all.
    """
    ua = _ua()
    key = str(pathlib.Path(root).resolve(strict=False))
    if not allow_stale:
        with ua._locked(root):
            return _advance_memo_locked(ua, root, key)
    with _ROWS_MEMO_LOCK:
        backing_off = time.monotonic() < _STALE_BACKOFF.get(key, 0.0)
    stale = _stale_memo_rows(key) if backing_off else None
    if stale is not None:
        return stale
    try:
        with ua._locked(root, timeout_sec=USAGE_DISPLAY_LOCK_TIMEOUT_SEC):
            fresh = _advance_memo_locked(ua, root, key)
    except UsageLockUnavailable:
        with _ROWS_MEMO_LOCK:
            _STALE_BACKOFF[key] = time.monotonic() + USAGE_DISPLAY_REVALIDATE_AFTER_SEC
        stale = _stale_memo_rows(key)
        if stale is None:
            raise
        return stale
    with _ROWS_MEMO_LOCK:
        _STALE_BACKOFF.pop(key, None)
    return fresh


def _advance_memo_locked(ua: Any, root: pathlib.Path, key: str) -> Tuple[list, bool, "_LedgerRowsMemo", int]:
    """Refold or advance ``key``'s memo; the caller holds the ledger lock."""
    with _ROWS_MEMO_LOCK:
        memo = _ROWS_MEMO.get(key)
    advanced = ua._read_new_records_locked(root, memo.resume) if memo is not None else None
    if advanced is None:
        records = ua._read_records_locked(root)
        memo = _LedgerRowsMemo(
            resume=ua._ledger_resume_state(root, records),
            final_rows=ua._final_rows(records),
            generation=(memo.generation + 1) if memo is not None else 0,
        )
    else:
        new_records, new_resume = advanced
        if new_records:
            # The batch, the generation bump and the renders clear MUST happen
            # under _ROWS_MEMO_LOCK. The publisher in _render_cached checks the
            # generation and writes under that lock only (it never holds the
            # ledger lock), so without it the check-then-publish pair could
            # interleave with this clear — a stale render published right
            # after the clear would then serve pre-append data to every warm
            # reader until the next append. And a display reader copies the
            # rows under that lock alone (``_stale_memo_rows``), so the batch
            # lands whole or not at all for it. Lock order stays "ledger lock
            # → memo lock"; both of those take the memo lock alone, so no
            # deadlock. The refold branch needs no such section: it swaps in a
            # NEW memo object and the publisher's `is memo` identity check
            # already rejects publications against a replaced object.
            with _ROWS_MEMO_LOCK:
                for row in new_records:
                    memo.final_rows[str(row["attempt_id"])] = row
                memo.generation += 1
                memo.renders.clear()
        memo.resume = new_resume
    with _ROWS_MEMO_LOCK:
        _ROWS_MEMO[key] = memo
    cacheable = memo.resume.st_ino != -2
    return list(memo.final_rows.values()), cacheable, memo, memo.generation

def _render_cached(
    root: pathlib.Path,
    cache_key: Tuple[Any, ...],
    render: Callable[[list, bool], Dict[str, Any]],
    *,
    allow_stale: bool = False,
) -> Dict[str, Any]:
    """Serve one display render through the memo's fingerprint-keyed cache.

    The cache lives INSIDE the memo, so its lifetime is exactly the rows':
    refold and non-empty advance both replace/clear ``renders`` under the
    ledger lock. The quarantine stat happens HERE — outside the memo but after
    the row read, because that read itself may quarantine a torn tail — and the
    resulting bool joins the cache key, so an integrity change alone can never
    serve a stale render. The render itself runs OUTSIDE any lock; publication
    happens under ``_ROWS_MEMO_LOCK`` and only when the memo object and its
    generation are unchanged since the rows were read — a concurrent append
    between read and publish means the render is returned to this caller but
    never cached. Both directions hand out deep copies: the cached object is
    shared between requests, and callers (``_with_limit``/``_with_integrity``,
    gateway handlers) mutate nested buckets in place. ``allow_stale`` forwards
    the display-reader contract of ``_memoized_final_rows`` unchanged."""
    rows, cacheable, memo, generation = _memoized_final_rows(root, allow_stale=allow_stale)
    integrity_degraded = (root / QUARANTINE_REL).is_file()
    full_key = (*cache_key, integrity_degraded)
    if cacheable:
        with _ROWS_MEMO_LOCK:
            if memo.generation == generation:
                cached = memo.renders.get(full_key)
                if cached is not None:
                    return copy.deepcopy(cached)
    result = render(rows, integrity_degraded)
    if cacheable:
        key = str(pathlib.Path(root).resolve(strict=False))
        with _ROWS_MEMO_LOCK:
            if _ROWS_MEMO.get(key) is memo and memo.generation == generation:
                memo.renders[full_key] = copy.deepcopy(result)
    return result


# razzant/ouroboros#129: the in-lock write paths (reserve/settle/_transition/
# release/legacy-import) each did a full parse+validate of the whole ledger
# under the 45s monetary flock, and the file grows unboundedly. This is their
# per-process warm cache of the last validated read per drive root: the next
# in-lock read parses only the bytes appended since. It is distinct from
# ``_ROWS_MEMO`` because writers need the FULL ordered records list (seq
# assignment + whole-history append validation), not just final rows. Rows are
# shared read-only snapshots, same as the memo's.
_LEDGER_READ_CACHE: "collections.OrderedDict[str, Tuple[LedgerResumeState, list]]" = (
    collections.OrderedDict()
)
_LEDGER_READ_CACHE_LOCK = threading.Lock()
_LEDGER_READ_CACHE_MAX_ROOTS = 8


def _ledger_cache_put(key: str, value: "Tuple[LedgerResumeState, list]") -> None:
    with _LEDGER_READ_CACHE_LOCK:
        _LEDGER_READ_CACHE[key] = value
        _LEDGER_READ_CACHE.move_to_end(key)
        while len(_LEDGER_READ_CACHE) > _LEDGER_READ_CACHE_MAX_ROOTS:
            _LEDGER_READ_CACHE.popitem(last=False)


def _read_records_locked_cached(root: pathlib.Path) -> list:
    """``_read_records_locked`` with an incremental warm path. Call under the
    held ledger lock (same contract as ``_read_records_locked``)."""
    ua = _ua()
    key = str(pathlib.Path(root).resolve(strict=False))  # one slot per physical root
    with _LEDGER_READ_CACHE_LOCK:
        cached = _LEDGER_READ_CACHE.get(key)
    if cached is not None:
        resume, rows = cached
        try:
            delta = ua._read_new_records_locked(root, resume)
        except Exception:  # noqa: BLE001 — any doubt = fall back to the full read
            delta = None
        if delta is not None:
            new_rows, new_resume = delta
            merged = rows if not new_rows else [*rows, *new_rows]
            _ledger_cache_put(key, (new_resume, merged))
            return list(merged)
    records = ua._read_records_locked(root)
    try:
        resume = ua._ledger_resume_state(root, records)
        _ledger_cache_put(key, (resume, list(records)))
    except Exception:  # noqa: BLE001 — caching is best-effort; correctness is the full read
        log.debug("ledger read-cache seed failed for %s", key, exc_info=True)
        with _LEDGER_READ_CACHE_LOCK:
            _LEDGER_READ_CACHE.pop(key, None)
    return records
