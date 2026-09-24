"""Deterministic ordering and freshness tests for the legacy budget projection."""

from __future__ import annotations

import threading
import time
import json

import pytest


def _breakdown(value: float, marker: int) -> dict:
    return {
        "accounted_usd": value,
        "physical_calls": int(value),
        "prompt_tokens": int(value),
        "completion_tokens": 0,
        "cached_tokens": 0,
        "settled_usd": value,
        "confirmed_usd": value,
        "estimated_usd": 0.0,
        "reserved_usd": 0.0,
        "unresolved_upper_bound_usd": 0.0,
        "unknown_unmetered": 0,
        "cost_final": True,
        "attempt_counts": {"settled": int(value)},
        "integrity_degraded": False,
        "_ledger_high_water_seq": [0, marker],
    }


@pytest.mark.serial
def test_budget_projection_reads_ledger_before_state_lock(tmp_path, monkeypatch):
    from supervisor import state
    import ouroboros.usage_accounting as accounting

    state.init(tmp_path, total_budget_limit=1.0)
    holding_state_lock = False
    operations = []

    def acquire(_path, *args, **kwargs):
        nonlocal holding_state_lock
        assert not holding_state_lock
        holding_state_lock = True
        operations.append("acquire")
        return 1

    def release(_path, _fd):
        nonlocal holding_state_lock
        assert holding_state_lock
        holding_state_lock = False
        operations.append("release")

    def read_ledger(*_args, **_kwargs):
        assert not holding_state_lock, "ledger I/O must precede STATE_LOCK"
        operations.append("ledger")

    monkeypatch.setattr(state, "acquire_file_lock", acquire)
    monkeypatch.setattr(state, "release_file_lock", release)
    monkeypatch.setattr(state, "_load_state_unlocked", lambda: (operations.append("load") or {}))
    monkeypatch.setattr(
        state,
        "_save_state_unlocked",
        lambda _st: (operations.append("save"), assert_not_holding(holding_state_lock)),
    )
    monkeypatch.setattr(state, "_openrouter_ledger_settled", lambda *_a, **_k: 1.0)
    monkeypatch.setattr(accounting, "ensure_legacy_imported", read_ledger)
    monkeypatch.setattr(accounting, "usage_breakdown", lambda *_a, **_k: (read_ledger() or _breakdown(1.0, 1)))
    monkeypatch.setattr(accounting, "usage_projection", lambda *_a, **_k: (read_ledger() or {"accounted_usd": 1.0}))

    state.update_budget_from_usage({})

    assert operations[:3] == ["ledger", "ledger", "ledger"]
    assert operations[3:] == ["acquire", "load", "save", "release"]
    assert not holding_state_lock


def assert_not_holding(value):
    assert value, "state load/mutate/save must stay inside STATE_LOCK"


@pytest.mark.serial
def test_older_budget_snapshot_cannot_regress_state(tmp_path, monkeypatch):
    from supervisor import state
    import ouroboros.usage_accounting as accounting

    state.init(tmp_path, total_budget_limit=1.0)
    first_started = threading.Event()
    release_first = threading.Event()
    calls = []

    def breakdown(_root, **_display_read):
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            first_started.set()
            assert release_first.wait(2.0)
            value = 1.0
        else:
            value = 2.0
        snapshot = _breakdown(value, int(value))
        snapshot["_usage_projection"] = {
            "accounted_usd": value, "integrity_degraded": False, "cost_final": True,
        }
        return snapshot

    monkeypatch.setattr(accounting, "ensure_legacy_imported", lambda *_a, **_k: None)
    monkeypatch.setattr(accounting, "usage_breakdown", breakdown)
    older = threading.Thread(target=state.update_budget_from_usage, args=({},))
    newer = threading.Thread(target=state.update_budget_from_usage, args=({},))
    older.start()
    assert first_started.wait(2.0)
    newer.start()
    newer.join(2.0)
    release_first.set()
    older.join(2.0)

    assert state.load_state()["spent_usd"] == 2.0
    assert state.load_state()["usage_ledger_high_water_seq"] == [0, 2]


@pytest.mark.serial
def test_limited_projection_uses_breakdown_snapshot(tmp_path, monkeypatch):
    from supervisor import state
    import ouroboros.usage_accounting as accounting

    state.init(tmp_path, total_budget_limit=5.0)
    snapshot = _breakdown(1.0, 1)
    snapshot["_usage_projection"] = {
        "accounted_usd": 1.0, "integrity_degraded": False, "cost_final": True,
    }
    monkeypatch.setattr(accounting, "ensure_legacy_imported", lambda *_a, **_k: None)
    monkeypatch.setattr(accounting, "usage_breakdown", lambda *_a, **_k: dict(snapshot))
    monkeypatch.setattr(
        accounting, "usage_projection",
        lambda *_a, **_k: pytest.fail("projection must come from the breakdown snapshot"),
    )

    assert state.update_budget_from_usage({}) is True
    stored = state.load_state()
    assert stored["spent_usd"] == 1.0
    assert stored["usage_accounting"]["accounted_usd"] == 1.0


@pytest.mark.serial
def test_compaction_provenance_keeps_high_water_marker(tmp_path, monkeypatch):
    from supervisor import state
    import ouroboros.usage_accounting as accounting

    state.init(tmp_path, total_budget_limit=0.0)
    from ouroboros import usage_compaction as compaction
    request = accounting.AttemptRequest(
        model="test/model", provider="test", drive_root=tmp_path,
        task_id="task", root_task_id="root", reservation_usd=1.0,
    )
    hold = accounting.reserve_attempt(request)
    accounting.mark_dispatched(hold)
    accounting.settle_attempt(hold, {"prompt_tokens": 1}, cost_usd=1.0, cost_final=True)
    before_rows = [line for line in (tmp_path / accounting.LEDGER_REL).read_text().splitlines() if line]
    before = accounting.usage_breakdown(tmp_path)["_ledger_high_water_seq"]
    state.update_budget_from_usage({})
    monkeypatch.setattr(compaction, "_fold_clock", lambda: time.time() + 2 * compaction.USAGE_LEDGER_FOLD_MIN_AGE_SEC)
    with accounting._locked(tmp_path) as heartbeat:
        assert compaction.compact_usage_ledger_locked(tmp_path, heartbeat=heartbeat)
    after_rows = [line for line in (tmp_path / accounting.LEDGER_REL).read_text().splitlines() if line]
    after = accounting.usage_breakdown(tmp_path)["_ledger_high_water_seq"]
    state.update_budget_from_usage({})

    stored = state.load_state()
    assert len(after_rows) < len(before_rows)
    assert max(row["seq"] for row in map(json.loads, after_rows)) < max(
        row["seq"] for row in map(json.loads, before_rows)
    )
    assert tuple(after) > tuple(before)
    assert stored["spent_usd"] == 1.0
    assert stored["usage_ledger_high_water_seq"] == after


@pytest.mark.serial
def test_reordered_writers_across_real_compaction_reject_lower_epoch(tmp_path, monkeypatch):
    from supervisor import state
    import ouroboros.usage_accounting as accounting
    from ouroboros import usage_compaction as compaction

    state.init(tmp_path, total_budget_limit=0.0)

    def settle(task):
        request = accounting.AttemptRequest(
            model="test/model", provider="test", drive_root=tmp_path,
            task_id=task, root_task_id="root", reservation_usd=1.0,
        )
        hold = accounting.reserve_attempt(request)
        accounting.mark_dispatched(hold)
        accounting.settle_attempt(hold, {"prompt_tokens": 1}, cost_usd=1.0, cost_final=True)

    settle("before")
    state.update_budget_from_usage({})
    started, release = threading.Event(), threading.Event()
    real_breakdown = accounting.usage_breakdown
    calls = 0

    def delayed_breakdown(root, **display_read):
        nonlocal calls
        snapshot = real_breakdown(root, **display_read)
        calls += 1
        if calls == 1:
            started.set()
            assert release.wait(2.0)
        return snapshot

    monkeypatch.setattr(accounting, "usage_breakdown", delayed_breakdown)
    older = threading.Thread(target=state.update_budget_from_usage, args=({},))
    older.start()
    assert started.wait(2.0)

    monkeypatch.setattr(
        compaction, "_fold_clock",
        lambda: time.time() + 2 * compaction.USAGE_LEDGER_FOLD_MIN_AGE_SEC,
    )
    with accounting._locked(tmp_path) as heartbeat:
        assert compaction.compact_usage_ledger_locked(tmp_path, heartbeat=heartbeat)
    settle("after")
    newer = threading.Thread(target=state.update_budget_from_usage, args=({},))
    newer.start()
    newer.join(2.0)
    release.set()
    older.join(2.0)

    final_marker = real_breakdown(tmp_path)["_ledger_high_water_seq"]
    stored = state.load_state()
    assert stored["spent_usd"] == 2.0
    assert stored["usage_ledger_high_water_seq"] == final_marker


@pytest.mark.serial
def test_stale_snapshot_is_rejected_without_state_lock(tmp_path, monkeypatch, caplog):
    """No-lock comparison rejects proven staleness, without claiming atomicity."""
    from supervisor import state
    import ouroboros.usage_accounting as accounting

    state.init(tmp_path, total_budget_limit=0.0)
    state.save_state({"spent_usd": 2.0, "usage_ledger_high_water_seq": [0, 2]})
    monkeypatch.setattr(state, "acquire_file_lock", lambda *_a, **_k: None)
    monkeypatch.setattr(state, "release_file_lock", lambda *_a, **_k: None)
    monkeypatch.setattr(accounting, "ensure_legacy_imported", lambda *_a, **_k: None)
    monkeypatch.setattr(accounting, "usage_breakdown", lambda *_a, **_k: _breakdown(1.0, 1))

    state.update_budget_from_usage({})

    # The marker check preserves the newer projection even after a lock
    # timeout. It does not serialize two writers that both continue without
    # STATE_LOCK; that pre-existing compare/save race remains outside this fix.
    assert state.load_state()["spent_usd"] == 2.0
    assert state.load_state()["usage_ledger_high_water_seq"] == [0, 2]
    assert "STALE SNAPSHOT REJECTED" in caplog.text


@pytest.mark.serial
def test_equal_marker_writes_newer_projection_even_when_spend_decreases(tmp_path, monkeypatch):
    from supervisor import state
    import ouroboros.usage_accounting as accounting

    state.init(tmp_path, total_budget_limit=0.0)
    state.save_state({"spent_usd": 9.0, "usage_ledger_high_water_seq": [2, 4]})
    monkeypatch.setattr(accounting, "ensure_legacy_imported", lambda *_a, **_k: None)
    monkeypatch.setattr(
        accounting,
        "usage_breakdown",
        lambda *_a, **_k: _breakdown(1.0, 4) | {"_ledger_high_water_seq": [2, 4]},
    )

    state.update_budget_from_usage({})

    assert state.load_state()["spent_usd"] == 1.0
    assert state.load_state()["usage_ledger_high_water_seq"] == [2, 4]


@pytest.mark.serial
def test_lower_epoch_is_rejected_and_preserves_money(tmp_path, monkeypatch, caplog):
    from supervisor import state
    import ouroboros.usage_accounting as accounting

    state.init(tmp_path, total_budget_limit=0.0)
    state.save_state({"spent_usd": 9.0, "usage_ledger_high_water_seq": [3, 4]})
    monkeypatch.setattr(accounting, "ensure_legacy_imported", lambda *_a, **_k: None)
    monkeypatch.setattr(
        accounting,
        "usage_breakdown",
        lambda *_a, **_k: _breakdown(1.0, 2) | {"_ledger_high_water_seq": [2, 2]},
    )

    state.update_budget_from_usage({})

    assert state.load_state()["spent_usd"] == 9.0
    assert state.load_state()["usage_ledger_high_water_seq"] == [3, 4]
    assert "STALE SNAPSHOT REJECTED" in caplog.text


@pytest.mark.serial
def test_missing_marker_fails_safe_without_fabricating_zero(tmp_path, monkeypatch, caplog):
    from supervisor import state
    import ouroboros.usage_accounting as accounting

    state.init(tmp_path, total_budget_limit=0.0)
    monkeypatch.setattr(accounting, "ensure_legacy_imported", lambda *_a, **_k: None)
    monkeypatch.setattr(accounting, "usage_breakdown", lambda *_a, **_k: {"accounted_usd": 9.0})

    state.update_budget_from_usage({})

    stored = state.load_state()
    assert stored["spent_usd"] == 0.0
    assert "usage_ledger_high_water_seq" not in stored
    assert "FRESHNESS MARKER UNKNOWN" in caplog.text

    state.save_state({"spent_usd": 7.0, "usage_ledger_high_water_seq": "corrupt"})
    state.update_budget_from_usage({})
    assert state.load_state()["spent_usd"] == 7.0


@pytest.mark.serial
def test_malformed_current_marker_is_unknown(tmp_path, monkeypatch):
    from supervisor import state
    import ouroboros.usage_accounting as accounting

    state.init(tmp_path, total_budget_limit=0.0)
    state.save_state({"spent_usd": 7.0, "usage_ledger_high_water_seq": [1, 3]})
    monkeypatch.setattr(accounting, "ensure_legacy_imported", lambda *_a, **_k: None)
    monkeypatch.setattr(
        accounting,
        "usage_breakdown",
        lambda *_a, **_k: _breakdown(1.0, 4) | {"_ledger_high_water_seq": ["bad", 4]},
    )

    state.update_budget_from_usage({})

    assert state.load_state()["spent_usd"] == 7.0
    assert state.load_state()["usage_ledger_high_water_seq"] == [1, 3]


@pytest.mark.serial
def test_corrupt_ledger_header_is_unknown_not_zero(tmp_path):
    from supervisor import state
    from ouroboros.usage_ledger import LEDGER_REL

    state.init(tmp_path, total_budget_limit=0.0)
    state.save_state({"spent_usd": 7.0, "usage_ledger_high_water_seq": [1, 3]})
    ledger_path = tmp_path / LEDGER_REL
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_bytes(b"{not-json}\n")

    state.update_budget_from_usage({})

    assert state.load_state()["spent_usd"] == 7.0
    assert state.load_state()["usage_ledger_high_water_seq"] == [1, 3]
