"""Recovery disposition is a producer fact, not a guess from an HTTP status."""
import asyncio
import json

import pytest

from ouroboros.gateway import host_service
from ouroboros.presence_admission import PresenceAdmissionError
from ouroboros.presence_runner import PresenceTurnError
from ouroboros.presence_runner import PresenceTurnGate, presence_turn_task_id, run_presence_turn
from ouroboros.presence_runner import _notify_unresolved_turn
from ouroboros.task_results import load_task_result, task_result_path, write_task_result
from tests.test_host_service_responsiveness import _answer, _event, _presence_app, _request, _turn
from tests.test_presence_delivery import _payload


def test_auth_block_and_origin_rejection_share_status_not_disposition(tmp_path):
    app, binding, _ctx = _presence_app(tmp_path, _answer)
    request = _request(app, {"binding_id": binding, "event": _event("event")})
    request.headers = {"x-skill-token": "invalid"}
    auth = asyncio.run(host_service._api_presence_turn(request))
    event = _event("event")
    event["provider"] = "wrong-provider"
    origin = asyncio.run(host_service._api_presence_turn(
        _request(app, {"binding_id": binding, "event": event})))
    assert auth.status_code == origin.status_code == 403
    assert json.loads(auth.body)["disposition"] == "blocked"
    assert json.loads(auth.body)["code"] == "presence_auth_blocked"
    assert json.loads(origin.body)["disposition"] == "rejected"
    assert json.loads(origin.body)["code"] == "presence_origin_mismatch"


@pytest.mark.parametrize("code,disposition", [
    ("chat_log_unwritable", "retry"),
    ("presence_result_missing", "retry"),
    ("presence_attachment_admission_rejected", "rejected"),
    ("presence_admission_conversation_mismatch", "rejected"),
])
def test_turn_refusals_keep_legacy_facts_and_semantic_disposition(tmp_path, code, disposition):
    manifest = [{"status": "rejected", "label": "input"}]

    def refuse(**kwargs):
        raise PresenceTurnError(code, "staged_files", attachment_manifest=manifest)

    app, binding, _ctx = _presence_app(tmp_path, refuse)
    response = asyncio.run(_turn(app, binding, "event"))
    assert response.status_code == 409
    body = json.loads(response.body)
    assert body == {"ok": False, "error": f"{code}: staged_files", "code": code,
                    "field": "staged_files", "attachment_manifest": manifest,
                    "disposition": disposition}


@pytest.mark.parametrize("code,disposition", [
    ("presence_behavior_skill_disabled", "blocked"),
    ("presence_required_capability_unavailable", "blocked"),
    ("presence_bindings_unreadable", "retry"),
    ("presence_binding_wrong_transport", "rejected"),
])
def test_admission_dispositions(tmp_path, monkeypatch, code, disposition):
    app, binding, _ctx = _presence_app(tmp_path, _answer)

    def refuse(*args):
        raise PresenceAdmissionError(code, "binding_id")

    monkeypatch.setattr(host_service, "_admit_presence", refuse)
    response = asyncio.run(_turn(app, binding, "event"))
    body = json.loads(response.body)
    assert response.status_code == 409
    assert body["code"] == code and body["field"] == "binding_id"
    assert body["disposition"] == disposition


def test_receipt_write_retry_and_conflict_do_not_share_disposition(tmp_path, monkeypatch):
    app, _binding, ctx = _presence_app(tmp_path, _answer)
    payload = _payload()
    first = asyncio.run(host_service._api_presence_delivery(_request(app, payload)))
    assert first.status_code == 200
    changed = dict(payload, text="different facts")
    conflict = asyncio.run(host_service._api_presence_delivery(_request(app, changed)))
    assert conflict.status_code == 409
    assert json.loads(conflict.body)["disposition"] == "rejected"

    def unwritable(*args):
        raise OSError("unreadable archive")

    monkeypatch.setattr(ctx.presence_deliveries, "record", unwritable)
    unavailable = asyncio.run(host_service._api_presence_delivery(_request(app, payload)))
    assert unavailable.status_code == 503
    assert json.loads(unavailable.body)["disposition"] == "retry"
    assert json.loads(unavailable.body)["code"] == "presence_receipt_unwritable"


def test_missing_work_is_rejected_without_losing_original_error(tmp_path):
    app, binding, _ctx = _presence_app(tmp_path, _answer)
    request = _request(app)
    request.path_params = {"work_ref": "missing-work"}
    request.query_params = {"binding_id": binding}
    response = asyncio.run(host_service._api_presence_work(request))
    assert response.status_code == 404
    assert json.loads(response.body) == {
        "ok": False, "error": "presence work reference not found",
        "code": "presence_work_not_found", "disposition": "rejected",
    }


@pytest.mark.parametrize("status,code,disposition", [
    ("running", "presence_attempt_outcome_unknown", "retry"),
    ("interrupted", "presence_attempt_outcome_unknown", "retry"),
    ("cancelled", "presence_turn_cancelled", "blocked"),
])
def test_lost_or_cancelled_turn_never_regenerates_and_releases_host_capacity(tmp_path, status, code, disposition):
    attempted = []

    def runner(**kwargs):
        return run_presence_turn(
            repo_dir=tmp_path, drive_root=tmp_path, gate=PresenceTurnGate(1),
            agent_factory=lambda **_kw: attempted.append("agent") or None, **kwargs,
        )

    app, binding, ctx = _presence_app(tmp_path, runner)
    turn_id = presence_turn_task_id(binding, "event")
    write_task_result(tmp_path, turn_id, status, result="prior attempt unconfirmed",
                      metadata={"source": "presence", "presence": {"binding_id": binding}})
    original = load_task_result(tmp_path, turn_id)
    for _ in range(2):
        response = asyncio.run(_turn(app, binding, "event"))
        body = json.loads(response.body)
        assert response.status_code == 409
        assert body["ok"] is False and body["code"] == code
        assert body["disposition"] == disposition and body["turn_ref"] == turn_id
        assert not body.get("text") and attempted == []
        assert ctx.presence_turns.live() == [] and not any(ctx._inflight.values())
        assert load_task_result(tmp_path, turn_id) == original


def test_unresolved_turn_asks_owner_once_and_never_addresses_correspondent(tmp_path, monkeypatch):
    from supervisor import message_bus

    task_id = "presence-unresolved"
    write_task_result(tmp_path, task_id, "running", metadata={"source": "presence"})
    deliveries = []
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: object())
    monkeypatch.setattr(message_bus, "send_with_budget", lambda *args, **kwargs: deliveries.append((args, kwargs)))
    _notify_unresolved_turn(tmp_path, task_id)
    _notify_unresolved_turn(tmp_path, task_id)
    assert len(deliveries) == 1
    args, kwargs = deliveries[0]
    assert args[0] == 1 and task_id in args[1]
    assert kwargs == {"role": "system", "system_type": "presence_recovery_required", "require_write": True}
    assert load_task_result(tmp_path, task_id)["presence_recovery_owner_notified"]


def test_unrecorded_owner_question_never_gets_a_notified_stamp(tmp_path, monkeypatch):
    from supervisor import message_bus

    task_id = "presence-unrecorded"
    write_task_result(tmp_path, task_id, "running", metadata={"source": "presence"})
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: object())
    monkeypatch.setattr(message_bus, "get_bridge", lambda: pytest.fail("unrecorded message was sent"))
    monkeypatch.setattr(message_bus, "log_chat", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk")))
    _notify_unresolved_turn(tmp_path, task_id)
    assert not load_task_result(tmp_path, task_id).get("presence_recovery_owner_notified")
    assert not (tmp_path / "logs" / "chat.jsonl").exists()


@pytest.mark.parametrize("quarantined", [False, True])
def test_unreadable_or_quarantined_result_never_becomes_a_new_turn(tmp_path, quarantined):
    invoked = []

    def runner(**kwargs):
        return run_presence_turn(
            repo_dir=tmp_path, drive_root=tmp_path, gate=PresenceTurnGate(1),
            agent_factory=lambda **_kw: invoked.append("agent") or None, **kwargs,
        )

    app, binding, ctx = _presence_app(tmp_path, runner)
    task_id = presence_turn_task_id(binding, "event")
    result_path = task_result_path(tmp_path, task_id)
    if quarantined:
        result_path = result_path.parent / "quarantine" / result_path.name
        result_path.parent.mkdir(parents=True)
    result_path.write_text("{broken", encoding="utf-8")
    for _ in range(2):
        response = asyncio.run(_turn(app, binding, "event"))
        body = json.loads(response.body)
        assert response.status_code == 409 and body["code"] == "presence_result_unreadable"
        assert body["disposition"] == "retry" and body["turn_ref"] == task_id
        assert invoked == [] and ctx.presence_turns.live() == [] and not any(ctx._inflight.values())
        assert result_path.read_text(encoding="utf-8") == "{broken"
