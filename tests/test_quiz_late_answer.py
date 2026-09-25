"""The LATE owner quiz answer (owner decision В17a=A, retiring 30=A).

A card no longer dies with its author: after the task-done seam has flipped the
block to ``expired_terminal``, the ingress still ACCEPTS an answer, stamps
``answered_after_terminal`` for audit, and — because no mailbox will ever be
drained — delivers the recorded answer into the card's own chat as the owner's
own message through the NAMED ingress ``message_bus.accept_local_message``
(idempotent per ``client_message_id``). The ingress-side siblings of these
invariants live in ``tests/test_quiz_answer.py``; the durable evidence half is
``tests/test_quiz_history_evidence.py``.
"""

from __future__ import annotations

import json

import pytest

from ouroboros.owner_quiz import (
    STATE_ANSWERED,
    quiz_states,
    reconcile_terminal,
    record_asked,
)
from tests.test_quiz_answer import _decision_app, _post


def _late_bridge(tmp_path, monkeypatch):
    """A real bridge for the late-answer path: named ingress plus the WS echo."""
    import supervisor.message_bus as mb

    bridge = mb.LocalChatBridge()
    frames = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(mb, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mb, "_BRIDGE", bridge)
    return bridge, frames


def _inbox(bridge):
    queued = []
    while True:
        try:
            queued.append(bridge._inbox.get_nowait())
        except Exception:
            return queued


def _accepted_rows(tmp_path, client_message_id):
    return [row for row in (json.loads(line) for line in
                            (tmp_path / "logs" / "chat.jsonl").read_text().splitlines())
            if row.get("client_message_id") == client_message_id]


def test_ingress_late_answer_is_accepted_and_delivered_as_an_owner_message(tmp_path, monkeypatch):
    """Owner decision В17a=A (retiring 30=A): the card outlives its author.

    A late answer is RECORDED on the expired block (with the audit flag) and,
    because no mailbox will ever be drained, delivered into the card's own chat
    as the owner's own message through the NAMED ingress — whose canonical row
    is the acceptance receipt and whose client_message_id makes a retry rejoin
    instead of enqueueing a second time."""
    record_asked(tmp_path, "task-1", quiz_id="q1", question="Which db?",
                 options=["sqlite", "postgres"], assumption="sqlite meanwhile", chat_id=1)
    reconcile_terminal(tmp_path, "task-1")  # the task settled
    bridge, frames = _late_bridge(tmp_path, monkeypatch)
    app = _decision_app(tmp_path, monkeypatch, live_task=None)
    resp = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                       "option_index": 1, "comment": "prod parity"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["state"] == STATE_ANSWERED and body["answered_index"] == 1
    assert body["answered_after_terminal"] is True and body["forwarded"] is True
    block = quiz_states(tmp_path, "task-1")["q1"]
    assert block["state"] == STATE_ANSWERED and block["answered_after_terminal"] is True

    source_id = "quiz_late_answer:task-1:q1"
    [queued] = _inbox(bridge)
    assert queued["chat_id"] == 1 and queued["user_id"] == 1 and queued["source"] == "web"
    assert queued["client_message_id"] == source_id
    # The owner's row and bubble are the HUMAN's words (the verbatim comment),
    # never the host frame; provenance is its own field and the real
    # transport's client_surface is never substituted.
    assert queued["text"] == "prod parity"
    assert "[Owner quiz answer]" not in queued["text"]
    assert queued["task_metadata"]["late_answer"] == {"task_id": "task-1", "quiz_id": "q1"}
    assert "client_surface" not in queued["task_metadata"]
    # The same user bubble the owner's own typing produces.
    echo = [f for f in frames if f.get("type") == "chat" and f.get("role") == "user"]
    assert len(echo) == 1 and echo[0]["client_message_id"] == source_id
    assert echo[0]["chat_id"] == 1 and echo[0]["content"] == queued["text"]
    [row] = _accepted_rows(tmp_path, source_id)
    assert row["text"] == echo[0]["content"] == "prod parity"

    # A retry of the SAME request re-enters delivery; the named ingress rejoins.
    again = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                        "option_index": 1, "comment": "prod parity"})
    assert again.status_code == 200 and again.json()["duplicate"] is True
    assert again.json()["forwarded"] is True
    assert _inbox(bridge) == [] and len(_accepted_rows(tmp_path, source_id)) == 1
    assert len([f for f in frames if f.get("type") == "chat"]) == 1


def test_an_answer_the_live_task_received_is_not_forwarded_when_retried_after_it_ended(tmp_path, monkeypatch):
    """A lost HTTP response and the UI's retry of the SAME request after the task ended
    must not turn an answer the task already received into a second owner turn: delivery
    follows the persisted acceptance route (answered_after_terminal), not liveness now."""
    record_asked(tmp_path, "task-2", quiz_id="q1", question="Which db?",
                 options=["sqlite", "postgres"], assumption="sqlite meanwhile", chat_id=1)
    bridge, frames = _late_bridge(tmp_path, monkeypatch)
    live = _decision_app(tmp_path, monkeypatch, live_task={"id": "task-2"})
    first = _post(live, {"request_id": "r2", "decision_id": "quiz:task-2:q1", "option_index": 0})
    assert first.status_code == 200, first.text
    assert first.json()["duplicate"] is False and not first.json().get("forwarded")
    assert "answered_after_terminal" not in quiz_states(tmp_path, "task-2")["q1"]
    reconcile_terminal(tmp_path, "task-2")  # the task ended with the answer in its mailbox
    gone = _decision_app(tmp_path, monkeypatch, live_task=None)
    again = _post(gone, {"request_id": "r2", "decision_id": "quiz:task-2:q1", "option_index": 0})
    assert again.status_code == 200 and again.json()["duplicate"] is True
    assert not again.json().get("forwarded")
    assert _inbox(bridge) == [] and not [f for f in frames if f.get("type") == "chat"]


def test_a_relayed_late_answer_keeps_the_relaying_skill_as_its_source(tmp_path, monkeypatch):
    """A late answer relayed by a transport skill (Telegram) is the owner's message from THAT
    transport, never a web message (astra round 3)."""
    import asyncio

    from ouroboros.gateway import task_decision as td

    record_asked(tmp_path, "task-3", quiz_id="q1", question="Which db?",
                 options=["sqlite", "postgres"], assumption="sqlite meanwhile", chat_id=1)
    reconcile_terminal(tmp_path, "task-3")
    bridge, frames = _late_bridge(tmp_path, monkeypatch)
    _decision_app(tmp_path, monkeypatch, live_task=None)  # binds the ingress's task lookup
    status, body = asyncio.run(td.answer_decision(
        tmp_path, {"request_id": "r3", "decision_id": "quiz:task-3:q1", "option_index": 1},
        source="skill:telegram"))
    assert status == 200 and body["forwarded"] is True
    [queued] = _inbox(bridge)
    assert queued["source"] == "skill:telegram" and queued["chat_id"] == 1
    echo = [f for f in frames if f.get("type") == "chat" and f.get("role") == "user"]
    assert echo and echo[0].get("source") == "skill:telegram"


def test_ingress_heals_an_unreconciled_quiz_of_a_dead_task(tmp_path, monkeypatch):
    """Crash window: the author died before the task-done seam expired its open
    quiz. The ingress still heals the lifecycle first — so the accepted late
    answer carries answered_after_terminal — and no mailbox control is written
    for a task nobody will drain."""
    from ouroboros.owner_mailbox import drain_owner_entries

    record_asked(tmp_path, "task-1", quiz_id="q1", question="?", options=["A", "B"],
                 assumption="a", chat_id=1)
    _late_bridge(tmp_path, monkeypatch)
    app = _decision_app(tmp_path, monkeypatch, live_task=None)
    resp = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                       "option_index": 0})
    assert resp.status_code == 200, resp.text
    assert resp.json()["answered_after_terminal"] is True
    block = quiz_states(tmp_path, "task-1")["q1"]
    assert block["state"] == STATE_ANSWERED and block["reconciled_at"]
    assert not drain_owner_entries(tmp_path, "task-1", set())


def test_late_answer_addresses_the_task_row_when_the_block_predates_chat_ids(tmp_path, monkeypatch):
    """An older block has no stored chat_id: the delivery falls back to the same
    addressing the answer's own history row already uses."""
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "task-1", "completed", chat_id=1)
    record_asked(tmp_path, "task-1", quiz_id="q1", question="?", options=["A", "B"],
                 assumption="a")
    assert "chat_id" not in quiz_states(tmp_path, "task-1")["q1"]
    bridge, _frames = _late_bridge(tmp_path, monkeypatch)
    app = _decision_app(tmp_path, monkeypatch, live_task=None)
    resp = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                       "option_index": 0})
    assert resp.status_code == 200 and resp.json()["forwarded"] is True
    [queued] = _inbox(bridge)
    assert queued["chat_id"] == 1


@pytest.mark.parametrize("chat_id,reason", [(-1001, "a2a_chat"), (0, "hidden_chat")])
def test_late_answer_to_a_machine_or_hidden_chat_is_recorded_but_not_forwarded(
    tmp_path, monkeypatch, chat_id, reason,
):
    """Canonical chat-id policy: synthetic A2A traffic has no owner turn to
    start and the hidden partition is a destination no surface reads. The answer
    is still recorded; the reply says honestly that nothing was delivered."""
    record_asked(tmp_path, "task-1", quiz_id="q1", question="?", options=["A", "B"],
                 assumption="a", chat_id=chat_id)
    reconcile_terminal(tmp_path, "task-1")
    bridge, frames = _late_bridge(tmp_path, monkeypatch)
    app = _decision_app(tmp_path, monkeypatch, live_task=None)
    resp = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                       "option_index": 0})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answered_after_terminal"] is True and body["forwarded"] is False
    assert body["reason_code"] == reason
    assert quiz_states(tmp_path, "task-1")["q1"]["state"] == STATE_ANSWERED
    assert _inbox(bridge) == [] and not [f for f in frames if f.get("type") == "chat"]


def test_a_second_answer_to_a_settled_card_is_still_a_first_wins_409(tmp_path, monkeypatch):
    """First-wins is untouched by the late path: only the transition from
    expired to answered delivers, and a competing request learns the winner."""
    record_asked(tmp_path, "task-1", quiz_id="q1", question="?", options=["A", "B"],
                 assumption="a", chat_id=1)
    reconcile_terminal(tmp_path, "task-1")
    bridge, _frames = _late_bridge(tmp_path, monkeypatch)
    app = _decision_app(tmp_path, monkeypatch, live_task=None)
    assert _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                       "option_index": 1}).status_code == 200
    _inbox(bridge)
    loser = _post(app, {"request_id": "r2", "decision_id": "quiz:task-1:q1",
                        "option_index": 0})
    assert loser.status_code == 409
    # Never a fabricated expiry: the true state is what the card settles on.
    assert loser.json()["state"] == STATE_ANSWERED
    assert loser.json()["answered_index"] == 1
    assert _inbox(bridge) == []


def test_a_button_only_late_answer_speaks_the_pressed_option_as_the_owner(tmp_path, monkeypatch):
    """No comment: the owner's row, queued text and bubble are the pressed
    option exactly as the button showed it (the ingress refuses empty text),
    and still never the English host frame."""
    record_asked(tmp_path, "task-1", quiz_id="q1", question="Which db?",
                 options=["sqlite", "postgres"], assumption="sqlite meanwhile", chat_id=1)
    reconcile_terminal(tmp_path, "task-1")
    bridge, frames = _late_bridge(tmp_path, monkeypatch)
    app = _decision_app(tmp_path, monkeypatch, live_task=None)
    resp = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1", "option_index": 1})
    assert resp.status_code == 200 and resp.json()["forwarded"] is True
    [queued] = _inbox(bridge)
    assert queued["text"] == "2. postgres"
    [echo] = [f for f in frames if f.get("type") == "chat" and f.get("role") == "user"]
    [row] = _accepted_rows(tmp_path, "quiz_late_answer:task-1:q1")
    assert echo["content"] == row["text"] == "2. postgres"
    assert "[Owner quiz answer]" not in row["text"]


def test_a_retry_of_a_late_answer_accepted_with_the_old_frame_text_rejoins(tmp_path, monkeypatch):
    """A late answer accepted before the row carried only the owner's words has
    the host frame under the same id; a retry now rejoins that delivery instead
    of failing forever on the text mismatch or enqueueing a second owner turn."""
    import supervisor.message_bus as mb

    record_asked(tmp_path, "task-1", quiz_id="q1", question="Which db?",
                 options=["sqlite", "postgres"], assumption="sqlite meanwhile", chat_id=1)
    reconcile_terminal(tmp_path, "task-1")
    bridge, frames = _late_bridge(tmp_path, monkeypatch)
    mb.log_chat("in", 1, 1, "[Owner quiz answer] quiz q1 -- old frame", source="web",
                client_message_id="quiz_late_answer:task-1:q1", drive_root=tmp_path,
                require_write=True)
    app = _decision_app(tmp_path, monkeypatch, live_task=None)
    resp = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1", "option_index": 1})
    assert resp.status_code == 200, resp.text
    assert resp.json()["forwarded"] is True
    assert _inbox(bridge) == [] and not [f for f in frames if f.get("type") == "chat"]
    assert len(_accepted_rows(tmp_path, "quiz_late_answer:task-1:q1")) == 1


def _answered_late(tmp_path, *, option_index, comment=""):
    from ouroboros.owner_quiz import record_answered

    record_asked(tmp_path, "task-1", quiz_id="q1", question="Which db for the pilot?",
                 options=["sqlite", "postgres"], assumption="sqlite meanwhile", chat_id=1)
    reconcile_terminal(tmp_path, "task-1")
    outcome = record_answered(tmp_path, "task-1", quiz_id="q1", option_index=option_index,
                              request_id="r1", comment=comment, allow_expired=True)
    assert outcome["ok"] is True
    return outcome["block"]


def test_the_drained_late_answer_gives_the_model_the_rebuilt_frame_while_the_row_stays_human(tmp_path):
    """Project-room mailbox delivery: the entry (and the owner's row) carry only
    the owner's words; the drain rebuilds the FULL card frame from the stored
    block for the model and records the owner directive with that frame."""
    import queue as queue_mod
    from types import SimpleNamespace

    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message

    block = _answered_late(tmp_path, option_index=None, comment="neither -- use duckdb")
    assert write_owner_message(
        tmp_path, "neither -- use duckdb", "live-root", msg_id="quiz_late_answer:task-1:q1:live-root",
        client_message_id="quiz_late_answer:task-1:q1",
        late_answer={"task_id": "task-1", "quiz_id": "q1"},
    )
    [entry] = drain_owner_entries(tmp_path, "live-root", set(), include_acknowledged=True)
    assert entry["text"] == "neither -- use duckdb"
    assert entry["late_answer"] == {"task_id": "task-1", "quiz_id": "q1"}

    ctx = SimpleNamespace()
    messages = [{"role": "user", "content": "Initial requirement"}]
    _drain_incoming_messages(messages, queue_mod.Queue(), tmp_path, "live-root", None, set(),
                             owner_ctx=ctx)
    delivered = str(messages[-1]["content"])
    assert "[Owner quiz answer] quiz q1" in delivered
    assert f"asked {block['asked_at']}" in delivered and f"answered {block['answered_at']}" in delivered
    assert "Question was: Which db for the pilot?" in delivered
    assert ("The owner answered in their own words without choosing an offered option. "
            "Verbatim: neither -- use duckdb") in delivered
    [directive] = [row for row in ctx._owner_directives if row["source"] == "owner_mailbox"]
    assert "[Owner quiz answer] quiz q1" in directive["content"]
    assert "Verbatim: neither -- use duckdb" in directive["content"]
    # The steer relay's delivery fact stays the owner's own message.
    assert ctx.last_owner_delivery["text"] == "neither -- use duckdb"


def test_an_ordinary_mailbox_message_is_not_reframed(tmp_path):
    """The still-working case: a mailbox message without late_answer provenance
    reaches the model as the owner's words, unchanged."""
    import queue as queue_mod
    from types import SimpleNamespace

    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import write_owner_message

    _answered_late(tmp_path, option_index=1)
    assert write_owner_message(tmp_path, "please also fix the test", "live-root", msg_id="m1")
    ctx = SimpleNamespace()
    messages = [{"role": "user", "content": "Initial requirement"}]
    _drain_incoming_messages(messages, queue_mod.Queue(), tmp_path, "live-root", None, set(),
                             owner_ctx=ctx)
    delivered = str(messages[-1]["content"])
    assert "please also fix the test" in delivered and "[Owner quiz answer]" not in delivered


def test_an_unreadable_card_is_disclosed_with_the_owners_words(tmp_path):
    """The block is gone (evicted, or the task result unreadable): the model
    gets the owner's words plus one host line naming the quiz, never silence."""
    import queue as queue_mod
    from types import SimpleNamespace

    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import write_owner_message

    assert write_owner_message(
        tmp_path, "2. postgres", "live-root", msg_id="late-1",
        late_answer={"task_id": "task-gone", "quiz_id": "q9"},
    )
    ctx = SimpleNamespace()
    messages = [{"role": "user", "content": "Initial requirement"}]
    _drain_incoming_messages(messages, queue_mod.Queue(), tmp_path, "live-root", None, set(),
                             owner_ctx=ctx)
    delivered = str(messages[-1]["content"])
    assert "2. postgres" in delivered
    assert "answers quiz q9 of task task-gone; that card could not be read" in delivered
    assert "[Owner quiz answer]" not in delivered


def test_a_late_answer_direct_turn_starts_with_the_rebuilt_frame(tmp_path, monkeypatch):
    """Main / no live root: the late answer starts an ordinary owner turn whose
    task carries the late_answer provenance; the model's first user content (and
    therefore the run's initial owner directive) is the rebuilt frame, while the
    owner's row stays their words. A turn without that provenance is unchanged."""
    import queue as queue_mod

    import supervisor.workers as workers
    from ouroboros.context import build_user_content

    block = _answered_late(tmp_path, option_index=1, comment="prod parity")
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "get_event_q", lambda: queue_mod.Queue())

    class _Agent:
        task = None

        def handle_task(self, task):
            self.task = task
            return []

    agent = _Agent()
    workers._run_chat_task(agent, 1, "prod parity", None, task_metadata={
        "client_message_id": "quiz_late_answer:task-1:q1",
        "late_answer": {"task_id": "task-1", "quiz_id": "q1"},
    })
    content = str(build_user_content(agent.task))
    assert "[Owner quiz answer] quiz q1" in content
    assert f"answered {block['answered_at']}" in content
    assert "The owner chose option 2: postgres" in content
    assert "Owner comment (verbatim): prod parity" in content

    plain = _Agent()
    workers._run_chat_task(plain, 1, "prod parity", None, task_metadata={"client_message_id": "m-2"})
    assert plain.task["text"] == "prod parity"


def test_a_late_answer_routed_into_a_project_rooms_live_root_carries_its_provenance(tmp_path, monkeypatch):
    """End to end through the real bridge intake: a late answer in a Project room
    with exactly one live root lands in THAT root's mailbox as the owner's words
    with its typed late_answer provenance, so the drain can rebuild the frame."""
    import queue as queue_mod
    from types import SimpleNamespace

    import server
    import supervisor.message_bus as mb
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "racer")
    chat_id = int(project["chat_id"])
    record_asked(tmp_path, "task-1", quiz_id="q1", question="Which db?",
                 options=["sqlite", "postgres"], assumption="sqlite meanwhile", chat_id=chat_id)
    reconcile_terminal(tmp_path, "task-1")
    bridge, _frames = _late_bridge(tmp_path, monkeypatch)
    monkeypatch.setattr(mb, "load_state", lambda: {"session_id": "s-1"})
    app = _decision_app(tmp_path, monkeypatch, live_task=None)
    resp = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                       "option_index": 1, "comment": "prod parity"})
    assert resp.status_code == 200 and resp.json()["forwarded"] is True

    pending = [{"id": "pending-root", "chat_id": chat_id, "root_task_id": "pending-root",
                "delegation_role": "root", "drive_root": str(tmp_path)}]
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path, PENDING=pending, RUNNING={},
        load_state=lambda: {"owner_id": 1, "owner_chat_id": 1, "session_id": "s-1"},
        update_state=lambda fn: fn({"owner_id": 1, "owner_chat_id": 1}),
        consciousness=SimpleNamespace(inject_observation=lambda _text: None),
        get_chat_agent=lambda: SimpleNamespace(_busy=False),
        handle_chat_direct=lambda *a, **k: pytest.fail("mailbox delivery must not run a turn"),
        send_with_budget=lambda *a, **k: None,
    )
    monkeypatch.setattr(bridge, "send_routing_ack", lambda *a, **k: None, raising=False)
    server._process_bridge_updates(bridge, 0, ctx)

    [entry] = drain_owner_entries(tmp_path, "pending-root", set(), include_acknowledged=True)
    assert entry["text"] == "prod parity"
    assert entry["late_answer"] == {"task_id": "task-1", "quiz_id": "q1"}
    owner_ctx = SimpleNamespace()
    messages = [{"role": "user", "content": "Initial requirement"}]
    _drain_incoming_messages(messages, queue_mod.Queue(), tmp_path, "pending-root", None, set(),
                             owner_ctx=owner_ctx)
    delivered = str(messages[-1]["content"])
    assert "The owner chose option 2: postgres" in delivered
    assert "Owner comment (verbatim): prod parity" in delivered
