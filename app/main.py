from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, Literal

import logfire

from app.agents.nodes.extractor import extract_and_apply
from app.agents.nodes.post_process import DONE_REPLY, run_post_process
from app.agents.nodes.process_case import run_process_case
from app.agents.nodes.resolve_intent import run_resolve_intent
from app.agents.nodes.state import AgentState
from app.agents.nodes.verify_id import run_verify_id
from app.config import resolve_groq_api_key
from app.guardrails.rails import (
    ALREADY_ESCALATED_REPLY,
    apply_rail_to_state,
    screen_user_turn,
)
from app.observability import log_current_node, safe_state_attrs

SOP_FLOW = (
    "guardrails",
    "extractor",
    "verify_id",
    "resolve_intent",
    "process_case",
    "post_process",
)
SOP_PHASE_NODES = ("verify_id", "resolve_intent", "process_case", "post_process")
TERMINAL_PHASES = frozenset({"HUMAN_ESCALATION", "DONE"})
MAX_SAME_TURN_HOPS = len(SOP_PHASE_NODES)

HALTED_REPLY = (
    "I could not complete that turn. I can connect you with a human representative."
)

Action = Literal["stop", "advance"]
EventKind = Literal["status", "text", "done"]

NODE_STATUS = {
    "guardrails": "Checking this message…",
    "extractor": "Reading your details…",
    "verify_id": "Verifying identity…",
    "resolve_intent": "Finding the claim…",
    "process_case": "Reviewing the claim file…",
    "post_process": "Wrapping up…",
    "human_escalation": "Connecting you with a human…",
    "done": "Call complete",
}


@dataclass
class Turn:
    """One user utterance moving through the SOP."""

    user_text: str
    state: AgentState
    api_key: str
    spoken: list[str] = field(default_factory=list)
    outgoing: list[str] = field(default_factory=list)

    def speak(self, message: str) -> None:
        text = (message or "").strip()
        if text:
            self.spoken.append(text)
            self.outgoing.append(text)

    def remember(self, message: str) -> None:
        self.speak(message)

    def reply_text(self, *parts: str) -> str:
        chunks = [*(self.spoken), *(part.strip() for part in parts if part and part.strip())]
        return " ".join(chunks).strip()


@dataclass(frozen=True)
class ReplyEvent:
    """One streamed update from the brain: status, spoken text, or the final state."""

    kind: EventKind
    text: str = ""
    node: str | None = None
    state: AgentState | None = None


@dataclass(frozen=True)
class Step:
    """What a SOP step tells the brain to do next."""

    action: Action
    message: str | None = None


StepRunner = Callable[["Turn", object], Step]


def _enter(node: str, state: AgentState, span) -> None:
    span.set_attribute("current_node", node)
    log_current_node(node, state)


def _status(node: str) -> ReplyEvent:
    return ReplyEvent(kind="status", text=NODE_STATUS.get(node, node), node=node)


def _flush(turn: Turn, node: str) -> Iterator[ReplyEvent]:
    for text in turn.outgoing:
        yield ReplyEvent(kind="text", text=text, node=node)
    turn.outgoing.clear()


def _finish(span, node: str, state: AgentState, message: str) -> tuple[str, AgentState]:
    span.set_attribute("outcome", node)
    span.set_attributes(safe_state_attrs(state))
    return message, state


def _case_is_ready(state: AgentState) -> bool:
    if state.get("phase") != "PROCESS_CASE":
        return False
    return bool(state.get("selected_case_id") or state.get("intent") == "policy_inquiry")


def next_node(state: AgentState) -> str | None:
    """Pick the SOP node that should run after extraction.

    This is the routing contract. Nodes may set `phase` as a result of their
    work; only this function maps that result onto the next runner.
    """
    phase = state.get("phase")
    if phase in TERMINAL_PHASES:
        return None
    if not state.get("verified"):
        return "verify_id"
    if phase == "POST_PROCESS":
        return "post_process"
    if _case_is_ready(state):
        return "process_case"
    return "resolve_intent"


def run_guardrails(turn: Turn, span) -> Step:
    _enter("guardrails", turn.state, span)
    decision = screen_user_turn(turn.user_text, api_key=turn.api_key)
    if not decision.blocked:
        return Step("advance")
    turn.state, message = apply_rail_to_state(turn.state, decision)
    span.set_attribute("rail_source", decision.source)
    span.set_attribute("outcome", decision.category or "blocked")
    turn.speak(message)
    return Step("stop", message)


def run_extractor(turn: Turn, span) -> Step:
    _enter("extractor", turn.state, span)
    turn.state, extracted = extract_and_apply(turn.state, turn.user_text, api_key=turn.api_key)
    if extracted is None:
        _enter("human_escalation", turn.state, span)
        turn.speak(ALREADY_ESCALATED_REPLY)
        return Step("stop", ALREADY_ESCALATED_REPLY)
    turn.state = {**turn.state, "out_of_scope_count": 0}
    return Step("advance")


def run_verify(turn: Turn, span) -> Step:
    _enter("verify_id", turn.state, span)
    turn.state, message = run_verify_id(turn.state)
    turn.speak(message)
    if turn.state.get("verified"):
        return Step("advance")
    return Step("stop", message)


def run_intent(turn: Turn, span) -> Step:
    turn.state = {**turn.state, "phase": "RESOLVE_INTENT"}
    _enter("resolve_intent", turn.state, span)
    turn.state, message = run_resolve_intent(turn.state)
    turn.speak(message)
    if _case_is_ready(turn.state):
        return Step("advance")
    return Step("stop", message)


def run_case(turn: Turn, span) -> Step:
    _enter("process_case", turn.state, span)
    turn.state, message = run_process_case(turn.state, turn.user_text)
    turn.speak(message)
    if turn.state.get("phase") == "POST_PROCESS":
        return Step("advance")
    return Step("stop", message)


def run_post(turn: Turn, span) -> Step:
    _enter("post_process", turn.state, span)
    turn.state, message = run_post_process(turn.state, turn.user_text)
    turn.speak(message)
    if turn.state.get("phase") == "PROCESS_CASE":
        return Step("advance")
    return Step("stop", message)


SOP_NODES: dict[str, StepRunner] = {
    "verify_id": run_verify,
    "resolve_intent": run_intent,
    "process_case": run_case,
    "post_process": run_post,
}


def iter_reply(
    user_text: str,
    state: AgentState,
    api_key: str | None = None,
) -> Iterator[ReplyEvent]:
    """Run one SOP turn and stream status plus spoken text as nodes finish."""
    turn = Turn(
        user_text=user_text,
        state=state,
        api_key=resolve_groq_api_key(api_key),
    )

    with logfire.span("agent.reply", **safe_state_attrs(turn.state)) as span:
        span.set_attribute("sop_flow", " → ".join(SOP_FLOW))

        def done(node: str, message: str | None = None) -> Iterator[ReplyEvent]:
            if message:
                turn.speak(message)
            yield from _flush(turn, node)
            text, updated = _finish(span, node, turn.state, turn.reply_text())
            if text and not turn.spoken:
                yield ReplyEvent(kind="text", text=text, node=node)
            yield ReplyEvent(kind="done", node=node, state=updated)

        phase = turn.state.get("phase")
        if phase == "HUMAN_ESCALATION":
            _enter("human_escalation", turn.state, span)
            yield _status("human_escalation")
            yield from done("already_escalated", ALREADY_ESCALATED_REPLY)
            return
        if phase == "DONE":
            _enter("done", turn.state, span)
            yield _status("done")
            yield from done("done", DONE_REPLY)
            return

        yield _status("guardrails")
        blocked = run_guardrails(turn, span)
        if blocked.action == "stop":
            yield from done("guardrails")
            return

        yield _status("extractor")
        extracted = run_extractor(turn, span)
        if extracted.action == "stop":
            yield from done("already_escalated")
            return

        for _ in range(MAX_SAME_TURN_HOPS):
            node = next_node(turn.state)
            if node is None:
                if turn.state.get("phase") == "HUMAN_ESCALATION":
                    _enter("human_escalation", turn.state, span)
                    yield _status("human_escalation")
                    yield from done("already_escalated", ALREADY_ESCALATED_REPLY)
                    return
                if turn.state.get("phase") == "DONE":
                    _enter("done", turn.state, span)
                    yield _status("done")
                    yield from done("done", DONE_REPLY)
                    return
                yield from done("sop_halted", HALTED_REPLY)
                return

            yield _status(node)
            step = SOP_NODES[node](turn, span)
            yield from _flush(turn, node)
            if step.action == "advance":
                continue
            yield from done(node)
            return

        yield from done("sop_halted", HALTED_REPLY)


def reply(user_text: str, state: AgentState, api_key: str | None = None) -> tuple[str, AgentState]:
    """Run one SOP turn and return the assistant reply plus updated state."""
    parts: list[str] = []
    final = state
    for event in iter_reply(user_text, state, api_key=api_key):
        if event.kind == "text" and event.text.strip():
            parts.append(event.text.strip())
        elif event.kind == "done" and event.state is not None:
            final = event.state
    return " ".join(parts), final
