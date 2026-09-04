"""SOP brain — LangGraph orchestration, LangChain phrasing.

Phase order is code-gated. Nodes gather facts; a speak node writes the turn
from the SOP system prompt. After a spoken reply the graph always stops.
The next user message starts a new turn at the current phase.

    guardrails → extractor → verify_id | resolve_intent | process_case | post_process
                                                                         ↘ speak → END
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterator, Literal

import logfire
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.agents.llm import phrase
from app.agents.nodes.extractor import extract_and_apply
from app.agents.nodes.post_process import DONE_REPLY, run_post_process
from app.agents.nodes.process_case import run_process_case
from app.agents.nodes.resolve_intent import run_resolve_intent
from app.agents.nodes.state import INITIAL_STATE, AgentState
from app.agents.nodes.verify_id import run_verify_id
from app.memory import affect_payload, get as memory_get, note_answer, patch as memory_patch
from app.agents.prompts import PHASE_SYSTEM, SOP_SYSTEM
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
TERMINAL_PHASES = frozenset({"HUMAN_ESCALATION", "DONE"})
AGENT_KEYS = tuple(INITIAL_STATE.keys())

NODE_STATUS = {
    "guardrails": "Checking this message…",
    "extractor": "Reading your details…",
    "verify_id": "Verifying identity…",
    "resolve_intent": "Finding the claim…",
    "process_case": "Reviewing the claim file…",
    "post_process": "Wrapping up…",
    "speak": "Writing the reply…",
    "human_escalation": "Connecting you with a human…",
    "already_escalated": "Connecting you with a human…",
    "done": "Call complete",
}

EventKind = Literal["status", "text", "done"]


class GraphState(TypedDict, total=False):
    messages: list
    phase: str
    identity: dict
    verified: bool
    matched_pii: list
    party_id: str | None
    caller_role: str | None
    memory: dict
    selected_case_id: str | None
    selected_case_type: str | None
    candidate_case_ids: list
    intent: str | None
    out_of_scope_count: int
    email_choice: str | None
    user_text: str
    api_key: str
    last_reply: str
    speak_facts: dict
    halt: bool
    current_node: str


class ReplyEvent:
    def __init__(
        self,
        kind: EventKind,
        text: str = "",
        node: str | None = None,
        state: AgentState | None = None,
    ) -> None:
        self.kind = kind
        self.text = text
        self.node = node
        self.state = state


def _agent_state(data: dict[str, Any]) -> AgentState:
    return {key: data.get(key, INITIAL_STATE[key]) for key in AGENT_KEYS}  # type: ignore[return-value]


def _case_is_ready(state: dict[str, Any]) -> bool:
    if state.get("phase") != "PROCESS_CASE":
        return False
    return bool(state.get("selected_case_id") or state.get("intent") == "policy_inquiry")


def next_node(state: dict[str, Any]) -> str:
    phase = state.get("phase")
    if state.get("halt") or phase in TERMINAL_PHASES:
        return END
    if not state.get("verified"):
        return "verify_id"
    if phase == "POST_PROCESS":
        return "post_process"
    if _case_is_ready(state):
        return "process_case"
    return "resolve_intent"


def _enter(node: str, state: dict[str, Any]) -> None:
    log_current_node(node, state)


def _sop_result(agent: AgentState, facts: dict[str, Any], node: str) -> dict[str, Any]:
    """One spoken reply per user turn. Empty facts mean a silent handoff to the next node."""
    terminal = agent.get("phase") in TERMINAL_PHASES
    has_speech = bool(facts)
    halt = True if has_speech or terminal else False
    return {
        **agent,
        "speak_facts": {**affect_payload(agent), **facts} if has_speech else {},
        "last_reply": "",
        "halt": halt,
        "current_node": node,
    }


def guardrails_node(state: GraphState) -> dict[str, Any]:
    _enter("guardrails", state)
    decision = screen_user_turn(
        state.get("user_text") or "",
        api_key=state.get("api_key"),
        phase=str(state.get("phase") or "VERIFY_ID"),
        last_agent_reply=str(memory_get(state, "last_agent_reply") or ""),
        state=_agent_state(state),
    )
    if not decision.blocked:
        return {"halt": False, "last_reply": "", "speak_facts": {}, "current_node": "guardrails"}
    agent, message = apply_rail_to_state(_agent_state(state), decision)
    agent = memory_patch(agent, last_agent_reply=message)
    return {**agent, "halt": True, "last_reply": message, "speak_facts": {}, "current_node": "guardrails"}


def extractor_node(state: GraphState) -> dict[str, Any]:
    _enter("extractor", state)
    agent, extracted = extract_and_apply(
        _agent_state(state),
        state.get("user_text") or "",
        api_key=state.get("api_key"),
    )
    if extracted is None:
        return {
            **agent,
            "halt": True,
            "last_reply": ALREADY_ESCALATED_REPLY,
            "speak_facts": {},
            "current_node": "extractor",
        }
    return {
        **agent,
        "out_of_scope_count": 0,
        "halt": False,
        "last_reply": "",
        "speak_facts": {},
        "current_node": "extractor",
    }


def verify_id_node(state: GraphState) -> dict[str, Any]:
    _enter("verify_id", state)
    agent, facts = run_verify_id(_agent_state(state), user_text=state.get("user_text") or "")
    return _sop_result(agent, facts, "verify_id")


def resolve_intent_node(state: GraphState) -> dict[str, Any]:
    _enter("resolve_intent", state)
    agent, facts = run_resolve_intent(
        _agent_state(state),
        user_text=state.get("user_text") or "",
    )
    return _sop_result(agent, facts, "resolve_intent")


def process_case_node(state: GraphState) -> dict[str, Any]:
    _enter("process_case", state)
    agent, facts = run_process_case(
        _agent_state(state),
        state.get("user_text") or "",
        api_key=state.get("api_key"),
    )
    return _sop_result(agent, facts, "process_case")


def post_process_node(state: GraphState) -> dict[str, Any]:
    _enter("post_process", state)
    agent, facts = run_post_process(
        _agent_state(state),
        state.get("user_text") or "",
        api_key=state.get("api_key"),
    )
    return _sop_result(agent, facts, "post_process")


def speak_node(state: GraphState) -> dict[str, Any]:
    _enter("speak", state)
    facts = dict(state.get("speak_facts") or {})
    if not facts:
        return {
            "last_reply": "",
            "speak_facts": {},
            "halt": True,
            "current_node": state.get("current_node") or "speak",
        }
    phase = str(facts.get("phase") or state.get("phase") or "VERIFY_ID")
    text = phrase(PHASE_SYSTEM.get(phase, SOP_SYSTEM), facts, state.get("api_key"))
    updates: dict[str, Any] = {
        "last_reply": text,
        "speak_facts": {},
        "halt": True,
        "current_node": state.get("current_node") or "speak",
    }
    if text:
        remembered = memory_patch(state, last_agent_reply=text)
        if phase == "PROCESS_CASE":
            remembered = note_answer(remembered, text)
        updates["memory"] = remembered["memory"]
    return updates


def already_done_node(state: GraphState) -> dict[str, Any]:
    _enter("done", state)
    return {"last_reply": DONE_REPLY, "halt": True, "speak_facts": {}, "current_node": "done", "phase": "DONE"}


def already_escalated_node(state: GraphState) -> dict[str, Any]:
    _enter("human_escalation", state)
    return {
        "last_reply": ALREADY_ESCALATED_REPLY,
        "halt": True,
        "speak_facts": {},
        "current_node": "human_escalation",
        "phase": "HUMAN_ESCALATION",
    }


def route_start(state: GraphState) -> str:
    phase = state.get("phase")
    if phase == "HUMAN_ESCALATION":
        return "already_escalated"
    if phase == "DONE":
        return "already_done"
    return "guardrails"


def route_after_guardrails(state: GraphState) -> str:
    if state.get("halt"):
        return END
    return "extractor"


def route_after_extractor(state: GraphState) -> str:
    if state.get("halt"):
        return END
    return next_node(state)


def route_after_sop(state: GraphState) -> str:
    if state.get("speak_facts"):
        return "speak"
    return next_node(state)


def route_after_speak(state: GraphState) -> str:
    return END


@lru_cache(maxsize=1)
def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("guardrails", guardrails_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("verify_id", verify_id_node)
    graph.add_node("resolve_intent", resolve_intent_node)
    graph.add_node("process_case", process_case_node)
    graph.add_node("post_process", post_process_node)
    graph.add_node("speak", speak_node)
    graph.add_node("already_done", already_done_node)
    graph.add_node("already_escalated", already_escalated_node)

    graph.add_conditional_edges(START, route_start)
    graph.add_conditional_edges("guardrails", route_after_guardrails)
    graph.add_conditional_edges("extractor", route_after_extractor)
    for node in ("verify_id", "resolve_intent", "process_case", "post_process"):
        graph.add_conditional_edges(node, route_after_sop)
    graph.add_conditional_edges("speak", route_after_speak)
    graph.add_edge("already_done", END)
    graph.add_edge("already_escalated", END)
    return graph.compile()


def _graph_input(user_text: str, state: AgentState, api_key: str | None) -> GraphState:
    payload: GraphState = {**state}  # type: ignore[typeddict-item]
    payload["user_text"] = user_text
    payload["api_key"] = resolve_groq_api_key(api_key)
    payload["last_reply"] = ""
    payload["speak_facts"] = {}
    payload["halt"] = False
    payload["current_node"] = ""
    return payload


def iter_reply(
    user_text: str,
    state: AgentState,
    api_key: str | None = None,
) -> Iterator[ReplyEvent]:
    """Stream status and spoken text as LangGraph nodes finish."""
    graph = build_graph()
    incoming = _graph_input(user_text, state, api_key)
    latest: dict[str, Any] = dict(incoming)

    with logfire.span("agent.reply", **safe_state_attrs(state)) as span:
        span.set_attribute("sop_flow", " → ".join(SOP_FLOW))
        for update in graph.stream(incoming, stream_mode="updates", config={"recursion_limit": 12}):
            for node, delta in update.items():
                latest.update(delta)
                span.set_attribute("current_node", node)
                if node != "speak":
                    yield ReplyEvent("status", NODE_STATUS.get(node, node), node)
                text = str(delta.get("last_reply") or "").strip()
                if text:
                    yield ReplyEvent("text", text, str(latest.get("current_node") or node))
        final = _agent_state(latest)
        span.set_attributes(safe_state_attrs(final))
        yield ReplyEvent("done", node=str(latest.get("current_node") or ""), state=final)


def reply(user_text: str, state: AgentState, api_key: str | None = None) -> tuple[str, AgentState]:
    parts: list[str] = []
    final = state
    for event in iter_reply(user_text, state, api_key=api_key):
        if event.kind == "text" and event.text.strip():
            parts.append(event.text.strip())
        elif event.kind == "done" and event.state is not None:
            final = event.state
    return " ".join(parts), final
