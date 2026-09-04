"""Call memory attached to AgentState['memory']. Nodes read and patch through here."""

from __future__ import annotations

from typing import Any, Mapping

from app.agents.emotion import Affect, detect_affect, tone_for, wants_human
from app.memory.merge import merge_maps, merge_unique
from app.memory.types import CASE_ANSWER_LIMIT, AgentMemory, CaseHint

State = Mapping[str, Any]


def load(state: State | None) -> AgentMemory:
    return dict((state or {}).get("memory") or {})


def save(state: State, memory: AgentMemory) -> dict[str, Any]:
    return {**dict(state), "memory": memory}


def patch(state: State, **fields: Any) -> dict[str, Any]:
    memory = load(state)
    memory.update(fields)
    return save(state, memory)


def get(state: State | None, key: str, default: Any = None) -> Any:
    return load(state).get(key, default)


def flagged(state: State | None, key: str) -> bool:
    return bool(load(state).get(key))


def hint(state: State | None) -> CaseHint:
    return dict(load(state).get("case_hint") or {})


def intent_hints(state: State | None) -> list[str]:
    return [str(item).lower() for item in (load(state).get("intent_hints") or [])]


def answers(state: State | None) -> list[str]:
    return [str(item).strip() for item in (load(state).get("case_answers") or []) if str(item).strip()]


def record_affect(state: State, user_text: str) -> dict[str, Any]:
    affect: Affect = detect_affect(user_text)
    memory = load(state)
    attempts = int(memory.get("persuasion_attempts") or 0)
    if affect != "calm":
        attempts += 1
    force_human = wants_human(user_text)
    memory["affect"] = affect
    memory["persuasion_attempts"] = attempts
    memory["affect_tone"] = tone_for(affect, attempts, force_human=force_human)
    return save(state, memory)


def affect_payload(state: State | None) -> dict[str, Any]:
    memory = load(state)
    return {
        "affect": memory.get("affect") or "calm",
        "affect_tone": memory.get("affect_tone") or "warm",
        "persuasion_attempts": int(memory.get("persuasion_attempts") or 0),
    }


def recalled_refs(state: State | None) -> dict[str, str | None]:
    """Policy and claim ids already given, from identity or earlier turns."""
    identity = dict((state or {}).get("identity") or {})
    memory = load(state)
    stored_hint = hint(state)
    policy = (
        identity.get("policy_number")
        or memory.get("policy_number")
        or ""
    )
    case_id = (
        stored_hint.get("case_id")
        or memory.get("case_id")
        or (state or {}).get("selected_case_id")
        or ""
    )
    return {
        "policy_number": str(policy).strip().upper() or None,
        "case_id": str(case_id).strip().upper() or None,
    }


def record_extraction(
    state: State,
    *,
    representative_name: str | None = None,
    relationship: str | None = None,
    declined_ssn: bool | None = None,
    policy_number: str | None = None,
    case_hint: dict[str, Any] | None = None,
    intent_hints_in: list[str] | None = None,
) -> dict[str, Any]:
    memory = load(state)
    if representative_name:
        memory["representative_name"] = representative_name
    if relationship:
        memory["relationship"] = relationship
    if declined_ssn is True:
        memory["declined_ssn"] = True
    elif declined_ssn is False:
        memory["declined_ssn"] = False
    if policy_number:
        memory["policy_number"] = str(policy_number).strip().upper()
    if case_hint:
        memory["case_hint"] = merge_maps(memory.get("case_hint"), case_hint)
        if case_hint.get("case_id"):
            memory["case_id"] = str(case_hint["case_id"]).strip().upper()
    if intent_hints_in:
        memory["intent_hints"] = merge_unique(memory.get("intent_hints"), intent_hints_in)
    return save(state, memory)


def add_topic(state: State, topic: str) -> dict[str, Any]:
    memory = load(state)
    memory["process_topics"] = merge_unique(list(memory.get("process_topics") or []), [topic])
    return save(state, memory)


def note_answer(state: State, text: str, *, limit: int = CASE_ANSWER_LIMIT) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        return dict(state)
    memory = load(state)
    notes = list(memory.get("case_answers") or [])
    if cleaned not in notes:
        notes.append(cleaned)
    memory["case_answers"] = notes[-limit:]
    return save(state, memory)
