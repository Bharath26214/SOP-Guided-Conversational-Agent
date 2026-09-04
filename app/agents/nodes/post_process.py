from __future__ import annotations

from typing import Any

import logfire

from app.agents.llm import classify_turn
from app.agents.nodes.state import AgentState, EmailChoice
from app.memory import answers, flagged, patch
from app.agents.tools.email import send_claims_email
from app.config import DUMMY_EMAIL_FROM
from app.data_loaders.queries import get_claim, get_policyholder, load_required_document_guideline
from app.observability import log_current_node, safe_state_attrs

DONE_REPLY = "This conversation is already complete. Reset the chat if you need to start again."


def _facts(extra: dict[str, Any]) -> dict[str, Any]:
    return {"phase": "POST_PROCESS", "verified": True, **extra}


def _recipient(state: AgentState) -> tuple[str | None, str]:
    person = get_policyholder(str(state.get("party_id") or "")) or {}
    name = str(person.get("name") or (state.get("identity") or {}).get("name") or "Member")
    email = str(person.get("email") or "").strip()
    return (email or None), name


def build_conversation_summary(state: AgentState) -> tuple[str, str]:
    _, name = _recipient(state)
    case_id = state.get("selected_case_id")
    claim = get_claim(str(case_id)) if case_id else None
    guideline = load_required_document_guideline()
    notes = answers(state)
    subject = f"Summary of your claims support call - {case_id or 'your claim'}"

    lines = [
        f"Hello {name},",
        "",
        "Here is a summary of today's insurance claims support conversation.",
        "",
        "What we discussed",
    ]
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- We reviewed your request after identity verification.")

    lines.extend(["", "Claim status / outcome"])
    if claim:
        lines.append(
            f"- Claim {claim.get('case_id')} is a {claim.get('case_type')} claim "
            f"created on {claim.get('created_at')}. Status on file: {claim.get('status')}."
        )
        if claim.get("summary"):
            lines.append(f"- {str(claim['summary']).rstrip('.')}.")
        if claim.get("denial_reason"):
            lines.append(f"- Denial reason on file: {claim['denial_reason']}.")
    elif state.get("intent") == "policy_inquiry":
        lines.append("- Policy details beyond the number on file are not in this claims file.")
    else:
        lines.append("- No specific claim was selected on this call.")

    lines.extend(["", "Follow-up items / next steps"])
    if claim and claim.get("documents_needed"):
        docs = ", ".join(str(item) for item in claim["documents_needed"])
        lines.append(f"- Submit {docs} so the file can be reviewed again.")
        default = (guideline.get("default_guidance") or {}).get("en")
        if default:
            lines.append(f"- {default}")
        if claim.get("appeal_deadline"):
            lines.append(f"- Appeal deadline on file: {claim['appeal_deadline']}.")
    else:
        lines.append("- No additional follow-up items were recorded on this call.")

    lines.extend(
        [
            "",
            f"This message was sent from {DUMMY_EMAIL_FROM} by insurance claims support.",
        ]
    )
    return subject, "\n".join(lines)


def _finish(state: AgentState, choice: EmailChoice, facts: dict[str, Any]) -> tuple[AgentState, dict[str, Any]]:
    return patch({**state, "phase": "DONE", "email_choice": choice}, email_offered=True), facts


def run_post_process(
    state: AgentState,
    user_text: str,
    api_key: str | None = None,
) -> tuple[AgentState, dict[str, Any]]:
    log_current_node("post_process", state)
    with logfire.span("post_process.run", **safe_state_attrs(state)) as span:
        updated, facts = _run_post_process(state, user_text, api_key)
        span.set_attributes(safe_state_attrs(updated))
        span.set_attribute("email_choice", updated.get("email_choice"))
        span.set_attribute("phase", updated.get("phase"))
        return updated, facts


def _run_post_process(
    state: AgentState,
    user_text: str,
    api_key: str | None,
) -> tuple[AgentState, dict[str, Any]]:
    if state.get("phase") == "DONE" or state.get("email_choice") in {"send", "skip"}:
        return state, _facts({"status": "already_done"})

    if not state.get("verified"):
        return state, _facts({"status": "not_verified", "verified": False})

    offered = flagged(state, "email_offered")
    move = classify_turn(user_text, api_key, offered_email=offered, awaiting_human=False)

    if offered and move == "claim_question":
        updated: AgentState = {**state, "phase": "PROCESS_CASE"}
        return updated, {}

    if move == "email_send":
        return _send(state)
    if move == "email_skip":
        return _finish(state, "skip", _facts({"status": "skip"}))
    if offered:
        return {**state, "phase": "POST_PROCESS"}, _facts({"status": "ask_again"})
    return _offer(state)


def _offer(state: AgentState) -> tuple[AgentState, dict[str, Any]]:
    updated = patch({**state, "phase": "POST_PROCESS"}, email_offered=True)
    recipient, _ = _recipient(updated)
    return updated, _facts({"status": "offer", "has_email_on_file": bool(recipient)})


def _send(state: AgentState) -> tuple[AgentState, dict[str, Any]]:
    recipient, _ = _recipient(state)
    if not recipient:
        return _finish(state, "skip", _facts({"status": "send", "has_email_on_file": False}))
    subject, body = build_conversation_summary(state)
    send_claims_email.invoke({"to_address": recipient, "subject": subject, "body": body})
    logfire.info("post_process emailed summary", selected_case_id=state.get("selected_case_id"))
    return _finish(state, "send", _facts({"status": "send", "has_email_on_file": True}))
