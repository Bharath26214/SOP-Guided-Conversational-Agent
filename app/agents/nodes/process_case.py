from __future__ import annotations

from typing import Any

import logfire

from app.agents.nodes.post_process import (
    looks_like_case_followup,
    wants_email_send,
    wants_email_skip,
    wants_wrap_up,
)
from app.agents.nodes.state import AgentMemory, AgentState, merge_unique
from app.data_loaders.queries import get_claim, get_policyholder, load_required_document_guideline
from app.observability import log_current_node, safe_state_attrs

HUMAN_OFFER = (
    "That question is outside what I can answer from this claim file. "
    "I can connect you with a human representative. Would you like me to do that?"
)
ESCALATION_REPLY = (
    "I am connecting you with a human representative, who can take it from here."
)
POLICY_LIMITED_REPLY = (
    "I can confirm the policy number on file, but I do not have coverage, premium, "
    "or benefit details in this claims file. I can connect you with a human representative "
    "for policy questions. Would you like me to do that?"
)
EMAIL_TEASER = (
    " If you have another question about this claim, I can help. "
    "When you are finished, I can email you a summary of this conversation."
)

IN_SCOPE_TOKENS = (
    "claim",
    "status",
    "denied",
    "denial",
    "document",
    "missing",
    "submit",
    "upload",
    "portal",
    "appeal",
    "deadline",
    "reimburse",
    "amount",
    "paid",
    "allowed",
    "net pay",
    "net fee",
    "summary",
    "resume",
    "how long",
    "how soon",
    "format",
    "confirm",
    "pathology",
    "estimate",
    "photo",
    "review",
    "processing",
    "email",
)

AMOUNT_FIELDS = (
    ("expected reimbursement", "expected_reimbursement_amount"),
    ("expected payment", "expected_reimbursement_amount"),
    ("reimbursement", "expected_reimbursement_amount"),
    ("allowed max", "allowed_max_amount"),
    ("allowed amount", "allowed_max_amount"),
    ("allowed", "allowed_max_amount"),
    ("maximum", "allowed_max_amount"),
    ("net pay", "net_pay"),
    ("paid", "net_pay"),
    ("payment", "net_pay"),
    ("net fee", "net_fee"),
)


def _memory(state: AgentState) -> AgentMemory:
    return dict(state.get("memory") or {})


def _hint(state: AgentState) -> dict[str, Any]:
    return dict(_memory(state).get("case_hint") or {})


def _join_docs(docs: list[str]) -> str:
    if not docs:
        return "the requested documents"
    if len(docs) == 1:
        return docs[0]
    if len(docs) == 2:
        return f"{docs[0]} and {docs[1]}"
    return ", ".join(docs[:-1]) + f", and {docs[-1]}"


def _lookup_en(mapping: dict[str, Any] | None, name: str) -> str | None:
    if not mapping:
        return None
    needle = name.strip().lower()
    for key, payload in mapping.items():
        key_l = str(key).lower()
        if needle == key_l or needle in key_l or key_l in needle:
            if isinstance(payload, dict):
                return payload.get("en")
            if isinstance(payload, str):
                return payload
    return None


def _claim_amount(claim: dict[str, Any], field: str) -> str | None:
    value = claim.get(field)
    if value in (None, ""):
        return None
    return str(value)


def _fill(template: str, claim: dict[str, Any], guideline: dict[str, Any]) -> str:
    docs = _join_docs(list(claim.get("documents_needed") or []))
    avg = (
        (guideline.get("claim_followup_settings") or {})
        .get("average_processing_time_after_submission") or {}
    ).get("en") or ""
    return template.format(
        case_id=claim.get("case_id") or "",
        documents=docs,
        average_processing_time_after_submission=avg,
    )


def _wants_human(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "connect me",
            "yes connect",
            "please connect",
            "human representative",
            "speak to a human",
            "talk to a human",
            "escalate",
            "yes please",
        )
    ) or lowered.strip() in {"yes", "yeah", "yep", "ok", "okay", "please"}


def _declines_human(text: str) -> bool:
    lowered = text.lower().strip()
    return lowered in {"no", "nope", "not now", "later"} or "back to my claim" in lowered


def _is_in_scope_followup(text: str, claim: dict[str, Any] | None) -> bool:
    lowered = text.lower()
    if any(token in lowered for token in IN_SCOPE_TOKENS):
        return True
    if "cl-" in lowered:
        return True
    if claim:
        for doc in claim.get("documents_needed") or []:
            if str(doc).lower() in lowered:
                return True
        if str(claim.get("case_id") or "").lower() in lowered:
            return True
    return False


def _resume_documents(claim: dict[str, Any], guideline: dict[str, Any]) -> str:
    docs = list(claim.get("documents_needed") or [])
    if not docs:
        return ""
    parts = [
        f"To resume claim {claim.get('case_id')}, we still need {_join_docs(docs)}."
    ]
    default = (guideline.get("default_guidance") or {}).get("en")
    if default:
        parts.append(default)
    case_type = str(claim.get("case_type") or "")
    type_guide = ((guideline.get("case_type_guidance") or {}).get(case_type) or {}).get("en")
    if type_guide:
        parts.append(type_guide)
    for name in docs:
        specific = _lookup_en(guideline.get("document_guidance") or {}, name)
        if specific:
            parts.append(f"For the {name}: {specific}")
    deadline = claim.get("appeal_deadline")
    if deadline:
        parts.append(f"The appeal deadline on file is {deadline}.")
    return " ".join(parts)


def _primary_answer(claim: dict[str, Any], intent: str | None, guideline: dict[str, Any]) -> str:
    parts = [
        (
            f"Claim {claim.get('case_id')} is a {claim.get('case_type')} claim created on "
            f"{claim.get('created_at')}. The status on file is {claim.get('status')}."
        )
    ]
    if claim.get("summary"):
        parts.append(str(claim["summary"]).rstrip(".") + ".")
    if intent == "claim_status":
        pass
    if claim.get("denial_reason"):
        parts.append(f"It was denied because {claim['denial_reason']}.")
    docs_section = _resume_documents(claim, guideline)
    if docs_section:
        parts.append(docs_section)
    elif intent in {None, "claim_issue", "claim_status"} and claim.get("status") == "closed":
        net_pay = _claim_amount(claim, "net_pay")
        if net_pay is not None:
            parts.append(f"The paid amount on file is {net_pay}.")
    return " ".join(parts)


def _match_followup(text: str, claim: dict[str, Any], guideline: dict[str, Any]) -> str | None:
    lowered = text.lower()
    docs = list(claim.get("documents_needed") or [])
    if any(
        token in lowered
        for token in ("don't have", "do not have", "can't get", "cannot get", "lost the", "no copy")
    ) and docs:
        alts = []
        for name in docs:
            alt = _lookup_en(guideline.get("document_alternative_guidance") or {}, name)
            if alt:
                alts.append(alt)
        if not alts:
            default_alt = (guideline.get("document_alternative_guidance") or {}).get("default") or {}
            if default_alt.get("en"):
                alts.append(default_alt["en"])
        exhausted = (
            (guideline.get("claim_followup_settings") or {})
            .get("human_review_after_document_alternatives_exhausted") or {}
        ).get("en")
        if exhausted:
            alts.append(exhausted)
        if alts:
            return " ".join(alts)

    for rule in guideline.get("claim_followup_guidance") or []:
        if rule.get("requires_documents") and not docs:
            continue
        if str(rule.get("topic") or "") == "missing_required_material_alternatives":
            continue
        phrases = [str(item).lower() for item in (rule.get("match_any") or [])]
        if phrases and any(phrase in lowered for phrase in phrases):
            return _fill(str(rule.get("en") or ""), claim, guideline)
    return None


def _amount_answer(text: str, claim: dict[str, Any]) -> str | None:
    lowered = text.lower()
    asked = any(
        token in lowered
        for token in ("how much", "amount", "paid", "reimburse", "payment", "allowed", "net pay", "net fee")
    )
    if not asked:
        return None
    matched_fields: list[tuple[str, str]] = []
    for phrase, field in AMOUNT_FIELDS:
        if phrase in lowered:
            value = _claim_amount(claim, field)
            if value is not None:
                matched_fields.append((field, value))
    if not matched_fields and ("how much" in lowered or "amount" in lowered):
        for field in (
            "expected_reimbursement_amount",
            "net_pay",
            "allowed_max_amount",
            "net_fee",
        ):
            value = _claim_amount(claim, field)
            if value is not None:
                matched_fields.append((field, value))
                break
    if not matched_fields:
        return (
            "I do not have that amount on this claim file. "
            + HUMAN_OFFER
        )
    labels = {
        "expected_reimbursement_amount": "expected reimbursement amount",
        "allowed_max_amount": "allowed maximum amount",
        "net_pay": "paid amount",
        "net_fee": "net fee",
    }
    parts = [
        f"The {labels.get(field, field)} on file for claim {claim.get('case_id')} is {value}."
        for field, value in matched_fields
    ]
    return " ".join(parts)


def _field_answer(text: str, claim: dict[str, Any], guideline: dict[str, Any]) -> str | None:
    lowered = text.lower()
    if any(token in lowered for token in ("status", "what's going on", "what is going on", "update")):
        return (
            f"Claim {claim.get('case_id')} status on file is {claim.get('status')}."
            + (" " + _resume_documents(claim, guideline) if claim.get("documents_needed") else "")
        )
    if any(token in lowered for token in ("why", "denied", "denial", "reason")) and claim.get("denial_reason"):
        parts = [f"Claim {claim.get('case_id')} was denied because {claim['denial_reason']}."]
        docs = _resume_documents(claim, guideline)
        if docs:
            parts.append(docs)
        return " ".join(parts)
    if any(
        token in lowered
        for token in ("document", "missing", "need to send", "resume", "what do i need")
    ):
        docs = _resume_documents(claim, guideline)
        if docs:
            return docs
        return f"Claim {claim.get('case_id')} does not list additional required documents on file."
    if "deadline" in lowered or "appeal" in lowered:
        deadline = claim.get("appeal_deadline")
        if deadline:
            return f"The appeal deadline on file for claim {claim.get('case_id')} is {deadline}."
        return (
            "I do not have an appeal deadline on this claim file. " + HUMAN_OFFER
        )
    if any(token in lowered for token in ("created", "date of the claim", "when was")):
        return f"Claim {claim.get('case_id')} was created on {claim.get('created_at')}."
    if "summary" in lowered or "what happened" in lowered:
        if claim.get("summary"):
            return str(claim["summary"]).rstrip(".") + "."
    return None


def _offer_human(state: AgentState, message: str) -> tuple[AgentState, str]:
    updated: AgentState = {**state}
    memory = _memory(updated)
    memory["awaiting_human_confirm"] = True
    updated["memory"] = memory
    updated["phase"] = "PROCESS_CASE"
    logfire.info("process_case offered human handoff")
    return updated, message


def _handoff_to_email(state: AgentState, memory: AgentMemory, message: str) -> tuple[AgentState, str]:
    updated: AgentState = {**state, "phase": "POST_PROCESS"}
    memory["awaiting_human_confirm"] = False
    updated["memory"] = memory
    logfire.info("process_case handed off to post_process")
    return updated, message


def _ready_for_email(state: AgentState, memory: AgentMemory) -> bool:
    return bool(state.get("verified") and memory.get("case_briefed"))


def run_process_case(state: AgentState, user_text: str) -> tuple[AgentState, str]:
    log_current_node("process_case", state)
    with logfire.span("process_case.run", **safe_state_attrs(state)) as span:
        updated, message = _run_process_case(state, user_text)
        memory = _memory(updated)
        spoken = (message or "").replace(EMAIL_TEASER, "").strip()
        if spoken and updated.get("phase") != "POST_PROCESS":
            notes = list(memory.get("case_answers") or [])
            if spoken not in notes:
                notes.append(spoken)
            memory["case_answers"] = notes[-8:]
            updated["memory"] = memory
        span.set_attributes(safe_state_attrs(updated))
        span.set_attribute("selected_case_id", updated.get("selected_case_id"))
        span.set_attribute("phase", updated.get("phase"))
        return updated, message


def _run_process_case(state: AgentState, user_text: str) -> tuple[AgentState, str]:
    if not state.get("verified"):
        return state, "I still need to verify your identity before I can discuss a claim."

    updated: AgentState = {**state, "phase": "PROCESS_CASE"}
    memory = _memory(updated)
    text = user_text or ""

    if memory.get("awaiting_human_confirm"):
        if _wants_human(text):
            updated["phase"] = "HUMAN_ESCALATION"
            memory["awaiting_human_confirm"] = False
            updated["memory"] = memory
            logfire.info("process_case escalated to human")
            return updated, ESCALATION_REPLY
        if _declines_human(text):
            memory["awaiting_human_confirm"] = False
            updated["memory"] = memory
            return updated, "Understood. I can keep helping with the claim on file. What else do you need about this case?"

    if any(
        token in text.lower()
        for token in (
            "connect me",
            "human representative",
            "speak to a human",
            "talk to a human",
            "escalate",
        )
    ):
        updated["phase"] = "HUMAN_ESCALATION"
        memory["awaiting_human_confirm"] = False
        updated["memory"] = memory
        logfire.info("process_case explicit human request")
        return updated, ESCALATION_REPLY

    if _ready_for_email(updated, memory):
        if wants_email_send(text) or wants_email_skip(text):
            return _handoff_to_email(updated, memory, "")
        if wants_wrap_up(text) and not looks_like_case_followup(text):
            return _handoff_to_email(updated, memory, "You're welcome.")

    if updated.get("intent") == "policy_inquiry" and not updated.get("selected_case_id"):
        person = get_policyholder(str(updated.get("party_id") or ""))
        policy_number = (person or {}).get("policy_number") or (updated.get("identity") or {}).get(
            "policy_number"
        )
        if policy_number:
            prefix = f"I have policy {policy_number} on file. "
        else:
            prefix = "I have your identity on file, but "
        memory["case_briefed"] = True
        updated["memory"] = memory
        return _offer_human(updated, prefix + POLICY_LIMITED_REPLY)

    case_id = updated.get("selected_case_id")
    hint_id = str(_hint(updated).get("case_id") or "").strip().upper()
    party_id = updated.get("party_id")
    if hint_id and hint_id != str(case_id or "").upper():
        other = get_claim(hint_id)
        if other and other.get("party_id") == party_id:
            updated["selected_case_id"] = other.get("case_id")
            updated["selected_case_type"] = other.get("case_type")
            case_id = other.get("case_id")
            memory["case_briefed"] = False
        elif other:
            return _offer_human(
                updated,
                "That claim ID is not on this policy. " + HUMAN_OFFER,
            )

    claim = get_claim(str(case_id)) if case_id else None
    if claim is None or claim.get("party_id") != party_id:
        return _offer_human(
            updated,
            "I do not have a selected claim I can discuss. " + HUMAN_OFFER,
        )

    guideline = load_required_document_guideline()
    topics = list(memory.get("process_topics") or [])

    if not memory.get("case_briefed"):
        answer = _primary_answer(claim, updated.get("intent"), guideline)
        memory["case_briefed"] = True
        memory["awaiting_human_confirm"] = False
        memory["process_topics"] = merge_unique(topics, ["case_brief"])
        updated["memory"] = memory
        logfire.info(
            "process_case grounded brief",
            selected_case_id=claim.get("case_id"),
            has_documents=bool(claim.get("documents_needed")),
        )
        answer = answer + EMAIL_TEASER
        if wants_email_send(text) or wants_email_skip(text):
            return _handoff_to_email(updated, memory, answer)
        if wants_wrap_up(text) and not looks_like_case_followup(text):
            return _handoff_to_email(updated, memory, answer)
        return updated, answer

    if not _is_in_scope_followup(text, claim):
        return _offer_human(updated, HUMAN_OFFER)

    amount = _amount_answer(text, claim)
    if amount:
        if HUMAN_OFFER in amount:
            return _offer_human(updated, amount)
        memory["process_topics"] = merge_unique(topics, ["amount"])
        memory["awaiting_human_confirm"] = False
        updated["memory"] = memory
        return updated, amount

    followup = _match_followup(text, claim, guideline)
    if followup:
        memory["process_topics"] = merge_unique(topics, ["followup"])
        memory["awaiting_human_confirm"] = False
        updated["memory"] = memory
        logfire.info("process_case follow-up from guidelines")
        return updated, followup

    field = _field_answer(text, claim, guideline)
    if field:
        if HUMAN_OFFER in field:
            return _offer_human(updated, field)
        memory["process_topics"] = merge_unique(topics, ["field"])
        memory["awaiting_human_confirm"] = False
        updated["memory"] = memory
        return updated, field

    fallback = (guideline.get("claim_followup_fallback") or {}).get("en")
    if fallback and _is_in_scope_followup(text, claim):
        memory["process_topics"] = merge_unique(topics, ["fallback"])
        memory["awaiting_human_confirm"] = False
        updated["memory"] = memory
        return updated, fallback

    return _offer_human(updated, HUMAN_OFFER)
