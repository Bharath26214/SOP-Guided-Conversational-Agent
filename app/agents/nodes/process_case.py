from __future__ import annotations

from typing import Any

import logfire

from app.agents.llm import classify_turn
from app.agents.nodes.state import AgentState
from app.data_loaders.queries import get_claim, get_claims_for_party, get_policyholder, load_required_document_guideline
from app.memory import add_topic, flagged, hint, patch, recalled_refs
from app.observability import log_current_node, safe_state_attrs


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


def _claim_facts(claim: dict[str, Any]) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "case_id": claim.get("case_id"),
        "case_type": claim.get("case_type"),
        "created_at": claim.get("created_at"),
        "status": claim.get("status"),
    }
    for key in (
        "summary",
        "denial_reason",
        "documents_needed",
        "appeal_deadline",
        "expected_reimbursement_amount",
        "allowed_max_amount",
        "net_pay",
        "net_fee",
    ):
        if claim.get(key) not in (None, "", []):
            facts[key] = claim[key]
    return facts


def _guidance_facts(claim: dict[str, Any], guideline: dict[str, Any]) -> dict[str, Any]:
    docs = [str(item) for item in (claim.get("documents_needed") or []) if item]
    guidance: dict[str, Any] = {}
    default = (guideline.get("default_guidance") or {}).get("en")
    if default:
        guidance["default"] = default
    case_type = str(claim.get("case_type") or "")
    type_guide = ((guideline.get("case_type_guidance") or {}).get(case_type) or {}).get("en")
    if type_guide:
        guidance["case_type"] = type_guide
    specifics = {}
    for name in docs:
        specific = _lookup_en(guideline.get("document_guidance") or {}, name)
        if specific:
            specifics[name] = specific
        alt = _lookup_en(guideline.get("document_alternative_guidance") or {}, name)
        if alt:
            specifics[f"{name}_alternative"] = alt
    if specifics:
        guidance["documents"] = specifics
    settings = guideline.get("claim_followup_settings") or {}
    avg = (settings.get("average_processing_time_after_submission") or {}).get("en")
    if avg:
        guidance["average_processing_time_after_submission"] = avg
    followups = []
    for rule in guideline.get("claim_followup_guidance") or []:
        if rule.get("requires_documents") and not docs:
            continue
        followups.append({"topic": rule.get("topic"), "text": rule.get("en")})
    if followups:
        guidance["followups"] = followups
    fallback = (guideline.get("claim_followup_fallback") or {}).get("en")
    if fallback:
        guidance["fallback"] = fallback
    return guidance


def _facts(extra: dict[str, Any]) -> dict[str, Any]:
    return {"phase": "PROCESS_CASE", "verified": True, **extra}


def _offer_human(state: AgentState, extra: dict[str, Any] | None = None) -> tuple[AgentState, dict[str, Any]]:
    updated = patch({**state, "phase": "PROCESS_CASE"}, awaiting_human_confirm=True)
    return updated, _facts({"status": "human_offer", "out_of_scope": True, **(extra or {})})


def _handoff_to_email(state: AgentState) -> tuple[AgentState, dict[str, Any]]:
    return patch(
        {**state, "phase": "POST_PROCESS"},
        awaiting_human_confirm=False,
        awaiting_more_help=False,
    ), {}


def _bare_more_yes(user_text: str) -> bool:
    return (user_text or "").strip().lower() in {
        "yes",
        "yeah",
        "yep",
        "ok",
        "okay",
        "sure",
        "please",
    }


def run_process_case(
    state: AgentState,
    user_text: str,
    api_key: str | None = None,
) -> tuple[AgentState, dict[str, Any]]:
    log_current_node("process_case", state)
    with logfire.span("process_case.run", **safe_state_attrs(state)) as span:
        updated, facts = _run_process_case(state, user_text, api_key)
        span.set_attributes(safe_state_attrs(updated))
        span.set_attribute("selected_case_id", updated.get("selected_case_id"))
        span.set_attribute("phase", updated.get("phase"))
        return updated, facts


def _run_process_case(
    state: AgentState,
    user_text: str,
    api_key: str | None,
) -> tuple[AgentState, dict[str, Any]]:
    if not state.get("verified"):
        return state, _facts({"status": "not_verified", "verified": False})

    updated: AgentState = {**state, "phase": "PROCESS_CASE"}
    awaiting = flagged(updated, "awaiting_human_confirm")
    awaiting_more = flagged(updated, "awaiting_more_help")
    on_claim = bool(updated.get("selected_case_id") or recalled_refs(updated).get("case_id"))
    move = classify_turn(
        user_text,
        api_key,
        offered_email=False,
        awaiting_human=awaiting,
        on_claim=on_claim or flagged(updated, "case_briefed"),
        awaiting_more_help=awaiting_more,
    )

    if awaiting:
        if move == "human_yes":
            updated = patch({**updated, "phase": "HUMAN_ESCALATION"}, awaiting_human_confirm=False)
            return updated, _facts({"status": "escalate"})
        if move == "human_no":
            updated = patch(updated, awaiting_human_confirm=False, awaiting_more_help=True)
            return updated, _facts({"status": "continue_case", "ask_anything_else": True})

    if move == "human_yes":
        updated = patch({**updated, "phase": "HUMAN_ESCALATION"}, awaiting_human_confirm=False)
        return updated, _facts({"status": "escalate"})

    done_moves = {"email_send", "email_skip", "wrap_up"}
    if (awaiting_more or flagged(updated, "case_briefed")) and move in done_moves:
        return _handoff_to_email(updated)

    if awaiting_more and (move == "other" or (move == "claim_question" and _bare_more_yes(user_text))):
        updated = patch(updated, awaiting_more_help=True)
        return updated, _facts({"status": "need_more_question", "ask_anything_else": False})

    if updated.get("intent") == "policy_inquiry" and not updated.get("selected_case_id"):
        person = get_policyholder(str(updated.get("party_id") or ""))
        policy_number = (person or {}).get("policy_number") or (updated.get("identity") or {}).get(
            "policy_number"
        )
        updated = patch(updated, case_briefed=True)
        return _offer_human(updated, {"policy_number": policy_number, "status": "policy_limited"})

    refs = recalled_refs(updated)
    case_id = updated.get("selected_case_id") or refs.get("case_id")
    hint_id = str(hint(updated).get("case_id") or refs.get("case_id") or "").strip().upper()
    party_id = updated.get("party_id")
    if hint_id and hint_id != str(case_id or "").upper():
        other = get_claim(hint_id)
        if other and other.get("party_id") == party_id:
            updated["selected_case_id"] = other.get("case_id")
            updated["selected_case_type"] = other.get("case_type")
            case_id = other.get("case_id")
            updated = patch(updated, case_briefed=False)
        elif other:
            return updated, _facts(
                {
                    "status": "stored_claim_not_found",
                    "tried_case_id": hint_id,
                    "tried_policy": refs.get("policy_number"),
                }
            )

    claim = get_claim(str(case_id)) if case_id else None
    if (claim is None or claim.get("party_id") != party_id) and party_id:
        owned = get_claims_for_party(str(party_id))
        if case_id:
            return updated, _facts(
                {
                    "status": "stored_claim_not_found",
                    "tried_case_id": case_id,
                    "tried_policy": refs.get("policy_number"),
                    "known_ids": [item.get("case_id") for item in owned],
                }
            )
        if owned:
            return updated, _facts(
                {
                    "status": "need_claim_id",
                    "tried_policy": refs.get("policy_number"),
                    "known_ids": [item.get("case_id") for item in owned],
                }
            )
        else:
            return updated, _facts(
                {
                    "status": "stored_claim_not_found",
                    "tried_policy": refs.get("policy_number"),
                }
            )
    if claim is None or claim.get("party_id") != party_id:
        return updated, _facts(
            {
                "status": "need_claim_id",
                "tried_policy": refs.get("policy_number"),
            }
        )

    guideline = load_required_document_guideline()
    first_brief = not flagged(updated, "case_briefed")
    updated = add_topic(
        patch(
            updated,
            case_briefed=True,
            awaiting_human_confirm=False,
            awaiting_more_help=True,
            last_question=user_text,
        ),
        "case",
    )
    return updated, _facts(
        {
            "status": "answer_claim",
            "first_brief": first_brief,
            "ask_anything_else": True,
            "question": user_text,
            "intent": updated.get("intent"),
            "selected_case_id": updated.get("selected_case_id") or claim.get("case_id"),
            "claim": _claim_facts(claim),
            "guidance": _guidance_facts(claim, guideline),
        }
    )
