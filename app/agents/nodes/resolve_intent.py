from __future__ import annotations

from datetime import datetime
from typing import Any

import logfire

from app.agents.nodes.state import AgentState
from app.data_loaders.queries import get_claim, get_claims_for_party
from app.memory import hint, intent_hints, recalled_refs
from app.observability import log_current_node, safe_state_attrs
MONTHS = {
    "january": "01",
    "jan": "01",
    "february": "02",
    "feb": "02",
    "march": "03",
    "mar": "03",
    "april": "04",
    "apr": "04",
    "may": "05",
    "june": "06",
    "jun": "06",
    "july": "07",
    "jul": "07",
    "august": "08",
    "aug": "08",
    "september": "09",
    "sep": "09",
    "sept": "09",
    "october": "10",
    "oct": "10",
    "november": "11",
    "nov": "11",
    "december": "12",
    "dec": "12",
}


def _hint(state: AgentState) -> dict[str, Any]:
    return hint(state)


def _intent_hints(state: AgentState) -> list[str]:
    return intent_hints(state)


def _normalize_month(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().lower()
    if text in MONTHS:
        return MONTHS[text]
    if text.isdigit() and 1 <= int(text) <= 12:
        return f"{int(text):02d}"
    try:
        return datetime.strptime(text, "%B").strftime("%m")
    except ValueError:
        return None


def _claim_month(claim: dict[str, Any]) -> str:
    created = str(claim.get("created_at") or "")
    return created[5:7] if len(created) >= 7 else ""


def _claim_year(claim: dict[str, Any]) -> str:
    created = str(claim.get("created_at") or "")
    return created[:4] if len(created) >= 4 else ""


def _caller_named_claim(hint: dict[str, Any]) -> bool:
    return any(hint.get(field) for field in ("case_id", "case_type", "status", "month", "year"))


def _confirms_listed_claim(user_text: str) -> bool:
    text = (user_text or "").strip().lower().rstrip(".!")
    return text in {
        "yes",
        "yeah",
        "yep",
        "correct",
        "confirm",
        "that one",
        "that's the one",
        "thats the one",
        "that's it",
        "that is the one",
    }


def infer_intent(state: AgentState) -> str | None:
    if state.get("intent"):
        return str(state["intent"])
    hint = _hint(state)
    hints = _intent_hints(state)
    summary = str(hint.get("summary_text") or "").lower()
    blob = " ".join(
        [
            summary,
            " ".join(hints),
            str(hint.get("status") or "").lower(),
        ]
    )
    if any(
        token in blob
        for token in (
            "policy question",
            "about my policy",
            "coverage",
            "deductible",
            "premium",
            "policy details",
        )
    ) and not (hint.get("case_id") or hint.get("case_type") or hint.get("status")):
        return "policy_inquiry"
    if "policy_question" in hints and not hint.get("case_id"):
        return "policy_inquiry"
    if (
        "status_inquiry" in hints
        or "claim status" in blob
        or "status of my claim" in blob
        or "what's the status" in summary
        or "what is the status" in summary
    ):
        return "claim_status"
    if (
        hint.get("case_id")
        or hint.get("case_type")
        or hint.get("status")
        or hint.get("month")
        or hint.get("summary_text")
        or any(item in hints for item in ("denial_question", "document_submission", "next_steps", "claim_issue"))
    ):
        return "claim_issue"
    return None


def filter_claims(claims: list[dict[str, Any]], hint: dict[str, Any]) -> list[dict[str, Any]]:
    results = list(claims)
    case_id = str(hint.get("case_id") or "").strip().upper()
    if case_id:
        return [claim for claim in results if str(claim.get("case_id") or "").upper() == case_id]
    case_type = str(hint.get("case_type") or "").strip().lower()
    if case_type:
        results = [
            claim for claim in results if str(claim.get("case_type") or "").lower() == case_type
        ]
    status = str(hint.get("status") or "").strip().lower()
    if status:
        results = [claim for claim in results if str(claim.get("status") or "").lower() == status]
    month = _normalize_month(hint.get("month"))
    if month:
        results = [claim for claim in results if _claim_month(claim) == month]
    year = str(hint.get("year") or "").strip()
    if year:
        results = [claim for claim in results if _claim_year(claim) == year]
    return results


def _brief(claim: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": claim.get("case_id"),
        "case_type": claim.get("case_type"),
        "status": claim.get("status"),
        "created_at": claim.get("created_at"),
    }


def _facts(state: AgentState | None, extra: dict[str, Any]) -> dict[str, Any]:
    refs = recalled_refs(state)
    return {
        "phase": "RESOLVE_INTENT",
        "verified": True,
        "recalled_policy_number": refs.get("policy_number"),
        "recalled_case_id": refs.get("case_id"),
        **extra,
    }


def _select_claim(state: AgentState, claim: dict[str, Any], intent: str) -> AgentState:
    updated: AgentState = {**state}
    updated["selected_case_id"] = claim.get("case_id")
    updated["selected_case_type"] = claim.get("case_type")
    updated["candidate_case_ids"] = [claim.get("case_id")] if claim.get("case_id") else []
    updated["intent"] = intent
    updated["phase"] = "PROCESS_CASE"
    return updated


def _ask_for_claim_id(
    state: AgentState,
    candidates: list[dict[str, Any]],
    intent: str,
) -> tuple[AgentState, dict[str, Any]]:
    updated: AgentState = {**state}
    ids = [str(claim.get("case_id")) for claim in candidates if claim.get("case_id")]
    updated["candidate_case_ids"] = ids
    updated["selected_case_id"] = None
    updated["intent"] = intent
    updated["phase"] = "RESOLVE_INTENT"
    return updated, _facts(
        updated,
        {"status": "need_claim_id", "candidates": [_brief(claim) for claim in candidates]},
    )


def run_resolve_intent(state: AgentState, user_text: str = "") -> tuple[AgentState, dict[str, Any]]:
    log_current_node("resolve_intent", state)
    with logfire.span("resolve_intent.run", **safe_state_attrs(state)) as span:
        updated, facts = _run_resolve_intent(state, user_text)
        span.set_attributes(safe_state_attrs(updated))
        span.set_attribute("intent", updated.get("intent"))
        span.set_attribute("selected_case_id", updated.get("selected_case_id"))
        return updated, facts


def _run_resolve_intent(state: AgentState, user_text: str = "") -> tuple[AgentState, dict[str, Any]]:
    if not state.get("verified") or not state.get("party_id"):
        return state, _facts(state, {"status": "not_verified", "verified": False})

    if state.get("phase") == "PROCESS_CASE" and state.get("selected_case_id"):
        return state, {}

    refs = recalled_refs(state)
    intent = infer_intent(state)
    updated: AgentState = {**state, "phase": "RESOLVE_INTENT"}
    named = _caller_named_claim(_hint(updated))
    if intent is None and (refs.get("case_id") or named):
        intent = "claim_issue"
    if intent:
        updated["intent"] = intent

    if intent is None:
        updated["intent"] = None
        return updated, _facts(updated, {"status": "ask_intent"})

    if intent == "policy_inquiry":
        updated["selected_case_id"] = None
        updated["selected_case_type"] = None
        updated["candidate_case_ids"] = []
        updated["phase"] = "PROCESS_CASE"
        return updated, {}

    hint = _hint(updated)
    party_id = str(updated.get("party_id"))
    claims = get_claims_for_party(party_id)
    if not claims:
        return updated, _facts(updated, {"status": "no_claims", "searched_memory": True})

    caller_id = str(hint.get("case_id") or "").strip().upper()
    stored = get_claim(caller_id) if caller_id else None
    if stored and stored.get("party_id") == party_id:
        # Unique id already known — brief the file this turn.
        return _select_claim(updated, stored, intent), {}
    if caller_id:
        return updated, _facts(
            updated,
            {
                "status": "stored_claim_not_found",
                "tried_case_id": caller_id,
                "known_ids": [c.get("case_id") for c in claims],
            },
        )

    listed = [
        claim
        for claim in claims
        if str(claim.get("case_id") or "").upper()
        in {str(item).upper() for item in (updated.get("candidate_case_ids") or []) if item}
    ]
    if len(listed) == 1 and _confirms_listed_claim(user_text) and not named:
        return _select_claim(updated, listed[0], intent), {}

    matched = filter_claims(claims, hint) if named else []
    if named and len(matched) == 1:
        # Hints such as "denied healthcare from January" uniquely match — open the file.
        return _select_claim(updated, matched[0], intent), {}

    if named and len(matched) > 1:
        types = {str(claim.get("case_type") or "") for claim in matched}
        if len(types) == 1 or hint.get("case_type"):
            return _ask_for_claim_id(updated, matched, intent)
        updated["candidate_case_ids"] = [
            str(claim.get("case_id")) for claim in matched if claim.get("case_id")
        ]
        return updated, _facts(
            updated,
            {"status": "need_type_or_id", "case_types": sorted(t for t in types if t)},
        )

    if named:
        return updated, _facts(updated, {"status": "no_match"})

    return _ask_for_claim_id(updated, claims, intent)
