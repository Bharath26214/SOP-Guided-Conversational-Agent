from __future__ import annotations

from datetime import datetime
from typing import Any

import logfire

from app.agents.nodes.state import AgentState
from app.data_loaders.queries import get_claim, get_claims_for_party
from app.observability import log_current_node, safe_state_attrs

ASK_DIFFICULTY = (
    "What difficulty are you facing today — a problem with a specific claim, "
    "a policy question, or a claim status update?"
)
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
    return dict((state.get("memory") or {}).get("case_hint") or {})


def _intent_hints(state: AgentState) -> list[str]:
    return [str(item).lower() for item in ((state.get("memory") or {}).get("intent_hints") or [])]


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


def _brief(claim: dict[str, Any]) -> str:
    return (
        f"{claim.get('case_id')} ({claim.get('case_type')}, {claim.get('status')}, "
        f"{claim.get('created_at')})"
    )


def _select_claim(state: AgentState, claim: dict[str, Any], intent: str) -> AgentState:
    updated: AgentState = {**state}
    updated["selected_case_id"] = claim.get("case_id")
    updated["selected_case_type"] = claim.get("case_type")
    updated["candidate_case_ids"] = [claim.get("case_id")] if claim.get("case_id") else []
    updated["intent"] = intent
    updated["phase"] = "PROCESS_CASE"
    return updated


def _ask_for_claim_id(state: AgentState, candidates: list[dict[str, Any]], intent: str) -> tuple[AgentState, str]:
    updated: AgentState = {**state}
    ids = [str(claim.get("case_id")) for claim in candidates if claim.get("case_id")]
    updated["candidate_case_ids"] = ids
    updated["selected_case_id"] = None
    updated["intent"] = intent
    updated["phase"] = "RESOLVE_INTENT"
    listed = ", ".join(_brief(claim) for claim in candidates)
    case_type = candidates[0].get("case_type") if candidates else "matching"
    return (
        updated,
        (
            f"I found more than one {case_type} claim on this policy: {listed}. "
            "Please share the claim ID so I can help with the right one."
        ),
    )


def run_resolve_intent(state: AgentState) -> tuple[AgentState, str]:
    log_current_node("resolve_intent", state)
    with logfire.span("resolve_intent.run", **safe_state_attrs(state)) as span:
        updated, message = _run_resolve_intent(state)
        span.set_attributes(safe_state_attrs(updated))
        span.set_attribute("intent", updated.get("intent"))
        span.set_attribute("selected_case_id", updated.get("selected_case_id"))
        return updated, message


def _run_resolve_intent(state: AgentState) -> tuple[AgentState, str]:
    if not state.get("verified") or not state.get("party_id"):
        return state, "I still need to verify your identity before I can look up a claim."

    if state.get("phase") == "PROCESS_CASE" and state.get("selected_case_id"):
        return (
            state,
            f"I already have claim {state.get('selected_case_id')} selected.",
        )

    intent = infer_intent(state)
    updated: AgentState = {**state, "phase": "RESOLVE_INTENT"}
    if intent:
        updated["intent"] = intent

    if intent is None:
        updated["intent"] = None
        return updated, ASK_DIFFICULTY

    if intent == "policy_inquiry":
        updated["selected_case_id"] = None
        updated["selected_case_type"] = None
        updated["candidate_case_ids"] = []
        updated["phase"] = "PROCESS_CASE"
        return updated, "I understand this is a policy question. I will help with that next."

    hint = _hint(updated)
    party_id = str(updated.get("party_id"))
    claims = get_claims_for_party(party_id)
    if not claims:
        return updated, "I do not have an existing claim on file for this policy. Please share a claim ID if you have one."

    if hint.get("case_id"):
        offered = str(hint["case_id"]).strip().upper()
        owned = get_claim(offered)
        if owned is None or owned.get("party_id") != party_id:
            known = ", ".join(claim.get("case_id") or "" for claim in claims)
            return (
                updated,
                (
                    "That claim ID is not on this policy. "
                    f"Please share one of: {known}."
                ),
            )
        selected = _select_claim(updated, owned, intent)
        return (
            selected,
            (
                f"I have claim {owned.get('case_id')} "
                f"({owned.get('case_type')}, {owned.get('status')}). I will help with that next."
            ),
        )

    matched = filter_claims(claims, hint)
    if len(matched) == 1:
        claim = matched[0]
        selected = _select_claim(updated, claim, intent)
        noted = []
        if hint.get("case_type"):
            noted.append(str(hint["case_type"]))
        if hint.get("status"):
            noted.append(str(hint["status"]))
        if hint.get("month"):
            noted.append(str(hint["month"]))
        noted_text = " ".join(noted) if noted else "your request"
        return (
            selected,
            (
                f"I used what you already shared about {noted_text} and matched claim "
                f"{claim.get('case_id')} ({claim.get('case_type')}, {claim.get('status')}, "
                f"{claim.get('created_at')}). I will help with that next."
            ),
        )

    if len(matched) > 1:
        types = {str(claim.get("case_type") or "") for claim in matched}
        if len(types) == 1 or hint.get("case_type"):
            return _ask_for_claim_id(updated, matched, intent)
        type_list = ", ".join(sorted(t for t in types if t))
        updated["candidate_case_ids"] = [
            str(claim.get("case_id")) for claim in matched if claim.get("case_id")
        ]
        return (
            updated,
            (
                "You have more than one kind of claim on file "
                f"({type_list}). Please share the claim type or the claim ID."
            ),
        )

    if hint.get("case_type") or hint.get("status") or hint.get("month"):
        return (
            updated,
            "I could not match those details to a claim on this policy. Please share the claim ID.",
        )

    types = {str(claim.get("case_type") or "") for claim in claims}
    if len(types) == 1:
        return _ask_for_claim_id(updated, claims, intent)
    type_list = ", ".join(sorted(t for t in types if t))
    updated["candidate_case_ids"] = [
        str(claim.get("case_id")) for claim in claims if claim.get("case_id")
    ]
    return (
        updated,
        (
            f"You have {len(claims)} claims on file ({type_list}). "
            "Please share the claim type or the claim ID."
        ),
    )
