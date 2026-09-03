from typing import Annotated, Any, Literal, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

Phase = Literal[
    "VERIFY_ID",
    "RESOLVE_INTENT",
    "PROCESS_CASE",
    "POST_PROCESS",
    "HUMAN_ESCALATION",
    "DONE",
]

CallerRole = Literal["policyholder", "representative"]
EmailChoice = Literal["send", "skip"]


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    return {**(left or {}), **(right or {})}


def merge_unique(left: list[str] | None, right: list[str] | None) -> list[str]:
    merged: list[str] = []
    for item in [*(left or []), *(right or [])]:
        if item not in merged:
            merged.append(item)
    return merged


class IdentityFields(TypedDict, total=False):
    name: str
    dob: str
    phone: str
    email: str
    ssn_last4: str
    policy_number: str
    member_name: str


class CaseHint(TypedDict, total=False):
    case_id: str
    case_type: str
    status: str
    month: str
    year: str
    summary_text: str


class AgentMemory(TypedDict, total=False):
    intent_hints: list[str]
    case_hint: CaseHint
    declined_ssn: bool
    representative_name: str
    relationship: str
    case_briefed: bool
    awaiting_human_confirm: bool
    process_topics: list[str]
    case_answers: list[str]
    email_offered: bool


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    phase: Phase
    identity: Annotated[IdentityFields, merge_dicts]
    verified: bool
    matched_pii: Annotated[list[str], merge_unique]
    party_id: Optional[str]
    caller_role: Optional[CallerRole]
    memory: Annotated[AgentMemory, merge_dicts]
    selected_case_id: Optional[str]
    selected_case_type: Optional[str]
    candidate_case_ids: list[str]
    intent: Optional[str]
    out_of_scope_count: int
    email_choice: Optional[EmailChoice]


INITIAL_STATE: AgentState = {
    "messages": [],
    "phase": "VERIFY_ID",
    "identity": {},
    "verified": False,
    "matched_pii": [],
    "party_id": None,
    "caller_role": None,
    "memory": {},
    "selected_case_id": None,
    "selected_case_type": None,
    "candidate_case_ids": [],
    "intent": None,
    "out_of_scope_count": 0,
    "email_choice": None,
}
