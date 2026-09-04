from typing import Annotated, Literal, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.memory import AgentMemory, CaseHint, merge_maps, merge_unique

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

merge_dicts = merge_maps


class IdentityFields(TypedDict, total=False):
    name: str
    dob: str
    phone: str
    email: str
    ssn_last4: str
    policy_number: str
    member_name: str


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    phase: Phase
    identity: Annotated[IdentityFields, merge_maps]
    verified: bool
    matched_pii: Annotated[list[str], merge_unique]
    party_id: Optional[str]
    caller_role: Optional[CallerRole]
    memory: Annotated[AgentMemory, merge_maps]
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
