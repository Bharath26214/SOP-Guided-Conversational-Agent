from typing import TypedDict

CASE_HINT_FIELDS = ("case_id", "case_type", "status", "month", "year", "summary_text")
CASE_ANSWER_LIMIT = 8


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
    policy_number: str
    case_id: str
    declined_ssn: bool
    representative_name: str
    relationship: str
    case_briefed: bool
    awaiting_human_confirm: bool
    process_topics: list[str]
    case_answers: list[str]
    last_question: str
    last_agent_reply: str
    email_offered: bool
    awaiting_more_help: bool
    affect: str
    affect_tone: str
    persuasion_attempts: int
