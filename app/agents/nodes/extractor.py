from __future__ import annotations

import re
from typing import Literal

# Load .env and configure Logfire before LangChain so OTEL tracing is enabled.
from app.config import GROQ_FALLBACK_MODEL, GROQ_MODEL, resolve_groq_api_key
from app.observability import log_current_node, safe_state_attrs

import logfire
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.agents.identity_parse import normalize_dob, normalize_phone, parse_identity_utterance
from app.agents.nodes.state import AgentState, merge_dicts
from app.memory import CASE_HINT_FIELDS, record_affect, record_extraction

IN_SCOPE_REPLY = (
    "I have noted that. I still need to verify your identity before I can discuss "
    "claim details."
)

IDENTITY_FIELDS = ("name", "dob", "phone", "email", "ssn_last4", "policy_number")


class TurnExtraction(BaseModel):
    name: str | None = Field(default=None, description="Full name")
    dob: str | None = Field(default=None, description="Date of birth YYYY-MM-DD")
    phone: str | None = Field(default=None, description="Phone number")
    email: str | None = Field(default=None, description="Email address")
    ssn_last4: str | None = Field(default=None, description="Last 4 SSN or national ID digits")
    policy_number: str | None = Field(default=None, description="Policy number")
    caller_role: Literal["policyholder", "representative"] | None = None
    member_name: str | None = Field(
        default=None,
        description="Policyholder name when the caller is a representative",
    )
    relationship: str | None = Field(
        default=None,
        description="Relationship to the policyholder, such as son or daughter",
    )
    declined_ssn: bool = Field(
        default=False,
        description="True if the caller does not want to share SSN or full SSN",
    )
    case_id: str | None = Field(default=None, description="Claim id such as CL-2048")
    case_type: str | None = Field(default=None, description="healthcare, dental, or auto")
    status: str | None = Field(default=None, description="denied, open, or closed")
    month: str | None = Field(default=None, description="Month mentioned for the claim")
    year: str | None = Field(default=None, description="Year mentioned for the claim")
    summary_text: str | None = Field(default=None, description="Short claim description from the caller")
    intent_hints: list[str] = Field(default_factory=list)


EXTRACTOR_PROMPT = """Extract insurance-call facts into JSON.

Copy values that appear in the utterance. Examples:
- My name is Margaret Chen -> name=Margaret Chen
- DOB is 1985-03-15 -> dob=1985-03-15
- SSN last four is 4472 -> ssn_last4=4472
- policy POL-9921 -> policy_number=POL-9921
- I am the policyholder -> caller_role=policyholder
- I am David Chen, her son, calling for Margaret Chen -> caller_role=representative, name=David Chen, relationship=son, member_name=Margaret Chen
- I do not want to give my SSN -> declined_ssn=true (still in scope)
- denied healthcare claim from January -> status=denied, case_type=healthcare, month=January

Copy only values that appear in the utterance. Do not invent a claim id, claim type, status, month, or year. Identity alone (name, date of birth, phone, email, SSN) is not a claim identifier.
Use null when a field is absent. intent_hints may include status_inquiry, denial_question, document_submission, next_steps, policy_question, claim_issue.
"""


def _compact(data: dict) -> dict:
    return {key: value for key, value in data.items() if value not in (None, "", [], {})}


def _mentioned_in_utterance(user_text: str, value: str) -> bool:
    text = (user_text or "").lower()
    needle = str(value).strip().lower()
    if not needle:
        return False
    if needle in text:
        return True
    compact_text = re.sub(r"[^a-z0-9]+", "", text)
    compact_needle = re.sub(r"[^a-z0-9]+", "", needle)
    return bool(compact_needle) and compact_needle in compact_text


def _ground_intent_hints(user_text: str, hints: list[str]) -> list[str]:
    text = (user_text or "").lower()
    allowed: list[str] = []
    for hint_name in hints:
        if hint_name == "claim_issue" and any(token in text for token in ("claim", "cl-", "case id", "caseid")):
            allowed.append(hint_name)
        elif hint_name == "denial_question" and any(token in text for token in ("denied", "denial")):
            allowed.append(hint_name)
        elif hint_name == "status_inquiry" and "status" in text:
            allowed.append(hint_name)
        elif hint_name == "document_submission" and any(token in text for token in ("document", "upload", "submit")):
            allowed.append(hint_name)
        elif hint_name == "next_steps" and any(token in text for token in ("next step", "what now", "appeal")):
            allowed.append(hint_name)
        elif hint_name == "policy_question" and "policy" in text:
            allowed.append(hint_name)
    return list(dict.fromkeys(allowed))


def _model_unavailable(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "does not exist" in message
        or "model_not_found" in message
        or "not have access" in message
        or "not found" in message
    )


def _chat_model(api_key: str, model_id: str) -> ChatGroq:
    kwargs: dict = {"api_key": api_key, "model": model_id, "temperature": 0}
    if model_id.startswith("openai/gpt-oss"):
        kwargs["reasoning_format"] = "hidden"
    return ChatGroq(**kwargs)


def extract_turn(user_text: str, state: AgentState, api_key: str | None = None) -> TurnExtraction:
    with logfire.span("extractor.extract_turn") as span:
        key = resolve_groq_api_key(api_key)
        if not key:
            raise ValueError("Enter a Groq API key in the sidebar to test this application.")

        payload = [
            SystemMessage(content=EXTRACTOR_PROMPT),
            HumanMessage(content=user_text),
        ]

        last_error: Exception | None = None
        for model_id in (GROQ_MODEL, GROQ_FALLBACK_MODEL):
            for method in ("json_mode", "function_calling"):
                try:
                    with logfire.span(
                        "extractor.llm_attempt",
                        llm_model=model_id,
                        structured_output_method=method,
                    ):
                        llm = _chat_model(key, model_id).with_structured_output(
                            TurnExtraction,
                            method=method,
                        )
                        result = llm.invoke(payload)
                    span.set_attribute("llm.model", model_id)
                    span.set_attribute("structured_output.method", method)
                    if isinstance(result, TurnExtraction):
                        return result
                    return TurnExtraction.model_validate(result)
                except Exception as exc:
                    last_error = exc
                    if _model_unavailable(exc):
                        break
                    continue
        raise RuntimeError(
            f"GPT-OSS 120B ({GROQ_MODEL}) and OSS 20B fallback ({GROQ_FALLBACK_MODEL}) are unavailable."
        ) from last_error


def overlay_utterance_rules(user_text: str, extracted: TurnExtraction) -> TurnExtraction:
    text = user_text.lower()
    data = extracted.model_copy()
    parsed = parse_identity_utterance(user_text)
    hints: list[str] = []
    for field, value in parsed.items():
        setattr(data, field, value)
    for field in CASE_HINT_FIELDS:
        current = getattr(data, field, None)
        if not current:
            continue
        if parsed.get(field) or _mentioned_in_utterance(user_text, current):
            continue
        setattr(data, field, None)
    if parsed.get("case_id") or parsed.get("case_type") or parsed.get("status"):
        hints.append("claim_issue")
    if parsed.get("status") == "denied":
        hints.append("denial_question")
    if "status" in text and "claim" in text:
        hints.append("status_inquiry")
    data.intent_hints = _ground_intent_hints(user_text, list(dict.fromkeys([*(data.intent_hints or []), *hints])))
    has_ssn_word = any(token in text for token in ("ssn", "social security", "social"))
    decline = any(
        token in text
        for token in (
            "don't want",
            "do not want",
            "rather not",
            "prefer not",
            "won't give",
            "will not give",
            "not giving",
            "not share",
            "not sharing",
            "too sensitive",
            "uncomfortable",
            "skip",
            "don't have",
            "do not have",
            "full ssn",
            "full social",
        )
    )
    if has_ssn_word and decline:
        data.declined_ssn = True
    if any(
        token in text
        for token in (
            "representative",
            "on behalf",
            "calling on behalf",
            "calling for my mother",
            "calling for my mom",
            "calling for my father",
            "her son",
            "his son",
            "her daughter",
            "his daughter",
            "authorized",
        )
    ):
        if data.caller_role is None:
            data.caller_role = "representative"
    if "policyholder" in text and "not the policyholder" not in text and data.caller_role is None:
        data.caller_role = "policyholder"
    return data


def apply_extraction(state: AgentState, extracted: TurnExtraction) -> AgentState:
    updated: AgentState = {**state}
    payload = extracted.model_dump()
    existing_role = state.get("caller_role")
    role = payload.get("caller_role") or existing_role

    identity = _compact({field: payload.get(field) for field in IDENTITY_FIELDS})
    if identity.get("dob"):
        identity["dob"] = normalize_dob(str(identity["dob"])) or identity["dob"]
    if identity.get("phone"):
        identity["phone"] = normalize_phone(str(identity["phone"])) or identity["phone"]
    if role == "representative":
        if payload.get("member_name"):
            identity["name"] = payload["member_name"]
            identity["member_name"] = payload["member_name"]
        elif payload.get("name"):
            identity.pop("name", None)
    updated["identity"] = merge_dicts(state.get("identity"), identity)
    if role:
        updated["caller_role"] = role

    declined: bool | None = None
    if payload.get("declined_ssn"):
        declined = True
    if payload.get("ssn_last4"):
        declined = False
    return record_extraction(
        updated,
        representative_name=payload.get("name") if role == "representative" else None,
        relationship=payload.get("relationship"),
        declined_ssn=declined,
        policy_number=updated["identity"].get("policy_number"),
        case_hint=_compact({field: payload.get(field) for field in CASE_HINT_FIELDS}) or None,
        intent_hints_in=extracted.intent_hints or None,
    )


def extract_and_apply(
    state: AgentState,
    user_text: str,
    api_key: str | None = None,
) -> tuple[AgentState, TurnExtraction | None]:
    with logfire.span("extractor.extract_and_apply", **safe_state_attrs(state)) as span:
        log_current_node("extractor", state)
        if state.get("phase") in {"HUMAN_ESCALATION", "DONE"}:
            span.set_attribute("skipped", True)
            return state, None
        extracted = TurnExtraction()
        try:
            extracted = extract_turn(user_text, state, api_key=api_key)
        except Exception as exc:
            span.set_attribute("llm_extract_failed", True)
            logfire.warning("extractor.llm_failed", error_type=type(exc).__name__)
        extracted = overlay_utterance_rules(user_text, extracted)
        payload = extracted.model_dump()
        present_fields = [
            field
            for field in (*IDENTITY_FIELDS, *CASE_HINT_FIELDS, "caller_role", "member_name")
            if payload.get(field)
        ]
        span.set_attribute("declined_ssn", extracted.declined_ssn)
        span.set_attribute("caller_role", extracted.caller_role)
        span.set_attribute("extracted_fields", present_fields)
        updated = apply_extraction(state, extracted)
        updated = record_affect(updated, user_text)
        span.set_attributes(safe_state_attrs(updated))
        return updated, extracted


def run_extractor_node(state: AgentState, user_text: str, api_key: str | None = None) -> tuple[AgentState, str]:
    if state.get("phase") == "DONE":
        return state, "This conversation is already complete. Reset the chat if you need to start again."
    if state.get("phase") == "HUMAN_ESCALATION":
        from app.guardrails.rails import ALREADY_ESCALATED_REPLY

        return state, ALREADY_ESCALATED_REPLY

    updated, extracted = extract_and_apply(state, user_text, api_key=api_key)
    if extracted is None:
        from app.guardrails.rails import ALREADY_ESCALATED_REPLY

        if updated.get("phase") == "DONE":
            return updated, "This conversation is already complete. Reset the chat if you need to start again."
        return updated, ALREADY_ESCALATED_REPLY
    if updated.get("phase") == "HUMAN_ESCALATION":
        from app.guardrails.rails import ESCALATION_REPLY

        return updated, ESCALATION_REPLY
    return updated, IN_SCOPE_REPLY
