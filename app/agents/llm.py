from __future__ import annotations

import json
from typing import Any, Literal

import logfire
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.config import GROQ_FALLBACK_MODEL, GROQ_MODEL, resolve_groq_api_key

from app.agents.prompts import CLASSIFY_SYSTEM, SCOPE_SYSTEM

Move = Literal[
    "claim_question",
    "wrap_up",
    "email_send",
    "email_skip",
    "human_yes",
    "human_no",
    "other",
]
Scope = Literal["on_sop", "off_topic"]


class CallerMove(BaseModel):
    move: Move = Field(description="What the caller is doing on this turn")


class ScopeVerdict(BaseModel):
    scope: Scope = Field(description="Whether the utterance belongs on this claims SOP call")


def _model_unavailable(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "does not exist" in message
        or "model_not_found" in message
        or "not have access" in message
        or "not found" in message
    )


def chat_model(api_key: str | None = None, *, temperature: float = 0.4) -> ChatGroq:
    key = resolve_groq_api_key(api_key)
    if not key:
        raise ValueError("Enter a Groq API key in the sidebar to test this application.")
    kwargs: dict[str, Any] = {"api_key": key, "model": GROQ_MODEL, "temperature": temperature}
    if GROQ_MODEL.startswith("openai/gpt-oss"):
        kwargs["reasoning_format"] = "hidden"
    return ChatGroq(**kwargs)


def _fallback_model(api_key: str | None = None, *, temperature: float = 0.2) -> ChatGroq:
    key = resolve_groq_api_key(api_key)
    kwargs: dict[str, Any] = {
        "api_key": key,
        "model": GROQ_FALLBACK_MODEL,
        "temperature": temperature,
    }
    if GROQ_FALLBACK_MODEL.startswith("openai/gpt-oss"):
        kwargs["reasoning_format"] = "hidden"
    return ChatGroq(**kwargs)


def phrase(system: str, facts: dict[str, Any], api_key: str | None) -> str:
    """Turn grounded facts into a short spoken reply from the SOP system prompt."""
    payload = [
        SystemMessage(content=system),
        HumanMessage(content=json.dumps(facts, ensure_ascii=True, default=str)),
    ]
    last_error: Exception | None = None
    for builder in (chat_model, _fallback_model):
        try:
            with logfire.span("llm.phrase"):
                result = builder(api_key).invoke(payload)
            text = str(getattr(result, "content", result) or "").strip()
            if text:
                return text
        except Exception as exc:
            last_error = exc
            if not _model_unavailable(exc):
                logfire.error("llm.phrase failed", error_type=type(exc).__name__)
                break
    if last_error:
        logfire.error("llm.phrase unavailable", error_type=type(last_error).__name__)
    return ""


def classify_scope(
    user_text: str,
    api_key: str | None,
    *,
    phase: str = "",
    last_agent_reply: str = "",
) -> Scope:
    facts = {
        "utterance": user_text,
        "phase": phase or "VERIFY_ID",
        "last_agent_reply": last_agent_reply or "",
    }
    payload = [
        SystemMessage(content=SCOPE_SYSTEM),
        HumanMessage(content=json.dumps(facts, ensure_ascii=True)),
    ]
    key = resolve_groq_api_key(api_key)
    if key:
        for builder in (chat_model, _fallback_model):
            try:
                with logfire.span("llm.scope"):
                    model = builder(api_key, temperature=0).with_structured_output(
                        ScopeVerdict, method="json_mode"
                    )
                    result = model.invoke(payload)
                if isinstance(result, ScopeVerdict):
                    return result.scope
            except Exception as exc:
                logfire.error("llm.scope failed", error_type=type(exc).__name__)
                if _model_unavailable(exc):
                    continue
                break
    return "on_sop"


def classify_turn(
    user_text: str,
    api_key: str | None,
    *,
    offered_email: bool = False,
    awaiting_human: bool = False,
    on_claim: bool = False,
    awaiting_more_help: bool = False,
) -> Move:
    facts = {
        "utterance": user_text,
        "offered_email": offered_email,
        "awaiting_human": awaiting_human,
        "on_claim": on_claim,
        "awaiting_more_help": awaiting_more_help,
    }
    payload = [
        SystemMessage(content=CLASSIFY_SYSTEM),
        HumanMessage(content=json.dumps(facts, ensure_ascii=True)),
    ]
    key = resolve_groq_api_key(api_key)
    if key:
        for builder in (chat_model, _fallback_model):
            try:
                with logfire.span("llm.classify"):
                    model = builder(api_key, temperature=0).with_structured_output(
                        CallerMove, method="json_mode"
                    )
                    result = model.invoke(payload)
                if isinstance(result, CallerMove):
                    return result.move
            except Exception as exc:
                logfire.error("llm.classify failed", error_type=type(exc).__name__)
                if _model_unavailable(exc):
                    continue
                break
    return _heuristic_move(
        user_text,
        offered_email=offered_email,
        awaiting_human=awaiting_human,
        on_claim=on_claim,
        awaiting_more_help=awaiting_more_help,
    )


def _heuristic_move(
    user_text: str,
    *,
    offered_email: bool,
    awaiting_human: bool,
    on_claim: bool = False,
    awaiting_more_help: bool = False,
) -> Move:
    text = (user_text or "").strip().lower()
    if not text:
        return "other"
    if awaiting_human:
        if text in {"yes", "yeah", "yep", "ok", "okay", "please"}:
            return "human_yes"
        if text in {"no", "nope", "not now", "later"}:
            return "human_no"
    if offered_email:
        if text in {"yes", "yeah", "yep", "ok", "okay", "please", "send"}:
            return "email_send"
        if text in {"no", "nope", "skip", "no thanks", "no thank you"}:
            return "email_skip"
    if awaiting_more_help:
        if text in {
            "no",
            "nope",
            "no thanks",
            "no thank you",
            "nothing else",
            "that's all",
            "that is all",
            "that's it",
            "that is it",
            "i'm good",
            "im good",
            "i'm done",
            "im done",
            "i'm fine",
            "all good",
            "no i don't",
            "no i do not",
        }:
            return "wrap_up"
        if text in {"yes", "yeah", "yep", "ok", "okay", "sure", "please"}:
            return "claim_question"
    if any(token in text for token in ("send the email", "email me a summary", "send a summary")):
        return "email_send"
    if any(token in text for token in ("skip the email", "don't send", "no email")):
        return "email_skip"
    if any(token in text for token in ("that's all", "that is all", "nothing else", "goodbye", "that's it")):
        return "wrap_up"
    if text in {"thanks", "thank you", "thx"}:
        return "wrap_up"
    if any(
        token in text
        for token in (
            "claim",
            "denied",
            "denial",
            "document",
            "deadline",
            "appeal",
            "status",
            "amount",
            "reimbursement",
            "reimburse",
            "allowed",
            "net pay",
            "net fee",
            "pathology",
            "office note",
            "why",
            "how much",
            "paid",
            "missing",
            "upload",
            "portal",
        )
    ):
        return "claim_question"
    if on_claim and text not in {"yes", "yeah", "yep", "ok", "okay", "no", "nope"}:
        return "claim_question"
    if any(token in text for token in ("human", "representative", "escalate")):
        return "human_yes"
    return "other"
