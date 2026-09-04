from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import logfire
from nemoguardrails.actions import action

from app.config import (
    PROMPT_GUARD_FALLBACK_MODEL,
    PROMPT_GUARD_MODEL,
    resolve_groq_api_key,
)

_active_groq_api_key = ""
_scope_phase = "VERIFY_ID"
_last_agent_reply = ""

GUARDRAILS_DIR = Path(__file__).resolve().parent
STOPWORDS = {
    "a",
    "an",
    "the",
    "your",
    "my",
    "to",
    "and",
    "or",
    "of",
    "is",
    "it",
    "in",
    "for",
    "on",
    "about",
    "me",
    "you",
    "are",
    "be",
    "with",
    "this",
    "that",
}

last_hit: dict[str, Any] = {"category": None, "source": None, "matched_example": None}


def set_groq_api_key(api_key: str | None) -> None:
    global _active_groq_api_key
    _active_groq_api_key = resolve_groq_api_key(api_key)


def set_scope_context(*, phase: str | None = None, last_agent_reply: str | None = None) -> None:
    global _scope_phase, _last_agent_reply
    if phase:
        _scope_phase = phase
    if last_agent_reply is not None:
        _last_agent_reply = last_agent_reply


def reset_last_hit() -> None:
    last_hit.update({"category": None, "source": None, "matched_example": None})


def _mark(category: str, source: str, matched_example: str | None = None) -> None:
    last_hit.update(
        {
            "category": category,
            "source": source,
            "matched_example": matched_example,
        }
    )


def _user_message(context: dict | None) -> str:
    context = context or {}
    return str(context.get("user_message") or context.get("last_user_message") or "")


def _parse_user_examples(colang_path: Path) -> list[str]:
    examples: list[str] = []
    collecting = False
    for raw in colang_path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("define user "):
            collecting = True
            continue
        if line.startswith("define "):
            collecting = False
            continue
        if collecting:
            match = re.search(r'"([^"]+)"', line)
            if match:
                examples.append(match.group(1))
    return examples


@lru_cache(maxsize=1)
def _off_topic_examples() -> tuple[str, ...]:
    return tuple(_parse_user_examples(GUARDRAILS_DIR / "off_topic.co"))


@lru_cache(maxsize=1)
def _jailbreak_examples() -> tuple[str, ...]:
    return tuple(_parse_user_examples(GUARDRAILS_DIR / "jailbreak.co"))


@lru_cache(maxsize=1)
def _injection_examples() -> tuple[str, ...]:
    return tuple(_parse_user_examples(GUARDRAILS_DIR / "prompt_injection.co"))


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9']+", text.lower()) if token not in STOPWORDS]


def matches_example(utterance: str, example: str) -> bool:
    haystack = utterance.lower()
    needle = example.lower().strip()
    if not needle:
        return False
    if needle in haystack:
        return True
    words = _tokens(needle)
    if len(words) < 2:
        return False
    return all(word in haystack for word in words)


def first_matching_example(utterance: str, examples: tuple[str, ...]) -> str | None:
    for example in examples:
        if matches_example(utterance, example):
            return example
    return None


def _looks_like_claims_turn(utterance: str) -> bool:
    text = utterance.lower()
    hints = (
        "claim",
        "policy",
        "policyholder",
        "insurance",
        "ssn",
        "date of birth",
        "dob",
        "case id",
        "appeal",
        "denial",
        "denied",
        "reimbursement",
        "representative",
        "email",
        "summary",
        "verify",
        "identity",
        "last four",
        "phone number",
        "my name",
    )
    return any(hint in text for hint in hints)


def _looks_like_sop_turn(utterance: str) -> bool:
    text = (utterance or "").strip().lower()
    if not text:
        return True
    if text in {
        "hi",
        "hello",
        "hey",
        "yes",
        "yeah",
        "yep",
        "ok",
        "okay",
        "sure",
        "continue",
        "please",
        "thanks",
        "thank you",
        "send",
        "skip",
        "no",
        "nope",
    }:
        return True
    if _looks_like_claims_turn(utterance):
        return True
    from app.agents.identity_parse import parse_identity_utterance

    parsed = parse_identity_utterance(utterance)
    if parsed:
        return True
    return False


def _prompt_guard_models() -> list[str]:
    models = [PROMPT_GUARD_MODEL]
    if PROMPT_GUARD_FALLBACK_MODEL and PROMPT_GUARD_FALLBACK_MODEL not in models:
        models.append(PROMPT_GUARD_FALLBACK_MODEL)
    return models


def _model_unavailable(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "does not exist" in message
        or "model_not_found" in message
        or "not have access" in message
        or "not found" in message
    )


def classify_prompt_guard(user_text: str) -> dict[str, Any]:
    if not user_text.strip():
        return {"available": False, "malicious": False, "label": None, "score": None}

    key = resolve_groq_api_key(_active_groq_api_key)
    if not key:
        logfire.warning("Prompt Guard 2 skipped; GROQ_API_KEY is missing")
        return {"available": False, "malicious": False, "label": None, "score": None}

    try:
        import httpx
    except Exception as exc:
        logfire.warning("Prompt Guard 2 unavailable", error_type=type(exc).__name__)
        return {"available": False, "malicious": False, "label": None, "score": None}

    last_error: Exception | None = None
    with httpx.Client(timeout=20.0) as client:
        for model_id in _prompt_guard_models():
            try:
                response = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": user_text[:512]}],
                        "temperature": 0,
                        "max_tokens": 16,
                    },
                )
                if response.status_code >= 400:
                    detail = response.text
                    last_error = RuntimeError(detail)
                    if _model_unavailable(last_error):
                        logfire.warning(
                            "Prompt Guard 2 model unavailable",
                            model_id=model_id,
                            error_type="HTTPStatusError",
                        )
                        continue
                    response.raise_for_status()
                payload = response.json()
                label = str(
                    (((payload.get("choices") or [{}])[0].get("message") or {}).get("content"))
                    or ""
                ).strip().upper()
                malicious = label == "MALICIOUS"
                logfire.info(
                    "Prompt Guard 2 scored input",
                    label=label or None,
                    malicious=malicious,
                    model_id=model_id,
                )
                return {
                    "available": True,
                    "malicious": malicious,
                    "label": label or None,
                    "score": 1.0 if malicious else 0.0,
                    "model_id": model_id,
                }
            except Exception as exc:
                last_error = exc
                if _model_unavailable(exc):
                    logfire.warning(
                        "Prompt Guard 2 model unavailable",
                        model_id=model_id,
                        error_type=type(exc).__name__,
                    )
                    continue
                logfire.warning(
                    "Prompt Guard 2 request failed",
                    model_id=model_id,
                    error_type=type(exc).__name__,
                )
                break

    if last_error is not None:
        logfire.warning(
            "Prompt Guard 2 failed; Colang rules still apply",
            error_type=type(last_error).__name__,
        )
    return {"available": False, "malicious": False, "label": None, "score": None}


@action(name="check_prompt_guard", is_system_action=True)
async def check_prompt_guard(context: dict | None = None) -> bool:
    user_text = _user_message(context)
    verdict = classify_prompt_guard(user_text)
    if verdict["malicious"]:
        example = first_matching_example(user_text, _jailbreak_examples())
        category = "jailbreak" if example else "prompt_injection"
        _mark(category, "prompt_guard_2", example)
        logfire.warning("Prompt Guard 2 blocked input", category=category)
        return True
    return False


@action(name="check_jailbreak", is_system_action=True)
async def check_jailbreak(context: dict | None = None) -> bool:
    user_text = _user_message(context)
    example = first_matching_example(user_text, _jailbreak_examples())
    if example:
        _mark("jailbreak", "colang", example)
        logfire.warning("Colang jailbreak rail matched")
        return True
    return False


@action(name="check_prompt_injection", is_system_action=True)
async def check_prompt_injection(context: dict | None = None) -> bool:
    user_text = _user_message(context)
    example = first_matching_example(user_text, _injection_examples())
    if example:
        _mark("prompt_injection", "colang", example)
        logfire.warning("Colang prompt-injection rail matched")
        return True
    return False


@action(name="check_off_topic", is_system_action=True)
async def check_off_topic(context: dict | None = None) -> bool:
    user_text = _user_message(context)
    if _looks_like_sop_turn(user_text):
        return False
    example = first_matching_example(user_text, _off_topic_examples())
    if example:
        _mark("off_topic", "colang", example)
        logfire.info("Colang off-topic rail matched")
        return True
    from app.agents.llm import classify_scope

    scope = classify_scope(
        user_text,
        _active_groq_api_key,
        phase=_scope_phase,
        last_agent_reply=_last_agent_reply,
    )
    if scope == "off_topic":
        _mark("off_topic", "scope")
        logfire.info("Scope classifier marked turn off-topic")
        return True
    return False
