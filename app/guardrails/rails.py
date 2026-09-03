from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import logfire
from nemoguardrails import LLMRails, RailsConfig

from app.agents.nodes.state import AgentState
from app.guardrails import actions as guardrail_actions
from app.observability import log_current_node

GUARDRAILS_DIR = Path(__file__).resolve().parent
RailCategory = Literal["off_topic", "jailbreak", "prompt_injection"]
RAIL_LIMIT = 3

OFF_TOPIC_REPLY = (
    "I can only help with this insurance claims call. Please share identity details "
    "or a question about your claim. If you keep asking unrelated questions, I will "
    "connect you with a human representative."
)
JAILBREAK_REPLY = (
    "I cannot change my role or ignore my instructions. I can only help with this "
    "insurance claims call."
)
PROMPT_INJECTION_REPLY = (
    "I cannot follow injected instructions. I can only help with this insurance claims call."
)
ESCALATION_REPLY = (
    "I am connecting you with a human representative, who can take it from here."
)
ALREADY_ESCALATED_REPLY = "A human representative will take it from here."

REPLIES: dict[str, str] = {
    "off_topic": OFF_TOPIC_REPLY,
    "jailbreak": JAILBREAK_REPLY,
    "prompt_injection": PROMPT_INJECTION_REPLY,
}


@dataclass(frozen=True)
class RailDecision:
    blocked: bool
    category: RailCategory | None = None
    source: str | None = None
    message: str = ""


@lru_cache(maxsize=1)
def _rails() -> LLMRails:
    logfire.info("Loading NeMo Guardrails config", path=str(GUARDRAILS_DIR))
    config = RailsConfig.from_path(str(GUARDRAILS_DIR))
    rails = LLMRails(config, verbose=False)
    rails.register_action(guardrail_actions.check_prompt_guard, "check_prompt_guard")
    rails.register_action(guardrail_actions.check_jailbreak, "check_jailbreak")
    rails.register_action(guardrail_actions.check_prompt_injection, "check_prompt_injection")
    rails.register_action(guardrail_actions.check_off_topic, "check_off_topic")
    return rails


def screen_user_turn(user_text: str, api_key: str | None = None) -> RailDecision:
    with logfire.span("guardrails.screen") as span:
        log_current_node("guardrails")
        guardrail_actions.reset_last_hit()
        guardrail_actions.set_groq_api_key(api_key)
        try:
            _rails().generate(
                messages=[{"role": "user", "content": user_text}],
                options={"rails": ["input"]},
            )
        except Exception as exc:
            logfire.error("NeMo Guardrails generate failed", error_type=type(exc).__name__)
            return _fallback_screen(user_text)

        hit = guardrail_actions.last_hit
        category = hit.get("category")
        blocked = category in REPLIES
        span.set_attribute("blocked", blocked)
        span.set_attribute("category", category)
        span.set_attribute("source", hit.get("source"))
        if blocked:
            logfire.warning(
                "Guardrail blocked user turn",
                category=category,
                source=hit.get("source"),
            )
            return RailDecision(
                blocked=True,
                category=category,
                source=hit.get("source"),
                message=REPLIES[category],
            )
        logfire.info("Guardrail allowed user turn")
        return RailDecision(blocked=False)


def _fallback_screen(user_text: str) -> RailDecision:
    """If LLMRails cannot start, still enforce Colang examples and Prompt Guard 2."""
    import asyncio

    async def _run() -> RailDecision:
        context = {"user_message": user_text, "last_user_message": user_text}
        if await guardrail_actions.check_prompt_guard(context):
            category = guardrail_actions.last_hit["category"]
            return RailDecision(True, category, "prompt_guard_2", REPLIES[category])
        if await guardrail_actions.check_jailbreak(context):
            return RailDecision(True, "jailbreak", "colang", JAILBREAK_REPLY)
        if await guardrail_actions.check_prompt_injection(context):
            return RailDecision(True, "prompt_injection", "colang", PROMPT_INJECTION_REPLY)
        if await guardrail_actions.check_off_topic(context):
            return RailDecision(True, "off_topic", "colang", OFF_TOPIC_REPLY)
        return RailDecision(blocked=False)

    return asyncio.run(_run())


def apply_rail_to_state(state: AgentState, decision: RailDecision) -> tuple[AgentState, str]:
    updated: AgentState = {**state}
    count = int(state.get("out_of_scope_count") or 0) + 1
    updated["out_of_scope_count"] = count
    logfire.info(
        "Guardrail strike recorded",
        category=decision.category,
        out_of_scope_count=count,
        limit=RAIL_LIMIT,
    )
    if count >= RAIL_LIMIT:
        updated["phase"] = "HUMAN_ESCALATION"
        logfire.warning("Guardrail limit reached; escalating to human")
        return updated, ESCALATION_REPLY
    return updated, decision.message
