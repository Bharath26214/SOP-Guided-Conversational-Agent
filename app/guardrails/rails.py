from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import logfire
from nemoguardrails import LLMRails, RailsConfig

from app.agents.nodes.state import AgentState
from app.guardrails import actions as guardrail_actions
from app.memory import get as memory_get
from app.memory import hint as case_hint
from app.observability import log_current_node

GUARDRAILS_DIR = Path(__file__).resolve().parent
RailCategory = Literal["off_topic", "jailbreak", "prompt_injection"]
RAIL_LIMIT = 3

OFF_TOPIC_REPLY = (
    "I am happy to help with your insurance claim, but I cannot cover that topic. "
    "If you have a question about your identity check or your claim, I can take that. "
    "If you would rather speak with a person, I can connect you with a human representative."
)
JAILBREAK_REPLY = (
    "I need to stay on this insurance claims call and cannot change my role. "
    "I can help with identity verification or your claim, or connect you with a person."
)
PROMPT_INJECTION_REPLY = (
    "I cannot follow those extra instructions. I can help with this insurance claims call, "
    "or connect you with a human representative if you prefer."
)
ESCALATION_REPLY = (
    "Of course. I will connect you with a human representative now. "
    "Thank you for your patience — they will take it from here."
)
ALREADY_ESCALATED_REPLY = (
    "A human representative will take it from here. Thank you for waiting."
)

REPLIES: dict[str, str] = {
    "off_topic": OFF_TOPIC_REPLY,
    "jailbreak": JAILBREAK_REPLY,
    "prompt_injection": PROMPT_INJECTION_REPLY,
}


def off_topic_reply(state: AgentState | None = None) -> str:
    stored = case_hint(state)
    verified = bool((state or {}).get("verified"))
    if verified:
        return (
            "I am happy to help with your insurance claim, but I cannot cover that topic. "
            "If you have another question about this claim, I can take that. "
            "If you would rather speak with a person, I can connect you with a human representative."
        )
    noted = ""
    if stored.get("case_id"):
        noted = f"I still have claim {stored['case_id']} noted for after identity verification. "
    elif stored.get("month"):
        month = str(stored["month"]).title()
        noted = f"I still have your {month} claim noted for after identity verification. "
    elif stored.get("case_type") or stored.get("status"):
        noted = "I still have the claim details you shared noted for after identity verification. "
    return (
        f"{noted}"
        "I cannot help with that other question. "
        "Please share identity details so we can continue with your claim, "
        "or I can connect you with a human representative."
    )


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


def screen_user_turn(
    user_text: str,
    api_key: str | None = None,
    *,
    phase: str | None = None,
    last_agent_reply: str | None = None,
    state: AgentState | None = None,
) -> RailDecision:
    with logfire.span("guardrails.screen") as span:
        log_current_node("guardrails")
        guardrail_actions.reset_last_hit()
        guardrail_actions.set_groq_api_key(api_key)
        guardrail_actions.set_scope_context(
            phase=phase or (state or {}).get("phase") or "VERIFY_ID",
            last_agent_reply=(
                last_agent_reply
                if last_agent_reply is not None
                else str(memory_get(state, "last_agent_reply") or "")
            ),
        )
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
                message=REPLIES.get(category) or OFF_TOPIC_REPLY,
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
    message = decision.message
    if decision.category == "off_topic":
        message = off_topic_reply(updated)
    return updated, message
