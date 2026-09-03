from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("LANGSMITH_OTEL_ENABLED", "true")
os.environ.setdefault("LANGSMITH_OTEL_ONLY", "true")
os.environ.setdefault("LANGSMITH_TRACING", "true")

import logfire

_configured = False


def configure_logfire() -> None:
    global _configured
    if _configured:
        return

    service_name = os.getenv("LOGFIRE_SERVICE_NAME") or "sop-claims-agent"
    console_enabled = os.getenv("LOGFIRE_CONSOLE", "true").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    logfire.configure(
        service_name=service_name,
        send_to_logfire="if-token-present",
        inspect_arguments=False,
        console=logfire.ConsoleOptions(min_log_level="info") if console_enabled else False,
        scrubbing=logfire.ScrubbingOptions(
            extra_patterns=[
                r"ssn",
                r"social.?security",
                r"ssn_last4",
                r"id_last4",
                r"\bdob\b",
            ]
        ),
    )
    try:
        logfire.instrument_httpx()
    except Exception:
        pass
    _configured = True
    logfire.info("Logfire configured", service_name=service_name)


def safe_state_attrs(state: dict[str, Any] | None) -> dict[str, Any]:
    state = state or {}
    identity = state.get("identity") or {}
    return {
        "phase": state.get("phase"),
        "verified": state.get("verified"),
        "caller_role": state.get("caller_role"),
        "party_id": state.get("party_id"),
        "matched_pii": list(state.get("matched_pii") or []),
        "out_of_scope_count": state.get("out_of_scope_count"),
        "identity_fields": sorted(identity.keys()),
        "has_case_hint": bool((state.get("memory") or {}).get("case_hint")),
        "declined_ssn": bool((state.get("memory") or {}).get("declined_ssn")),
        "intent": state.get("intent"),
        "selected_case_id": state.get("selected_case_id"),
        "candidate_case_ids": list(state.get("candidate_case_ids") or []),
        "email_choice": state.get("email_choice"),
        "email_offered": bool((state.get("memory") or {}).get("email_offered")),
    }


def log_current_node(node: str, state: dict[str, Any] | None = None, **extra: Any) -> None:
    attrs = {**safe_state_attrs(state), **extra}
    logfire.info("Current node: {node}", node=node, **attrs)
