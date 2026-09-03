from __future__ import annotations

import re

import logfire

from app.agents.nodes.state import AgentMemory, AgentState, EmailChoice
from app.agents.tools.email import send_claims_email
from app.config import DUMMY_EMAIL_FROM
from app.data_loaders.queries import get_claim, get_policyholder, load_required_document_guideline
from app.observability import log_current_node, safe_state_attrs

OFFER_REPLY = (
    "Would you like me to email you a summary of this conversation, including what we "
    "discussed, the claim status or outcome, and the main follow-up items? "
    "You can say send or skip."
)
ASK_AGAIN_REPLY = (
    "Please say send if you want the summary emailed, or skip if you do not."
)
SKIP_REPLY = "Understood. I will not send an email. This conversation is now complete. Goodbye."
SENT_REPLY = (
    "Thank you. I have emailed you a summary. This conversation is now complete. Goodbye."
)
DONE_REPLY = "This conversation is already complete. Reset the chat if you need to start again."
NO_RECIPIENT_REPLY = (
    "Thank you. I do not have an email address on file, so I cannot send a summary. "
    "This conversation is now complete. Goodbye."
)

SEND_PHRASES = (
    "send the email",
    "send an email",
    "email me a summary",
    "email the summary",
    "send a summary",
    "send the summary",
    "yes send",
    "please send",
    "send it to my email",
    "send that to my email",
)
EMAIL_ME_PHRASES = ("email me", "e-mail me")
SKIP_PHRASES = (
    "skip the email",
    "skip email",
    "don't send",
    "do not send",
    "no email",
    "don't email",
    "do not email",
    "no summary",
)
WRAP_PHRASES = (
    "that's all",
    "that is all",
    "that will be all",
    "nothing else",
    "no more questions",
    "that's it",
    "that is it",
    "all set",
    "we're done",
    "we are done",
    "we're all set",
    "goodbye",
    "good bye",
    "bye",
)
YES_SET = {"yes", "yeah", "yep", "ok", "okay", "please", "send"}
SKIP_SET = {"no", "nope", "skip", "no thanks", "no thank you", "not now", "later"}
FOLLOWUP_HINTS = (
    "claim",
    "status",
    "denied",
    "denial",
    "document",
    "deadline",
    "appeal",
    "amount",
    "portal",
    "submit",
    "upload",
    "pathology",
    "office note",
    "reimburse",
    "why",
    "how long",
    "how soon",
)


def _memory(state: AgentState) -> AgentMemory:
    return dict(state.get("memory") or {})


def wants_email_send(text: str, *, offered: bool = False) -> bool:
    lowered = (text or "").lower().strip()
    if not lowered:
        return False
    if any(phrase in lowered for phrase in SEND_PHRASES):
        return True
    if any(phrase in lowered for phrase in EMAIL_ME_PHRASES):
        if looks_like_case_followup(lowered) and "summary" not in lowered:
            return False
        return True
    return offered and lowered in YES_SET


def wants_email_skip(text: str, *, offered: bool = False) -> bool:
    lowered = (text or "").lower().strip()
    if not lowered:
        return False
    if any(phrase in lowered for phrase in SKIP_PHRASES):
        return True
    return offered and lowered in SKIP_SET


def wants_wrap_up(text: str) -> bool:
    lowered = (text or "").lower().strip()
    if not lowered:
        return False
    if any(phrase in lowered for phrase in WRAP_PHRASES):
        return True
    compact = re.sub(r"[^a-z\s']", " ", lowered)
    tokens = compact.split()
    if len(tokens) > 6:
        return False
    if not any(token in compact for token in ("thank you", "thanks", "thx")):
        return False
    if "?" in (text or "") or any(hint in compact for hint in FOLLOWUP_HINTS):
        return False
    return True


def looks_like_case_followup(text: str) -> bool:
    lowered = (text or "").lower()
    if "cl-" in lowered:
        return True
    return any(hint in lowered for hint in FOLLOWUP_HINTS)


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _recipient(state: AgentState) -> tuple[str | None, str]:
    person = get_policyholder(str(state.get("party_id") or "")) or {}
    name = str(person.get("name") or (state.get("identity") or {}).get("name") or "Member")
    email = str(person.get("email") or "").strip()
    return (email or None), name


def build_conversation_summary(state: AgentState) -> tuple[str, str]:
    """Grounded summary from the claim file and what this call already covered."""
    _, name = _recipient(state)
    case_id = state.get("selected_case_id")
    claim = get_claim(str(case_id)) if case_id else None
    guideline = load_required_document_guideline()
    notes = [str(item).strip() for item in (_memory(state).get("case_answers") or []) if str(item).strip()]
    topics = [str(item) for item in (_memory(state).get("process_topics") or []) if item]

    subject_id = str(case_id or "your claim")
    subject = f"Summary of your claims support call - {subject_id}"

    lines = [
        f"Hello {name},",
        "",
        "Here is a summary of today's insurance claims support conversation.",
        "",
        "What we discussed",
    ]
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    elif topics:
        lines.append("- We reviewed your claim and related follow-up questions.")
    else:
        lines.append("- We completed identity verification and reviewed your request.")

    lines.extend(["", "Claim status / outcome"])
    if claim:
        lines.append(
            f"- Claim {claim.get('case_id')} is a {claim.get('case_type')} claim "
            f"created on {claim.get('created_at')}. Status on file: {claim.get('status')}."
        )
        if claim.get("summary"):
            lines.append(f"- {str(claim['summary']).rstrip('.')}.")
        if claim.get("denial_reason"):
            lines.append(f"- Denial reason on file: {claim['denial_reason']}.")
        net_pay = claim.get("net_pay")
        if claim.get("status") == "closed" and net_pay not in (None, ""):
            lines.append(f"- Paid amount on file: {net_pay}.")
    elif state.get("intent") == "policy_inquiry":
        person = get_policyholder(str(state.get("party_id") or "")) or {}
        policy_number = person.get("policy_number") or (state.get("identity") or {}).get("policy_number")
        if policy_number:
            lines.append(f"- Policy {policy_number} is on file.")
        lines.append(
            "- Coverage, premium, and benefit details are not in this claims file, "
            "so those questions were referred for human help."
        )
    else:
        lines.append("- No specific claim was selected on this call.")

    lines.extend(["", "Follow-up items / next steps"])
    followups: list[str] = []
    if claim:
        docs = [str(item) for item in (claim.get("documents_needed") or []) if item]
        if docs:
            followups.append(f"Submit {_join(docs)} so the file can be reviewed again.")
            default = (guideline.get("default_guidance") or {}).get("en")
            if default:
                followups.append(default)
        deadline = claim.get("appeal_deadline")
        if deadline:
            followups.append(f"Appeal deadline on file: {deadline}.")
        if not followups and claim.get("status") == "closed":
            followups.append("No further documents are listed on this closed claim.")
        if not followups and claim.get("status") == "open":
            followups.append("The claim remains open. Watch the member portal for updates.")
    if state.get("intent") == "policy_inquiry" and not claim:
        followups.append("Ask a human representative if you still need coverage or benefit details.")
    if not followups:
        followups.append("No additional follow-up items were recorded on this call.")
    for item in followups:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            f"This message was sent from {DUMMY_EMAIL_FROM} by insurance claims support.",
            "This is a demo mailbox; the message is also stored in the local outbox.",
        ]
    )
    return subject, "\n".join(lines)


def _finish(state: AgentState, choice: EmailChoice, message: str) -> tuple[AgentState, str]:
    memory = _memory(state)
    memory["email_offered"] = True
    updated: AgentState = {
        **state,
        "phase": "DONE",
        "email_choice": choice,
        "memory": memory,
    }
    return updated, message


def _offer(state: AgentState) -> tuple[AgentState, str]:
    memory = _memory(state)
    memory["email_offered"] = True
    updated: AgentState = {**state, "phase": "POST_PROCESS", "memory": memory}
    recipient, _ = _recipient(updated)
    extra = (
        f" I would send it from {DUMMY_EMAIL_FROM} to the email address on file."
        if recipient
        else " I do not see an email address on file, so I can only skip sending."
    )
    return updated, OFFER_REPLY + extra


def _send(state: AgentState) -> tuple[AgentState, str]:
    recipient, _ = _recipient(state)
    if not recipient:
        logfire.info("post_process send skipped; no recipient on file")
        return _finish(state, "skip", NO_RECIPIENT_REPLY)

    subject, body = build_conversation_summary(state)
    send_claims_email.invoke(
        {"to_address": recipient, "subject": subject, "body": body}
    )
    logfire.info("post_process emailed summary", selected_case_id=state.get("selected_case_id"))
    return _finish(state, "send", SENT_REPLY)


def run_post_process(state: AgentState, user_text: str) -> tuple[AgentState, str]:
    log_current_node("post_process", state)
    with logfire.span("post_process.run", **safe_state_attrs(state)) as span:
        updated, message = _run_post_process(state, user_text)
        span.set_attributes(safe_state_attrs(updated))
        span.set_attribute("email_choice", updated.get("email_choice"))
        span.set_attribute("phase", updated.get("phase"))
        return updated, message


def _run_post_process(state: AgentState, user_text: str) -> tuple[AgentState, str]:
    if state.get("phase") == "DONE" or state.get("email_choice") in {"send", "skip"}:
        return state, DONE_REPLY

    if not state.get("verified"):
        return state, "I still need to verify your identity before I can email a summary."

    offered = bool((_memory(state).get("email_offered")))
    text = user_text or ""

    if (
        offered
        and looks_like_case_followup(text)
        and not wants_email_send(text, offered=True)
        and not wants_email_skip(text, offered=True)
    ):
        updated: AgentState = {**state, "phase": "PROCESS_CASE"}
        logfire.info("post_process returned to process_case")
        return updated, ""

    if wants_email_send(text, offered=offered):
        return _send(state)
    if wants_email_skip(text, offered=offered):
        logfire.info("post_process skipped email")
        return _finish(state, "skip", SKIP_REPLY)
    if offered:
        return {**state, "phase": "POST_PROCESS"}, ASK_AGAIN_REPLY
    return _offer(state)
