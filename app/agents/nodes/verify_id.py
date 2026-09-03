from __future__ import annotations

from datetime import datetime
from typing import Any

import logfire

from app.agents.nodes.state import AgentState
from app.data_loaders.queries import get_policyholder, get_representative, lookup_policyholders
from app.observability import log_current_node, safe_state_attrs

PII_FIELDS = ("name", "dob", "phone", "email", "ssn_last4")
FIELD_LABELS = {
    "name": "full name",
    "dob": "date of birth",
    "phone": "phone number",
    "email": "email address",
    "ssn_last4": "SSN or national ID last four digits",
    "policy_number": "policy number",
}
CONFIDENTIAL_NOTE = (
    "I cannot share any claim details until your identity is verified."
)
LAST4_NOTE = (
    "You do not need to provide a full Social Security number; only the last four digits are needed."
)
ANY_THREE_NOTE = (
    "Any 3 of full name, date of birth, phone number, email address, "
    "or SSN last four digits will verify your identity."
)


def _norm_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _norm_email(value: str) -> str:
    return value.strip().lower()


def _norm_phone(value: str) -> str:
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits[-10:] if len(digits) >= 10 else digits


def _norm_last4(value: str) -> str:
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits[-4:]


def _norm_dob(value: str) -> str | None:
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text.lower() or None


def _names_for(record: dict[str, Any]) -> list[str]:
    names = [record.get("name") or ""]
    names.extend(record.get("name_aliases") or [])
    return [name for name in names if name]


def _phones_for(record: dict[str, Any]) -> list[str]:
    phones = [record.get("phone") or ""]
    phones.extend(record.get("phone_aliases") or [])
    return [phone for phone in phones if phone]


def _emails_for(record: dict[str, Any]) -> list[str]:
    emails = [record.get("email") or ""]
    emails.extend(record.get("email_aliases") or [])
    return [email for email in emails if email]


def _field_matches(field: str, offered: str, record: dict[str, Any]) -> bool:
    if field == "name":
        offered_name = _norm_name(offered)
        return any(_norm_name(name) == offered_name for name in _names_for(record))
    if field == "dob":
        return _norm_dob(offered) == _norm_dob(str(record.get("dob") or ""))
    if field == "phone":
        offered_phone = _norm_phone(offered)
        return any(_norm_phone(phone) == offered_phone for phone in _phones_for(record))
    if field == "email":
        offered_email = _norm_email(offered)
        return any(_norm_email(email) == offered_email for email in _emails_for(record))
    if field == "ssn_last4":
        return _norm_last4(offered) == _norm_last4(str(record.get("id_last4") or ""))
    return False


def _join_labels(fields: list[str]) -> str:
    labels = [FIELD_LABELS[field] for field in fields]
    if not labels:
        return "identity details"
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _given_pii(identity: dict[str, Any]) -> dict[str, str]:
    given: dict[str, str] = {}
    for field in PII_FIELDS:
        value = identity.get(field)
        if isinstance(value, str) and value.strip():
            given[field] = value.strip()
    return given


def _resolve_representative(state: AgentState) -> dict[str, Any] | None:
    if state.get("caller_role") == "policyholder":
        return None
    memory = state.get("memory") or {}
    identity = state.get("identity") or {}
    name = memory.get("representative_name")
    if not name and state.get("caller_role") == "representative":
        name = identity.get("name")
    if not name:
        return None
    rep = get_representative(name)
    if not rep:
        return None
    relationship = memory.get("relationship")
    if relationship and rep.get("relationship"):
        if _norm_name(str(relationship)) != _norm_name(str(rep["relationship"])):
            return None
    return rep


def _identity_for_match(state: AgentState) -> dict[str, Any]:
    identity = dict(state.get("identity") or {})
    memory = state.get("memory") or {}
    rep = _resolve_representative(state)
    if identity.get("member_name"):
        identity["name"] = identity["member_name"]
        return identity
    caller_name = memory.get("representative_name") or (
        identity.get("name") if state.get("caller_role") == "representative" else None
    )
    if caller_name and identity.get("name") and _norm_name(identity["name"]) == _norm_name(caller_name):
        identity.pop("name", None)
    if rep and identity.get("name") and _norm_name(identity["name"]) == _norm_name(rep["rep_name"]):
        identity.pop("name", None)
    return identity


def _ask_order(state: AgentState) -> tuple[str, ...]:
    if (state.get("memory") or {}).get("declined_ssn"):
        return tuple(field for field in PII_FIELDS if field != "ssn_last4")
    return PII_FIELDS


def _ssn_guidance(state: AgentState, asked: list[str]) -> str:
    declined = bool((state.get("memory") or {}).get("declined_ssn"))
    if declined:
        return f" {LAST4_NOTE} You may use full name, date of birth, phone, or email instead."
    if "ssn_last4" in asked:
        return f" {LAST4_NOTE}"
    return ""


def _missing_pii(state: AgentState) -> list[str]:
    identity = _identity_for_match(state)
    given = _given_pii(identity)
    return [field for field in _ask_order(state) if field not in given]


def _please_share(state: AgentState, fields: list[str], need: int | None = None) -> str:
    remaining = fields or _missing_pii(state)
    if not remaining:
        return "Please share more identity details."
    labels = _join_labels(remaining)
    whose = "the policyholder's" if state.get("caller_role") == "representative" else "your"
    count = need if need is not None else min(3, len(remaining))
    if count >= 3 and len(remaining) >= 4:
        return f"Please share any 3 of {whose} {labels}."
    if count <= 1:
        return f"Please share one more of {whose} {labels}."
    return f"Please share {count} more of {whose} {labels}."


def _acknowledge(state: AgentState) -> str:
    identity = dict(state.get("identity") or {})
    memory = state.get("memory") or {}
    noted = [field for field in (*PII_FIELDS, "policy_number") if identity.get(field)]
    if state.get("caller_role") == "representative" and memory.get("representative_name"):
        prefix = f"I have you as {memory['representative_name']}"
        if memory.get("relationship"):
            prefix += f", {memory['relationship']} of the policyholder"
        prefix += "."
        if not noted:
            return prefix
        return f"{prefix} I have noted the policyholder's {_join_labels(noted)}."
    if not noted:
        return "I do not have enough identity details yet."
    return f"I have noted your {_join_labels(noted)}."


def find_candidate(identity: dict[str, Any]) -> dict[str, Any] | None:
    policy_number = str(identity.get("policy_number") or "").strip()
    given = _given_pii(identity)
    if not policy_number and not any(given.get(field) for field in ("name", "phone", "email", "ssn_last4")):
        return None

    people = lookup_policyholders(
        policy_number=policy_number or None,
        name=given.get("name"),
        email=given.get("email"),
        phone_digits=given.get("phone"),
        ssn_last4=given.get("ssn_last4"),
    )
    if not people:
        return None

    if policy_number:
        for person in people:
            if str(person.get("policy_number") or "").strip().upper() == policy_number.upper():
                return person

    if given.get("name"):
        named = [person for person in people if _field_matches("name", given["name"], person)]
        if len(named) == 1:
            return named[0]
        if len(named) > 1:
            narrowed = named
            for field, value in given.items():
                if field == "name":
                    continue
                narrowed = [person for person in narrowed if _field_matches(field, value, person)]
            if len(narrowed) == 1:
                return narrowed[0]
            return None

    for field in ("phone", "email", "ssn_last4"):
        if not given.get(field):
            continue
        matches = [person for person in people if _field_matches(field, given[field], person)]
        if len(matches) == 1:
            return matches[0]
    return None


def compare_pii(identity: dict[str, Any], record: dict[str, Any]) -> tuple[list[str], list[str]]:
    matched: list[str] = []
    incorrect: list[str] = []
    for field, value in _given_pii(identity).items():
        if _field_matches(field, value, record):
            matched.append(field)
        else:
            incorrect.append(field)
    return matched, incorrect


def _reply_need_more(state: AgentState, matched: list[str], need: int) -> str:
    missing = _missing_pii(state)
    ack = _acknowledge(state)
    if missing:
        return (
            f"{ack} I have {len(matched)} matching detail"
            f"{'' if len(matched) == 1 else 's'} so far. "
            f"{_please_share(state, missing, need)}{_ssn_guidance(state, missing)} {CONFIDENTIAL_NOTE}"
        )
    return f"{ack} I still need more identity details.{_ssn_guidance(state, [])} {CONFIDENTIAL_NOTE}"


def _member_from_rep(rep: dict[str, Any]) -> dict[str, Any] | None:
    buyer_id = rep.get("buyer_party_id")
    if not buyer_id:
        return None
    return get_policyholder(str(buyer_id))


def run_verify_id(state: AgentState) -> tuple[AgentState, str]:
    log_current_node("verify_id", state)
    with logfire.span("verify_id.run", **safe_state_attrs(state)) as span:
        updated, message = _run_verify_id(state)
        span.set_attributes(safe_state_attrs(updated))
        span.set_attribute("already_verified", bool(state.get("verified")))
        return updated, message


def _run_verify_id(state: AgentState) -> tuple[AgentState, str]:
    if state.get("verified"):
        return state, "Your identity is already verified."

    updated: AgentState = {**state}
    memory = dict(state.get("memory") or {})
    match_identity = _identity_for_match(updated)
    given = _given_pii(match_identity)
    has_lookup_key = bool(
        match_identity.get("policy_number")
        or given.get("name")
        or given.get("phone")
        or given.get("email")
        or given.get("ssn_last4")
    )

    if state.get("caller_role") == "representative":
        rep_name = memory.get("representative_name") or (state.get("identity") or {}).get("name")
        if not rep_name:
            updated["verified"] = False
            return (
                updated,
                (
                    "I can help if you are an authorized representative. "
                    "Please share your full name and relationship to the policyholder, "
                    f"then any 3 of the policyholder's full name, date of birth, phone number, "
                    f"email address, or SSN last four digits. {LAST4_NOTE} {CONFIDENTIAL_NOTE}"
                ),
            )
        rep = _resolve_representative(updated)
        if rep is None:
            updated["verified"] = False
            return (
                updated,
                (
                    "I do not have you on file as an authorized representative for this policyholder. "
                    "The policyholder can verify, or I can connect you with a human representative. "
                    f"{CONFIDENTIAL_NOTE}"
                ),
            )
        candidate = _member_from_rep(rep)
    elif not has_lookup_key:
        updated["matched_pii"] = []
        updated["verified"] = False
        missing = _missing_pii(updated)
        ask = f" {_please_share(updated, missing, 3)}" if missing else f" {ANY_THREE_NOTE}"
        return (
            updated,
            f"{_acknowledge(updated)}{ask}{_ssn_guidance(updated, missing)} {CONFIDENTIAL_NOTE}",
        )
    else:
        candidate = find_candidate(match_identity)

    if candidate is None:
        updated["matched_pii"] = []
        updated["verified"] = False
        missing = _missing_pii(updated)
        need = 3 if not given else max(1, 3 - len(given))
        ask = f" {_please_share(updated, missing, need)}" if missing else f" {ANY_THREE_NOTE}"
        return (
            updated,
            f"{_acknowledge(updated)}{ask}{_ssn_guidance(updated, missing)} {CONFIDENTIAL_NOTE}",
        )

    matched, incorrect = compare_pii(match_identity, candidate)
    updated["matched_pii"] = matched

    if incorrect:
        updated["verified"] = False
        return (
            updated,
            (
                f"{_acknowledge(updated)} The {_join_labels(incorrect)} I received "
                f"{'does' if len(incorrect) == 1 else 'do'} not match our records. "
                f"Please try again with the correct {_join_labels(incorrect)}. "
                f"{_ssn_guidance(updated, incorrect)} {CONFIDENTIAL_NOTE}"
            ),
        )

    if len(matched) >= 3:
        updated["verified"] = True
        updated["party_id"] = candidate["party_id"]
        updated["phase"] = "RESOLVE_INTENT"
        if state.get("caller_role") == "representative":
            return (
                updated,
                "Thank you. I have verified you as an authorized representative for this policyholder.",
            )
        return updated, "Thank you. I have verified your identity."

    if len(matched) == 2:
        updated["verified"] = False
        return updated, _reply_need_more(updated, matched, 1)

    if len(matched) == 1:
        updated["verified"] = False
        return updated, _reply_need_more(updated, matched, 2)

    updated["verified"] = False
    missing = _missing_pii(updated)
    ask = f" {_please_share(updated, missing, 3)}" if missing else f" {ANY_THREE_NOTE}"
    return (
        updated,
        f"{_acknowledge(updated)}{ask}{_ssn_guidance(updated, missing)} {CONFIDENTIAL_NOTE}",
    )
