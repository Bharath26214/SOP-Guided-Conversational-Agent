from __future__ import annotations

from typing import Any

import logfire

from app.agents.emotion import pushing_claim_before_verify, wants_human
from app.agents.identity_parse import normalize_dob, normalize_phone
from app.agents.nodes.state import AgentState
from app.data_loaders.queries import get_policyholder, get_representative, lookup_policyholders
from app.memory import affect_payload, flagged, get as memory_get, hint as case_hint
from app.observability import log_current_node, safe_state_attrs

PII_FIELDS = ("name", "dob", "phone", "email", "ssn_last4")
FIELD_LABELS = {
    "name": "full name",
    "dob": "date of birth",
    "phone": "phone number",
    "email": "email address",
    "ssn_last4": "SSN last four",
    "policy_number": "policy number",
}


def _norm_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _norm_email(value: str) -> str:
    return value.strip().lower()


def _norm_phone(value: str) -> str:
    return normalize_phone(value) or "".join(ch for ch in value if ch.isdigit())


def _norm_last4(value: str) -> str:
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits[-4:]


def _norm_dob(value: str) -> str | None:
    return normalize_dob(value)


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


def _labels(fields: list[str]) -> list[str]:
    return [FIELD_LABELS[field] for field in fields if field in FIELD_LABELS]


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
    identity = state.get("identity") or {}
    name = memory_get(state, "representative_name")
    if not name and state.get("caller_role") == "representative":
        name = identity.get("name")
    if not name:
        return None
    rep = get_representative(name)
    if not rep:
        return None
    relationship = memory_get(state, "relationship")
    if relationship and rep.get("relationship"):
        if _norm_name(str(relationship)) != _norm_name(str(rep["relationship"])):
            return None
    return rep


def _identity_for_match(state: AgentState) -> dict[str, Any]:
    identity = dict(state.get("identity") or {})
    rep = _resolve_representative(state)
    if identity.get("member_name"):
        identity["name"] = identity["member_name"]
        return identity
    caller_name = memory_get(state, "representative_name") or (
        identity.get("name") if state.get("caller_role") == "representative" else None
    )
    if caller_name and identity.get("name") and _norm_name(identity["name"]) == _norm_name(caller_name):
        identity.pop("name", None)
    if rep and identity.get("name") and _norm_name(identity["name"]) == _norm_name(rep["rep_name"]):
        identity.pop("name", None)
    return identity


def _ask_order(state: AgentState) -> tuple[str, ...]:
    if flagged(state, "declined_ssn"):
        return tuple(field for field in PII_FIELDS if field != "ssn_last4")
    return PII_FIELDS


def _missing_pii(state: AgentState) -> list[str]:
    identity = _identity_for_match(state)
    given = _given_pii(identity)
    return [field for field in _ask_order(state) if field not in given]


def _noted_claim(state: AgentState) -> dict[str, str]:
    stored = case_hint(state)
    noted: dict[str, str] = {}
    for field in ("case_id", "case_type", "status", "month", "year"):
        value = stored.get(field)
        if value:
            noted[field] = str(value)
    return noted


def _have_labels(state: AgentState, given: dict[str, str]) -> list[str]:
    labels = _labels(list(given.keys()))
    if (state.get("identity") or {}).get("policy_number") and "policy number" not in labels:
        labels.append("policy number")
    return labels


def _facts(state: AgentState, extra: dict[str, Any], user_text: str = "") -> dict[str, Any]:
    given = _given_pii(_identity_for_match(state))
    payload = {
        "phase": "VERIFY_ID",
        "verified": bool(state.get("verified")),
        "caller_role": state.get("caller_role") or "policyholder",
        "declined_ssn": flagged(state, "declined_ssn"),
        "has_policy_number": bool((state.get("identity") or {}).get("policy_number")),
        "have": extra.get("have") or _have_labels(state, given),
        "noted_claim": extra.get("noted_claim") or _noted_claim(state),
        "asked_for_claim": extra.get("asked_for_claim", False),
        "last_agent_reply": memory_get(state, "last_agent_reply") or "",
        **affect_payload(state),
        **extra,
    }
    payload["utterance"] = user_text or str(payload.get("utterance") or "")
    return payload



def _lookup_people(identity: dict[str, Any]) -> list[dict[str, Any]]:
    policy_number = str(identity.get("policy_number") or "").strip()
    given = _given_pii(identity)
    if not policy_number and not any(given.get(field) for field in ("name", "phone", "email", "ssn_last4")):
        return []
    return lookup_policyholders(
        policy_number=policy_number or None,
        name=given.get("name"),
        email=given.get("email"),
        phone_digits=given.get("phone"),
        ssn_last4=given.get("ssn_last4"),
    )


def _name_on_file(name: str, people: list[dict[str, Any]]) -> bool:
    return any(_field_matches("name", name, person) for person in people)


def find_candidate(identity: dict[str, Any]) -> dict[str, Any] | None:
    people = _lookup_people(identity)
    return _pick_candidate(identity, people)


def _pick_candidate(identity: dict[str, Any], people: list[dict[str, Any]]) -> dict[str, Any] | None:
    policy_number = str(identity.get("policy_number") or "").strip()
    given = _given_pii(identity)
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


def _member_from_rep(rep: dict[str, Any]) -> dict[str, Any] | None:
    buyer_id = rep.get("buyer_party_id")
    if not buyer_id:
        return None
    return get_policyholder(str(buyer_id))


def run_verify_id(state: AgentState, user_text: str = "") -> tuple[AgentState, dict[str, Any]]:
    log_current_node("verify_id", state)
    with logfire.span("verify_id.run", **safe_state_attrs(state)) as span:
        updated, facts = _run_verify_id(state, user_text)
        span.set_attributes(safe_state_attrs(updated))
        span.set_attribute("already_verified", bool(state.get("verified")))
        span.set_attribute("affect", facts.get("affect"))
        span.set_attribute("affect_tone", facts.get("affect_tone"))
        return updated, facts


def _run_verify_id(state: AgentState, user_text: str = "") -> tuple[AgentState, dict[str, Any]]:
    if state.get("verified"):
        return state, _facts(state, {"status": "already_verified"}, user_text)

    updated: AgentState = {**state}
    asked_for_claim = pushing_claim_before_verify(user_text)
    match_identity = _identity_for_match(updated)
    given = _given_pii(match_identity)
    has_lookup_key = bool(
        match_identity.get("policy_number")
        or given.get("name")
        or given.get("phone")
        or given.get("email")
        or given.get("ssn_last4")
    )

    def pack(extra: dict[str, Any]) -> dict[str, Any]:
        return _facts(
            updated,
            {"asked_for_claim": asked_for_claim, **extra},
            user_text,
        )

    def need_more(need: int, status: str = "need_more") -> tuple[AgentState, dict[str, Any]]:
        packed = pack(
            {
                "status": status,
                "have": _have_labels(updated, given),
                "noted_claim": _noted_claim(updated),
                "matched": _labels(list(updated.get("matched_pii") or [])),
                "need": need,
                "missing": _labels(_missing_pii(updated)),
            },
        )
        if packed.get("affect_tone") == "escalate" or wants_human(user_text):
            updated["phase"] = "HUMAN_ESCALATION"
            updated["verified"] = False
            packed["status"] = "escalate_affect"
            packed["phase"] = "VERIFY_ID"
            return updated, packed
        return updated, packed

    if state.get("caller_role") == "representative":
        rep_name = memory_get(state, "representative_name") or (state.get("identity") or {}).get("name")
        if not rep_name:
            updated["verified"] = False
            return updated, pack({"status": "need_rep_name"})
        rep = _resolve_representative(updated)
        if rep is None:
            updated["verified"] = False
            return updated, pack({"status": "rep_not_on_file"})
        candidate = _member_from_rep(rep)
    elif not has_lookup_key:
        updated["matched_pii"] = []
        updated["verified"] = False
        return need_more(3)
    else:
        people = _lookup_people(match_identity)
        name_unknown = bool(given.get("name") and not _name_on_file(given["name"], people))
        if not people:
            updated["matched_pii"] = []
            updated["verified"] = False
            updated["phase"] = "DONE"
            return updated, pack(
                {"status": "not_on_file", "have": _have_labels(updated, given), "need": 0, "missing": []},
            )
        candidate = _pick_candidate(match_identity, people)
        if candidate is None and name_unknown:
            updated["matched_pii"] = []
            updated["verified"] = False
            updated["phase"] = "DONE"
            return updated, pack(
                {"status": "not_on_file", "have": _have_labels(updated, given), "need": 0, "missing": []},
            )

    if candidate is None:
        updated["matched_pii"] = []
        updated["verified"] = False
        if len(given) >= 3:
            updated["phase"] = "DONE"
            return need_more(0, "not_on_file")
        return need_more(3 if not given else max(1, 3 - len(given)), "no_match_yet")

    matched, incorrect = compare_pii(match_identity, candidate)
    updated["matched_pii"] = matched

    if incorrect:
        updated["verified"] = False
        return updated, pack(
            {"status": "mismatch", "incorrect": _labels(incorrect), "missing": _labels(_missing_pii(updated))},
        )

    if len(matched) >= 3:
        updated["verified"] = True
        updated["party_id"] = candidate["party_id"]
        updated["phase"] = "RESOLVE_INTENT"
        return updated, pack(
            {"status": "verified", "representative": state.get("caller_role") == "representative"},
        )

    updated["verified"] = False
    return need_more(max(1, 3 - len(matched)))
