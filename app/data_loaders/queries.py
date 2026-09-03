from __future__ import annotations

import json
from typing import Any

from app.data_loaders.db import connect, ensure_database, run


def _parse_json(value: str | None) -> Any:
    if not value:
        return None
    return json.loads(value)


def _policyholder_from_row(row) -> dict[str, Any]:
    person = {
        "party_id": row["party_id"],
        "name": row["name"],
        "policy_number": row["policy_number"],
        "dob": row["dob"],
        "id_type": row["id_type"],
        "id_last4": row["id_last4"],
        "phone": row["phone"],
        "email": row["email"],
    }
    for field in ("name_aliases", "phone_aliases", "email_aliases"):
        parsed = _parse_json(row[field])
        if parsed:
            person[field] = parsed
    return person


def _claim_from_row(row) -> dict[str, Any]:
    claim = {
        "case_id": row["case_id"],
        "party_id": row["party_id"],
        "case_type": row["case_type"],
        "created_at": row["created_at"],
        "status": row["status"],
        "summary": row["summary"],
        "expected_reimbursement_amount": row["expected_reimbursement_amount"],
        "allowed_max_amount": row["allowed_max_amount"],
        "net_pay": row["net_pay"],
        "net_fee": row["net_fee"],
    }
    if row["denial_reason"]:
        claim["denial_reason"] = row["denial_reason"]
    documents = _parse_json(row["documents_needed"])
    if documents:
        claim["documents_needed"] = documents
    if row["appeal_deadline"]:
        claim["appeal_deadline"] = row["appeal_deadline"]
    return claim


def _representative_from_row(row) -> dict[str, Any]:
    return {
        "rep_name": row["rep_name"],
        "relationship": row["relationship"],
        "buyer_name": row["buyer_name"],
        "buyer_party_id": row["buyer_party_id"],
    }


def load_policyholders() -> list[dict[str, Any]]:
    ensure_database()
    with connect() as connection:
        rows = run(connection, "SELECT * FROM policyholders ORDER BY party_id").fetchall()
    return [_policyholder_from_row(row) for row in rows]


def load_claims() -> list[dict[str, Any]]:
    ensure_database()
    with connect() as connection:
        rows = run(connection, "SELECT * FROM claims ORDER BY case_id").fetchall()
    return [_claim_from_row(row) for row in rows]


def load_representatives() -> list[dict[str, Any]]:
    ensure_database()
    with connect() as connection:
        rows = run(connection, "SELECT * FROM representatives ORDER BY rep_name").fetchall()
    return [_representative_from_row(row) for row in rows]


def _load_reference_doc(doc_key: str) -> dict[str, Any]:
    ensure_database()
    with connect() as connection:
        row = run(
            connection,
            "SELECT payload FROM reference_docs WHERE doc_key = ?",
            (doc_key,),
        ).fetchone()
    if row is None:
        raise KeyError(f"Missing reference document: {doc_key}")
    return json.loads(row["payload"])


def load_claim_schema() -> dict[str, Any]:
    return _load_reference_doc("claim_schema")


def load_required_document_guideline() -> dict[str, Any]:
    return _load_reference_doc("required_document_guideline")


def load_consent_scenarios() -> dict[str, Any]:
    return _load_reference_doc("consent_scenarios")


def load_all() -> dict[str, Any]:
    return {
        "policyholders": load_policyholders(),
        "claims": load_claims(),
        "representatives": load_representatives(),
        "claim_schema": load_claim_schema(),
        "required_document_guideline": load_required_document_guideline(),
        "consent_scenarios": load_consent_scenarios(),
    }


def lookup_policyholders(
    *,
    policy_number: str | None = None,
    name: str | None = None,
    email: str | None = None,
    phone_digits: str | None = None,
    ssn_last4: str | None = None,
    party_id: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch rows that could match identifiers the caller already provided."""
    clauses: list[str] = []
    params: list[str] = []
    if party_id:
        clauses.append("party_id = ?")
        params.append(party_id)
    if policy_number:
        clauses.append("UPPER(policy_number) = ?")
        params.append(policy_number.strip().upper())
    if name:
        needle = " ".join(name.strip().lower().split())
        clauses.append("LOWER(TRIM(name)) = ?")
        params.append(needle)
        clauses.append("LOWER(COALESCE(name_aliases, '')) LIKE ?")
        params.append(f"%{needle}%")
    if email:
        needle = email.strip().lower()
        clauses.append("LOWER(email) = ?")
        params.append(needle)
        clauses.append("LOWER(COALESCE(email_aliases, '')) LIKE ?")
        params.append(f"%{needle}%")
    if phone_digits:
        digits = "".join(ch for ch in phone_digits if ch.isdigit())
        if digits:
            tail = digits[-10:]
            stripped = (
                "REPLACE(REPLACE(REPLACE(REPLACE(REPLACE("
                "COALESCE(phone, '') || COALESCE(phone_aliases, ''), "
                "'-', ''), ' ', ''), '(', ''), ')', ''), '+', '')"
            )
            clauses.append(f"{stripped} LIKE ?")
            params.append(f"%{tail}%")
    if ssn_last4:
        digits = "".join(ch for ch in ssn_last4 if ch.isdigit())[-4:]
        if digits:
            clauses.append("id_last4 = ?")
            params.append(digits)
    if not clauses:
        return []

    ensure_database()
    sql = "SELECT * FROM policyholders WHERE " + " OR ".join(clauses)
    with connect() as connection:
        rows = run(connection, sql, params).fetchall()
    return [_policyholder_from_row(row) for row in rows]


def get_policyholder_by_policy_number(policy_number: str) -> dict[str, Any] | None:
    needle = policy_number.strip().upper()
    ensure_database()
    with connect() as connection:
        row = run(
            connection,
            "SELECT * FROM policyholders WHERE UPPER(policy_number) = ?",
            (needle,),
        ).fetchone()
    return _policyholder_from_row(row) if row else None


def get_policyholder(party_id: str) -> dict[str, Any] | None:
    ensure_database()
    with connect() as connection:
        row = run(
            connection,
            "SELECT * FROM policyholders WHERE party_id = ?",
            (party_id,),
        ).fetchone()
    return _policyholder_from_row(row) if row else None


def get_claim(case_id: str) -> dict[str, Any] | None:
    ensure_database()
    with connect() as connection:
        row = run(
            connection,
            "SELECT * FROM claims WHERE UPPER(case_id) = ?",
            (case_id.strip().upper(),),
        ).fetchone()
    return _claim_from_row(row) if row else None


def get_claims_for_party(party_id: str) -> list[dict[str, Any]]:
    ensure_database()
    with connect() as connection:
        rows = run(
            connection,
            "SELECT * FROM claims WHERE party_id = ? ORDER BY created_at",
            (party_id,),
        ).fetchall()
    return [_claim_from_row(row) for row in rows]


def get_representative(rep_name: str) -> dict[str, Any] | None:
    ensure_database()
    with connect() as connection:
        row = run(
            connection,
            "SELECT * FROM representatives WHERE LOWER(rep_name) = ?",
            (rep_name.strip().lower(),),
        ).fetchone()
    return _representative_from_row(row) if row else None
