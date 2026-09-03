from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "DATA"
DB_PATH = DATA_DIR / "insurance.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS policyholders (
    party_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    policy_number TEXT,
    dob TEXT,
    id_type TEXT,
    id_last4 TEXT,
    phone TEXT,
    email TEXT,
    name_aliases TEXT,
    phone_aliases TEXT,
    email_aliases TEXT
);

CREATE TABLE IF NOT EXISTS claims (
    case_id TEXT PRIMARY KEY,
    party_id TEXT NOT NULL,
    case_type TEXT NOT NULL,
    created_at TEXT,
    status TEXT,
    summary TEXT,
    denial_reason TEXT,
    documents_needed TEXT,
    appeal_deadline TEXT,
    expected_reimbursement_amount TEXT,
    allowed_max_amount TEXT,
    net_pay TEXT,
    net_fee TEXT,
    FOREIGN KEY (party_id) REFERENCES policyholders(party_id)
);

CREATE TABLE IF NOT EXISTS representatives (
    rep_name TEXT PRIMARY KEY,
    relationship TEXT,
    buyer_name TEXT,
    buyer_party_id TEXT,
    FOREIGN KEY (buyer_party_id) REFERENCES policyholders(party_id)
);

CREATE TABLE IF NOT EXISTS reference_docs (
    doc_key TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
"""


def _read_json(filename: str) -> Any:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run(connection: sqlite3.Connection, sql: str, parameters: tuple | list = ()):
    # Cursor.execute is what Logfire instruments; Connection.execute is not.
    return connection.cursor().execute(sql, parameters)


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_SQL)
    connection.commit()


def _dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value)


def seed_from_json(connection: sqlite3.Connection) -> None:
    policyholders = _read_json("policyholders.json")
    claims = _read_json("claims.json")
    representatives = _read_json("representatives.json")

    run(connection, "DELETE FROM claims")
    run(connection, "DELETE FROM representatives")
    run(connection, "DELETE FROM policyholders")
    run(connection, "DELETE FROM reference_docs")

    for person in policyholders:
        run(
            connection,
            """
            INSERT INTO policyholders (
                party_id, name, policy_number, dob, id_type, id_last4,
                phone, email, name_aliases, phone_aliases, email_aliases
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                person.get("party_id"),
                person.get("name"),
                person.get("policy_number"),
                person.get("dob"),
                person.get("id_type"),
                person.get("id_last4"),
                person.get("phone"),
                person.get("email"),
                _dumps(person.get("name_aliases")),
                _dumps(person.get("phone_aliases")),
                _dumps(person.get("email_aliases")),
            ),
        )

    for claim in claims:
        run(
            connection,
            """
            INSERT INTO claims (
                case_id, party_id, case_type, created_at, status, summary,
                denial_reason, documents_needed, appeal_deadline,
                expected_reimbursement_amount, allowed_max_amount, net_pay, net_fee
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim.get("case_id"),
                claim.get("party_id"),
                claim.get("case_type"),
                claim.get("created_at"),
                claim.get("status"),
                claim.get("summary"),
                claim.get("denial_reason"),
                _dumps(claim.get("documents_needed")),
                claim.get("appeal_deadline"),
                claim.get("expected_reimbursement_amount"),
                claim.get("allowed_max_amount"),
                claim.get("net_pay"),
                claim.get("net_fee"),
            ),
        )

    for representative in representatives:
        run(
            connection,
            """
            INSERT INTO representatives (
                rep_name, relationship, buyer_name, buyer_party_id
            ) VALUES (?, ?, ?, ?)
            """,
            (
                representative.get("rep_name"),
                representative.get("relationship"),
                representative.get("buyer_name"),
                representative.get("buyer_party_id"),
            ),
        )

    for key, filename in (
        ("claim_schema", "claim_schema.json"),
        ("required_document_guideline", "required_document_guideline.json"),
        ("consent_scenarios", "consent_scenarios.json"),
    ):
        run(
            connection,
            "INSERT INTO reference_docs (doc_key, payload) VALUES (?, ?)",
            (key, json.dumps(_read_json(filename))),
        )

    connection.commit()


_ready = False


def _is_empty(connection: sqlite3.Connection) -> bool:
    has_policyholder = run(connection, "SELECT 1 FROM policyholders LIMIT 1").fetchone()
    if has_policyholder is None:
        return True
    has_claim = run(connection, "SELECT 1 FROM claims LIMIT 1").fetchone()
    return has_claim is None


def ensure_database() -> Path:
    global _ready
    if _ready and DB_PATH.exists():
        return DB_PATH
    with connect() as connection:
        init_schema(connection)
        if _is_empty(connection):
            seed_from_json(connection)
    _ready = True
    return DB_PATH
