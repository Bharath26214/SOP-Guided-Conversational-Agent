"""Dummy claims-support mailbox.

The agent always sends from a fixed from-address. Delivery writes an RFC 822
`.eml` file to the local outbox so the demo does not need SMTP.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path

import logfire
from langchain_core.tools import tool

from app.config import DUMMY_EMAIL_FROM
from app.data_loaders.db import DATA_DIR

OUTBOX_DIR = DATA_DIR / "outbox"


@dataclass(frozen=True)
class EmailSendResult:
    ok: bool
    path: str
    from_address: str
    subject: str


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")
    return (cleaned or "recipient")[:40]


def deliver_dummy_email(
    *,
    to_address: str,
    subject: str,
    body: str,
    from_address: str | None = None,
) -> EmailSendResult:
    """Write a sent message into the local outbox from the dummy mailbox."""
    sender = (from_address or DUMMY_EMAIL_FROM).strip()
    recipient = (to_address or "").strip()
    if not recipient or "@" not in recipient:
        raise ValueError("A valid recipient email is required.")
    if not (subject or "").strip() or not (body or "").strip():
        raise ValueError("Email subject and body are required.")

    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject.strip()
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid(domain="example.com")
    message.set_content(body.strip() + "\n")

    OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    local_part = recipient.split("@", 1)[0]
    path: Path = OUTBOX_DIR / f"{stamp}_{_safe_stem(local_part)}.eml"
    path.write_bytes(message.as_bytes())

    logfire.info(
        "Dummy email delivered",
        from_address=sender,
        outbox_path=str(path),
        subject=subject.strip(),
        body_chars=len(body.strip()),
    )
    return EmailSendResult(ok=True, path=str(path), from_address=sender, subject=subject.strip())


@tool("send_claims_email")
def send_claims_email(to_address: str, subject: str, body: str) -> str:
    """Send a claims conversation summary from the dummy claims-support mailbox."""
    result = deliver_dummy_email(to_address=to_address, subject=subject, body=body)
    return (
        f"Email sent from {result.from_address}. "
        f"A copy was saved to the local outbox at {result.path}."
    )
