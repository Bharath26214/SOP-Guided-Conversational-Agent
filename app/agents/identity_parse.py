"""Deterministic identity parsing so messy utterances still fill AgentState."""

from __future__ import annotations

import re
from datetime import datetime

MONTHS = {
    "january": "01",
    "jan": "01",
    "february": "02",
    "feb": "02",
    "march": "03",
    "mar": "03",
    "april": "04",
    "apr": "04",
    "may": "05",
    "june": "06",
    "jun": "06",
    "july": "07",
    "jul": "07",
    "august": "08",
    "aug": "08",
    "september": "09",
    "sep": "09",
    "sept": "09",
    "october": "10",
    "oct": "10",
    "november": "11",
    "nov": "11",
    "december": "12",
    "dec": "12",
}

DOB_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%b %d %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%d-%B-%Y",
    "%d-%b-%Y",
)

NAME_RE = re.compile(
    r"(?:my\s+name\s+is|name\s+is|i\s+am|i'm|this\s+is)\s+"
    r"([A-Za-z][A-Za-z.'\- ]{0,60}?)"
    r"(?=\s*(?:,|\.|$| and | dob | date | phone | email | ssn | policy | last\s+four))",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
POLICY_RE = re.compile(r"\bPOL[- ]?\d+\b", re.IGNORECASE)
CASE_ID_RE = re.compile(r"\bCL[- ]?\d+\b", re.IGNORECASE)
CASE_TYPES = ("healthcare", "dental", "auto")
CASE_STATUSES = ("denied", "open", "closed")
CLAIM_YEAR_RE = re.compile(r"\b(20[2-3]\d)\b")
PHONE_RE = re.compile(
    r"(?:phone|mobile|cell)(?:\s+number)?\s*(?:is|:)?\s*([+\d][\d\s().-]{6,20})",
    re.IGNORECASE,
)
BARE_PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d\s().-]{9,18}\d)(?!\d)")
ISO_DATE_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")
LAST4_RE = re.compile(
    r"(?:ssn|social(?:\s+security)?|last\s+four|last\s+4|id\s+last\s+four)[^\d]{0,20}(\d{4})",
    re.IGNORECASE,
)
DAY_MONTH_YEAR_RE = re.compile(
    r"\b(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})\b",
    re.IGNORECASE,
)
MONTH_DAY_YEAR_RE = re.compile(
    r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)


def normalize_dob(value: str | None) -> str | None:
    if not value:
        return None
    text = " ".join(value.strip().replace(",", " ").split())
    match = DAY_MONTH_YEAR_RE.search(text)
    if match:
        day, month, year = match.groups()
        return f"{year}-{MONTHS[month.lower()]}-{int(day):02d}"
    match = MONTH_DAY_YEAR_RE.search(text)
    if match:
        month, day, year = match.groups()
        return f"{year}-{MONTHS[month.lower()]}-{int(day):02d}"
    for fmt in DOB_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text.lower() or None


def normalize_phone(value: str | None) -> str | None:
    if not value:
        return None
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) < 7:
        return None
    return digits[-10:] if len(digits) >= 10 else digits


def parse_identity_utterance(user_text: str) -> dict[str, str]:
    text = user_text or ""
    found: dict[str, str] = {}

    email = EMAIL_RE.search(text)
    if email:
        found["email"] = email.group(0)

    policy = POLICY_RE.search(text)
    if policy:
        found["policy_number"] = policy.group(0).upper().replace(" ", "")

    case_id = CASE_ID_RE.search(text)
    if case_id:
        found["case_id"] = case_id.group(0).upper().replace(" ", "")

    lowered = text.lower()
    for case_type in CASE_TYPES:
        if case_type in lowered:
            found["case_type"] = case_type
            break
    for status in CASE_STATUSES:
        if re.search(rf"\b{status}\b", lowered):
            found["status"] = status
            break
    claim_context = bool(found.get("case_id")) or any(
        token in lowered for token in ("claim", "denied", "denial")
    )
    if claim_context:
        for month, numeral in MONTHS.items():
            if re.search(rf"\b{month}\b", lowered):
                found["month"] = datetime.strptime(numeral, "%m").strftime("%B")
                break
        year = CLAIM_YEAR_RE.search(text)
        if year:
            found["year"] = year.group(1)

    last4 = LAST4_RE.search(text)
    if last4:
        found["ssn_last4"] = last4.group(1)

    phone = PHONE_RE.search(text)
    explicit_phone = phone is not None
    if phone is None:
        phone = BARE_PHONE_RE.search(text)
    if phone:
        raw = phone.group(1) if phone.lastindex else phone.group(0)
        if not ISO_DATE_RE.match(raw.strip()):
            digits = "".join(ch for ch in raw if ch.isdigit())
            if explicit_phone or len(digits) >= 10:
                normalized = normalize_phone(raw)
                if normalized:
                    found["phone"] = normalized

    dob = normalize_dob(text) if DAY_MONTH_YEAR_RE.search(text) or MONTH_DAY_YEAR_RE.search(text) else None
    if not dob:
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
            match = re.search(r"\b\d{1,4}[-/]\d{1,2}[-/]\d{2,4}\b", text)
            if match:
                dob = normalize_dob(match.group(0))
                if dob:
                    break
    if dob and re.fullmatch(r"\d{4}-\d{2}-\d{2}", dob):
        found["dob"] = dob

    name = None
    for match in NAME_RE.finditer(text):
        cleaned = " ".join(match.group(1).strip(" .,;:").split())
        if cleaned and cleaned.lower() not in {"the policyholder", "policyholder", "calling"}:
            name = cleaned
            break
    if name:
        found["name"] = name

    return found
