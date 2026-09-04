"""Detect caller affect so SOP replies can de-escalate without skipping gates."""

from __future__ import annotations

from typing import Literal

Affect = Literal["calm", "frustration", "anger", "anxiety", "confusion", "refusal"]
Tone = Literal["warm", "empathetic", "formal", "escalate"]

PERSUTION_LIMIT = 3

_ANGER = (
    "ridiculous",
    "unacceptable",
    "this is stupid",
    "are you kidding",
    "furious",
    "outraged",
    "waste of time",
    "incompetent",
    "useless",
)
_FRUSTRATION = (
    "already told",
    "already gave",
    "already provided",
    "i already",
    "how many times",
    "this is taking too long",
    "just tell me",
    "just give me",
    "i've said",
    "i have told",
    "why won't you",
    "this is frustrating",
    "frustrated",
    "enough already",
)
_ANXIETY = (
    "worried",
    "anxious",
    "scared",
    "please hurry",
    "i need this today",
    "urgent",
    "desperate",
    "i'm stressed",
)
_CONFUSION = (
    "i don't understand",
    "i do not understand",
    "confused",
    "what do you mean",
    "why is this",
    "why is that",
    "why do you need",
    "why are you asking",
    "i'm lost",
)
_REFUSAL = (
    "i'm not giving",
    "i am not giving",
    "won't give",
    "will not give",
    "refuse to",
    "not sharing",
    "not going to verify",
    "skip verification",
    "just look it up",
)
_HUMAN = (
    "supervisor",
    "manager",
    "real person",
    "transfer me",
    "speak to someone",
    "talk to a person",
    "connect me with a human",
    "speak to a human",
    "human representative",
    "i want a human",
    "get me a human",
)
_CLAIM_PUSH = (
    "denied",
    "denial",
    "claim",
    "reimbursement",
    "why was",
    "status",
)


def detect_affect(user_text: str) -> Affect:
    text = (user_text or "").lower()
    if not text:
        return "calm"
    if any(cue in text for cue in _ANGER):
        return "anger"
    if any(cue in text for cue in _REFUSAL):
        return "refusal"
    if any(cue in text for cue in _FRUSTRATION):
        return "frustration"
    if any(cue in text for cue in _ANXIETY):
        return "anxiety"
    if any(cue in text for cue in _CONFUSION):
        return "confusion"
    return "calm"


def wants_human(user_text: str) -> bool:
    text = (user_text or "").lower()
    return any(cue in text for cue in _HUMAN)


def pushing_claim_before_verify(user_text: str) -> bool:
    text = (user_text or "").lower()
    return any(cue in text for cue in _CLAIM_PUSH)


def tone_for(affect: Affect, attempts: int, *, force_human: bool) -> Tone:
    if force_human or (affect in {"anger", "frustration", "refusal"} and attempts >= PERSUTION_LIMIT):
        return "escalate"
    if affect == "calm":
        return "warm"
    if attempts >= 2:
        return "formal"
    return "empathetic"
