"""FastAPI backend — HTTP routes into the LangGraph SOP agent."""

from __future__ import annotations

import json
from typing import Any

import logfire
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents.nodes.state import INITIAL_STATE, AgentState
from app.config import BACKEND_URL, GROQ_FALLBACK_MODEL, GROQ_MODEL, resolve_groq_api_key
from app.main import SOP_FLOW, iter_reply, reply
from app.memory import patch as memory_patch

GREETING = (
    "Hi, I can help with an existing insurance claim. "
    "I will confirm your identity first, then we can look at your case. "
    "You can start with your name, or any other details you have handy."
)

app = FastAPI(
    title="SOP Claims Agent",
    description="Backend routes for the SOP-guided insurance claims agent.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_text: str = Field(..., min_length=1)
    state: dict[str, Any] | None = None
    api_key: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str
    state: dict[str, Any]
    phase: str | None = None
    verified: bool = False


class SessionResponse(BaseModel):
    greeting: str
    state: dict[str, Any]
    sop_flow: list[str]
    model: str
    fallback_model: str


def _public_state(state: AgentState | dict[str, Any] | None) -> dict[str, Any]:
    source = state or INITIAL_STATE
    payload: dict[str, Any] = {}
    for key, default in INITIAL_STATE.items():
        value = source.get(key, default)
        if key == "messages":
            payload[key] = []
        else:
            payload[key] = value
    return payload


def _agent_state(raw: dict[str, Any] | None) -> AgentState:
    return _public_state(raw)  # type: ignore[return-value]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "sop-claims-agent"}


@app.get("/v1/session", response_model=SessionResponse)
def create_session() -> SessionResponse:
    return SessionResponse(
        greeting=GREETING,
        state=_public_state(memory_patch(INITIAL_STATE, last_agent_reply=GREETING)),
        sop_flow=list(SOP_FLOW),
        model=GROQ_MODEL,
        fallback_model=GROQ_FALLBACK_MODEL,
    )


@app.post("/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    api_key = resolve_groq_api_key(request.api_key)
    if not api_key:
        raise HTTPException(status_code=400, detail="A Groq API key is required.")
    try:
        answer, updated = reply(
            user_text=request.user_text,
            state=_agent_state(request.state),
            api_key=api_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logfire.error("chat route failed", error_type=type(exc).__name__)
        raise HTTPException(status_code=500, detail="The agent could not complete that turn.") from exc
    public = _public_state(updated)
    return ChatResponse(
        reply=answer,
        state=public,
        phase=public.get("phase"),
        verified=bool(public.get("verified")),
    )


@app.post("/v1/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    api_key = resolve_groq_api_key(request.api_key)
    if not api_key:
        raise HTTPException(status_code=400, detail="A Groq API key is required.")

    def events():
        try:
            for event in iter_reply(
                user_text=request.user_text,
                state=_agent_state(request.state),
                api_key=api_key,
            ):
                payload: dict[str, Any] = {
                    "kind": event.kind,
                    "text": event.text,
                    "node": event.node,
                }
                if event.kind == "done" and event.state is not None:
                    payload["state"] = _public_state(event.state)
                    payload["phase"] = event.state.get("phase")
                    payload["verified"] = bool(event.state.get("verified"))
                yield f"data: {json.dumps(payload, default=str)}\n\n"
        except Exception as exc:
            logfire.error("chat stream failed", error_type=type(exc).__name__)
            yield f"data: {json.dumps({'kind': 'error', 'text': str(exc)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/json/version", include_in_schema=False)
def devtools_probe() -> dict[str, str]:
    """Cursor/Chrome DevTools hits this; it is not an app route."""
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "sop-claims-agent",
        "backend": BACKEND_URL,
        "docs": "/docs",
        "routes": ["/health", "/v1/session", "/v1/chat", "/v1/chat/stream"],
    }
