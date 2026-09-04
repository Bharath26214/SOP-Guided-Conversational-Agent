from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx
import streamlit as st

from app.config import BACKEND_URL, GROQ_FALLBACK_MODEL, GROQ_MODEL

WORD_DELAY_SECONDS = 0.02
GROQ_KEY_URL = "https://console.groq.com/keys"
SAMPLE_TEST_CASE = (
    "I'm the policyholder. My name is Margaret Chen, policy POL-9921. "
    "I'm calling about my denied healthcare claim from January. "
    "DOB is 1985-03-15, SSN last four is 4472."
)
FALLBACK_GREETING = (
    "Hi, I can help with an existing insurance claim. "
    "I will confirm your identity first, then we can look at your case. "
    "You can start with your name, or any other details you have handy."
)
_API_PROC: subprocess.Popen | None = None


def _backend_base() -> str:
    return (os.environ.get("BACKEND_URL") or BACKEND_URL).rstrip("/")


def _backend_reachable(base: str) -> bool:
    try:
        response = httpx.get(f"{base}/health", timeout=1.5)
        return response.status_code == 200
    except Exception:
        return False


def _ensure_backend() -> None:
    """Start a local API process when the configured backend is loopback and down."""
    global _API_PROC
    base = _backend_base()
    if _backend_reachable(base):
        return
    parsed = urlparse(base)
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost"}:
        return
    port = parsed.port or 8000
    if _API_PROC is not None and _API_PROC.poll() is None:
        return
    _API_PROC = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(ROOT_DIR),
        env={**os.environ, "BACKEND_URL": base},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(lambda: _API_PROC is not None and _API_PROC.terminate())
    for _ in range(90):
        if _backend_reachable(base):
            return
        time.sleep(0.5)


def _typewriter(text: str):
    first = True
    for part in text.splitlines(keepends=True):
        if part.startswith("\n") or part == "":
            yield part
            first = True
            continue
        words = part.split()
        suffix = "\n" if part.endswith("\n") else ""
        for index, word in enumerate(words):
            yield word if first else f" {word}"
            first = False
            time.sleep(WORD_DELAY_SECONDS)
        if suffix:
            yield suffix
            first = True


def _backend_session() -> dict:
    response = httpx.get(f"{_backend_base()}/v1/session", timeout=10.0)
    response.raise_for_status()
    return response.json()


def _iter_backend(user_text: str, state: dict, api_key: str):
    with httpx.Client(timeout=180.0) as client:
        with client.stream(
            "POST",
            f"{_backend_base()}/v1/chat/stream",
            json={"user_text": user_text, "state": state, "api_key": api_key},
        ) as response:
            response.raise_for_status()
            buffer = ""
            for chunk in response.iter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    for line in block.splitlines():
                        if line.startswith("data: "):
                            yield json.loads(line[6:])


def _init_session() -> None:
    if "messages" in st.session_state and "agent_state" in st.session_state:
        return
    _ensure_backend()
    try:
        session = _backend_session()
        st.session_state.backend_ok = True
        st.session_state.backend_error = ""
        greeting = session.get("greeting") or FALLBACK_GREETING
        st.session_state.agent_state = session.get("state") or {}
    except Exception as exc:
        st.session_state.backend_ok = False
        st.session_state.backend_error = str(exc)
        greeting = FALLBACK_GREETING
        st.session_state.agent_state = {}
    st.session_state.messages = [{"role": "assistant", "content": greeting}]


def _reset_chat() -> None:
    st.session_state.pop("messages", None)
    st.session_state.pop("agent_state", None)
    _init_session()


st.set_page_config(page_title="SOP Claims Agent", page_icon="📋", layout="centered")
_init_session()

st.title("SOP Guided Claims Agent")
st.caption("Insurance claims support · identity first, then the case")

with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input(
        "Groq API key",
        type="password",
        placeholder="gsk_...",
        help="Required. Paste a Groq API key to use the demo.",
    )
    st.markdown(f"Need a key? [Create one at Groq]({GROQ_KEY_URL})")
    st.caption(f"Model: `{GROQ_MODEL}`")
    st.caption(f"Fallback: `{GROQ_FALLBACK_MODEL}`")
    st.button("Reset conversation", on_click=_reset_chat)
    st.divider()
    phase = (st.session_state.get("agent_state") or {}).get("phase") or "VERIFY_ID"
    st.markdown(f"**Phase:** `{phase}`")
    st.divider()
    st.markdown("**Sample test case**")
    st.caption("Copy into the chat to try the demo flow.")
    st.code(SAMPLE_TEST_CASE, language=None)

if not groq_api_key.strip():
    st.warning(
        f"Enter your Groq API key in the sidebar to chat. "
        f"You can create one at [console.groq.com/keys]({GROQ_KEY_URL})."
    )

if not st.session_state.get("backend_ok", True):
    st.error(
        f"Cannot reach the FastAPI backend at {BACKEND_URL}. "
        "Start it with `poetry run uvicorn app.api:app --reload --port 8000`. "
        f"({st.session_state.get('backend_error')})"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(
    "Ask about a claim…",
    disabled=not groq_api_key.strip() or not st.session_state.get("backend_ok", True),
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Working…", expanded=False)
        streamed = {"state": st.session_state.agent_state, "started": False}

        def _chunks():
            try:
                for event in _iter_backend(prompt, st.session_state.agent_state, groq_api_key.strip()):
                    kind = event.get("kind")
                    text = (event.get("text") or "").strip()
                    if kind == "status" and text:
                        status.update(label=text, state="running")
                    elif kind == "text" and text:
                        if streamed["started"]:
                            yield "\n\n"
                        streamed["started"] = True
                        yield from _typewriter(text)
                    elif kind == "done" and event.get("state") is not None:
                        streamed["state"] = event["state"]
                    elif kind == "error":
                        raise RuntimeError(text or "The backend could not complete that turn.")
                status.update(label="Done", state="complete")
            except Exception:
                status.update(label="Turn failed", state="error")
                raise

        try:
            answer = st.write_stream(_chunks()) or ""
            st.session_state.agent_state = streamed["state"]
        except Exception as exc:
            answer = f"I could not complete that turn. {exc}"
            st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
