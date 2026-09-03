from __future__ import annotations

import importlib
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from app.config import GROQ_FALLBACK_MODEL, GROQ_MODEL, LOGFIRE_TOKEN, resolve_groq_api_key

import logfire

from app import main as main_mod
from app.agents.nodes import extractor as extractor_mod
from app.agents.nodes import post_process as post_process_mod
from app.agents.nodes import process_case as process_case_mod
from app.agents.nodes import resolve_intent as resolve_intent_mod
from app.agents.nodes import state as state_mod
from app.agents.nodes import verify_id as verify_id_mod
from app.agents.tools import email as email_tool_mod

state_mod = importlib.reload(state_mod)
email_tool_mod = importlib.reload(email_tool_mod)
extractor_mod = importlib.reload(extractor_mod)
verify_id_mod = importlib.reload(verify_id_mod)
resolve_intent_mod = importlib.reload(resolve_intent_mod)
post_process_mod = importlib.reload(post_process_mod)
process_case_mod = importlib.reload(process_case_mod)
main_mod = importlib.reload(main_mod)

INITIAL_STATE = state_mod.INITIAL_STATE
iter_reply = main_mod.iter_reply
WORD_DELAY_SECONDS = 0.02


def _typewriter(text: str):
    words = text.split()
    for index, word in enumerate(words):
        yield word if index == 0 else f" {word}"
        time.sleep(WORD_DELAY_SECONDS)


GREETING = (
    "Hello, this is insurance claims support. I can help with an existing claim. "
    "First I need to verify your identity with any 3 of: full name, date of birth, "
    "phone number, email address, or the last four digits of your SSN."
)


def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": GREETING}]
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = deepcopy(INITIAL_STATE)


def _reset_chat() -> None:
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]
    st.session_state.agent_state = deepcopy(INITIAL_STATE)


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
        help="Paste your own Groq key to test. If empty, the app uses GROQ_API_KEY from .env.",
    )
    st.caption(f"Model: `{GROQ_MODEL}`")
    st.caption(f"Fallback: `{GROQ_FALLBACK_MODEL}`")
    st.caption(
        "Logfire: sending to the cloud"
        if LOGFIRE_TOKEN
        else "Logfire: console only unless LOGFIRE_TOKEN is set or you ran `logfire projects use`"
    )
    st.button("Reset conversation", on_click=_reset_chat)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about a claim…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Working…", expanded=False)
        streamed = {"state": st.session_state.agent_state, "started": False}

        def _chunks():
            try:
                for event in iter_reply(
                    user_text=prompt,
                    state=st.session_state.agent_state,
                    api_key=resolve_groq_api_key(groq_api_key),
                ):
                    if event.kind == "status" and event.text:
                        status.update(label=event.text, state="running")
                    elif event.kind == "text" and event.text.strip():
                        if streamed["started"]:
                            yield " "
                        streamed["started"] = True
                        yield from _typewriter(event.text.strip())
                    elif event.kind == "done" and event.state is not None:
                        streamed["state"] = event.state
                status.update(label="Done", state="complete")
            except Exception as exc:
                logfire.error("chat turn failed", error_type=type(exc).__name__)
                status.update(label="Turn failed", state="error")
                raise

        try:
            answer = st.write_stream(_chunks()) or ""
            st.session_state.agent_state = streamed["state"]
        except Exception as exc:
            answer = f"I could not complete that turn. {exc}"
            st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
