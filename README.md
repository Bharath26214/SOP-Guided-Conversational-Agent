# SOP Guided Conversational Agent

## Introduction

This project is an insurance claims support chatbot that follows a fixed Standard Operating Procedure (SOP). Phase order is enforced in code with LangGraph, not left to the model: the agent verifies the caller’s identity first, then resolves which claim they mean, answers from the claim file, and finally offers an email summary.

Each user turn runs through guardrails and fact extraction, then the current SOP node. Nodes gather structured facts; a separate speak step phrases one natural-language reply from those facts. The model does not invent claim IDs, denial reasons, or amounts.

**SOP phases:** `VERIFY_ID` → `RESOLVE_INTENT` → `PROCESS_CASE` → `POST_PROCESS` → `DONE` (with optional human escalation).

The UI is Streamlit; the backend is FastAPI. Callers paste their own Groq API key to run the demo.

## Implementation


| Layer                 | Stack                                                                            |
| --------------------- | -------------------------------------------------------------------------------- |
| Orchestration         | LangGraph (phase routing, silent handoffs)                                       |
| Phrasing / extraction | LangChain + Groq (`gpt-oss-120b`, fallback `gpt-oss-20b`)                        |
| Guardrails            | NeMo Guardrails (off-topic, jailbreak, injection)                                |
| API                   | FastAPI (`/v1/session`, `/v1/chat/stream`)                                       |
| UI                    | Streamlit chatbot                                                                |
| Data                  | Local JSON policyholders / claims; email summaries written to `app/DATA/outbox/` |
| Observability         | Logfire (optional; runs without a token)                                         |


See [ARCHITECTURE.md](ARCHITECTURE.md) for the node graph.

## Walkthrough Video

**[https://screenrec.com/share/ntrQ3hdLXq](https://screenrec.com/share/ntrQ3hdLXq)**

## Live demo

**[https://sop-claims-agent.onrender.com](https://sop-claims-agent.onrender.com)**

**It takes time to render for the first time. You can go through the walkthrough video until then.**

You must paste a **Groq API key** in the sidebar. Create one here:  
[https://console.groq.com/keys](https://console.groq.com/keys)

## Run locally



### 1. Install

```bash
poetry install
cp .env.example .env
```

A Logfire token is **not required**. Leave `LOGFIRE_TOKEN` empty in `.env`. The app runs without sending traces to Logfire.

### 2. Start the backend

```bash
poetry run uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
```

API: [http://127.0.0.1:8000](http://127.0.0.1:8000)

### 3. Start the UI (second terminal)

```bash
poetry run streamlit run ui/app.py
```

UI: [http://localhost:8501](http://localhost:8501)

### 4. Use the app

1. Open the Streamlit UI.
2. Paste your Groq API key in the sidebar ([get a key](https://console.groq.com/keys)).
3. Optionally copy the sample test case from the sidebar into the chat.



## Future scope

- **Post-train on SOP trajectories:** Fine-tune or post-train the model on different conversation trajectories and phase sequences so phrasing and turn-taking better match real SOP paths.
- **Vector DB at scale:** As policyholder and claim data grow, store embeddings in a vector database for faster retrieval of cases, documents, and similar past interactions.
- **User login instead of per-call ID checks:** Authorize each user once through login/session so the agent does not need to re-verify identity on every conversation.

Please reach out to me if the deliverables aren't accessible or you have any questions.