# SOP Guided Conversational Agent

Insurance claims support agent. Identity is verified first (`VERIFY_ID`), then later SOP phases handle the case.

## Run locally

```bash
poetry install
cp .env.example .env
```

Set `GROQ_API_KEY` in `.env`, then start the FastAPI backend and the Streamlit UI:

```bash
poetry run uvicorn app.api:app --reload --port 8000
poetry run streamlit run ui/app.py
```

The API is at http://localhost:8000 (`/docs`, `/v1/session`, `/v1/chat`, `/v1/chat/stream`).  
The UI is at http://localhost:8501 and routes every turn to that backend.

## Logfire observability

Traces cover each chat turn, extraction, identity verification, and Groq HTTP calls. LangChain/LangGraph spans are exported through OpenTelemetry. The database is queried only after the caller has given identity details, and those lookups are not instrumented as raw SQL spans.

Custom spans do **not** record utterance text, SSN, DOB, phone, or email values. They record phase, verification outcome, matched PII **field names**, and which identity fields were present.

LangChain's OpenTelemetry export can still attach prompts and model output. Scrubbing is on for SSN/DOB-like keys. Use a Logfire project you trust, and do not log production PII to a shared demo project.

### 1. Create a project

1. Sign in at [logfire.pydantic.dev](https://logfire.pydantic.dev).
2. Create a project (for example `sop-claims-agent`).

### 2. Connect this repo

**Local development** (browser login, no token in `.env`):

```bash
poetry run logfire auth
poetry run logfire projects use sop-claims-agent
```

**CI or another machine** (write token):

1. In the Logfire project, open **Settings → Write tokens** and create a token.
2. Add it to `.env`:

```bash
LOGFIRE_TOKEN=your_write_token
LOGFIRE_SERVICE_NAME=sop-claims-agent
LOGFIRE_CONSOLE=true
```

Leave `LOGFIRE_TOKEN` empty to run without sending data. Spans still print to the terminal when `LOGFIRE_CONSOLE=true`.

Set `LOGFIRE_CONSOLE=false` if you do not want terminal trace output (useful in Streamlit).

### 3. Confirm it is working

Start the app, send a chat message, then open the project **Live** view. You should see a trace named `agent.reply` with child spans:

- `extractor.extract_and_apply` / `extractor.extract_turn` / `extractor.llm_attempt`
- LangChain model spans (Groq)
- `verify_id.run`

The first process start also logs `Logfire configured`.

### How it is wired

`app/observability.py` calls `logfire.configure()` once, instruments `httpx`, and enables LangSmith OTEL (`LANGSMITH_OTEL_ENABLED`, `LANGSMITH_OTEL_ONLY`, `LANGSMITH_TRACING`) **before** LangChain is imported. `app/config.py` loads `.env` and then configures Logfire on startup.
